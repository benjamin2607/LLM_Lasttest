"""
Model control for the dashboard: list, load, reload and unload LM Studio models, and keep
named model configurations.

Loading and unloading use LM Studio's native REST API (``/api/v1/models``,
``/api/v1/models/load``, ``/api/v1/models/unload``). A model configuration has two parts:

* load settings, applied when the model is loaded: context length, parallel requests,
  flash attention, evaluation batch size, KV cache on the GPU, number of experts;
* request settings, sent with every request of an "LM Studio native" load test:
  system prompt, maximum response tokens, temperature, top P, top K, min P, repeat and
  presence penalty, thinking.

The *active* configuration is what the next LM Studio load test uses. It and the saved
configurations are stored in ``model_configs.json`` in the project folder (settings only,
no user data). Settings LM Studio's REST API does not offer (GPU offload, context overflow,
unloading on idle) are left to LM Studio.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

LOAD_SETTINGS = {  # name: (type, min, max); booleans use (bool, None, None)
    "context_length": (int, 256, 1048576), "parallel": (int, 1, 64), "flash_attention": (bool, None, None),
    "eval_batch_size": (int, 1, 65536), "offload_kv_cache_to_gpu": (bool, None, None), "num_experts": (int, 1, 512),
}
REQUEST_SETTINGS = {
    "max_tokens": (int, 0, 131072), "temperature": (float, 0, 2), "top_p": (float, 0, 1), "top_k": (int, 0, 1000),
    "min_p": (float, 0, 1), "repeat_penalty": (float, 0.5, 3), "presence_penalty": (float, -2, 2),
}
REASONING_OPTIONS = ("off", "on", "low", "medium", "high", "xhigh")
MAX_SYSTEM_PROMPT = 8000
MAX_CONFIGS = 50
LOAD_TIMEOUT_S = 900  # Large models can take minutes to load.


def _clean(settings, rules):
    """Validate settings against ``rules``; empty values are left out. Returns (values, error)."""
    if settings in (None, ""):
        settings = {}
    if not isinstance(settings, dict):
        return None, "settings must be an object"
    out = {}
    for key, (kind, lo, hi) in rules.items():
        value = settings.get(key)
        if value in (None, ""):
            continue
        try:
            if kind is bool:
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
            else:
                value = kind(value)
        except (TypeError, ValueError):
            return None, f"{key} must be a number"
        if kind is not bool and not lo <= value <= hi:
            return None, f"{key} must be between {lo} and {hi}"
        out[key] = value
    return out, None


def clean_request_settings(settings):
    """Request settings incl. system prompt and thinking. Returns (values, error)."""
    out, error = _clean(settings, REQUEST_SETTINGS)
    if error:
        return None, error
    settings = settings or {}
    prompt = settings.get("system_prompt") or ""
    if not isinstance(prompt, str):
        return None, "the system prompt must be text"
    if len(prompt) > MAX_SYSTEM_PROMPT:
        return None, f"the system prompt must not exceed {MAX_SYSTEM_PROMPT} characters"
    if prompt.strip():
        out["system_prompt"] = prompt
    reasoning = str(settings.get("reasoning") or "")
    if reasoning and reasoning not in REASONING_OPTIONS:
        return None, "thinking must be one of " + ", ".join(REASONING_OPTIONS)
    if reasoning:
        out["reasoning"] = reasoning
    return out, None


def request_arguments(settings):
    """Command-line options of lmstudio_user.py for the request settings of a configuration."""
    flags = {"max_tokens": "--max-tokens", "temperature": "--temperature", "top_p": "--top-p", "top_k": "--top-k",
             "min_p": "--min-p", "repeat_penalty": "--repeat-penalty", "presence_penalty": "--presence-penalty",
             "reasoning": "--reasoning", "system_prompt": "--system-prompt"}
    # "--flag=value" keeps values that start with "-" (e.g. a negative penalty) from being read as options.
    return [f"{flags[k]}={v}" for k, v in (settings or {}).items() if k in flags]


class ModelControl:
    """LM Studio models, the active model configuration and saved configurations."""

    def __init__(self, lmstudio_url, token, store_path, busy=lambda: False):
        self.url = lmstudio_url.rstrip("/")
        self.token = token
        self.store_path = Path(store_path)
        self.busy = busy  # True while a load test runs: models must not change then.
        self.lock = threading.Lock()
        self.progress = {"phase": "idle"}
        self._cache = (0.0, None)
        self.store = {"active": None, "configs": []}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        # The file may have been edited by hand: keep only well-formed entries.
        if isinstance(data, dict):
            if isinstance(data.get("active"), dict):
                self.store["active"] = data["active"]
            self.store["configs"] = [c for c in data.get("configs") or [] if isinstance(c, dict) and c.get("name")]

    # --- LM Studio API ---
    def _api(self, method, path, body=None, timeout=10):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read()).get("error", {}).get("message")
            except (ValueError, AttributeError):
                message = None
            raise RuntimeError(message or f"LM Studio answered HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RuntimeError(f"LM Studio is not reachable at {self.url}") from exc

    def models(self, max_age=3.0):
        """Language models on disk with their load state, merged by model key; cached for a few seconds."""
        at, cached = self._cache
        if cached is not None and time.time() - at < max_age:
            return cached
        merged = {}
        for m in self._api("GET", "/api/v1/models").get("models", []):
            if m.get("type") not in ("llm", "vlm"):
                continue
            entry = merged.setdefault(m["key"], {
                "key": m["key"], "name": m.get("display_name") or m["key"], "params": m.get("params_string"),
                "quantization": (m.get("quantization") or {}).get("name"),
                "size_gb": round(m["size_bytes"] / 1e9, 1) if m.get("size_bytes") else None,
                "max_context": m.get("max_context_length"),
                "reasoning": ((m.get("capabilities") or {}).get("reasoning") or None), "instances": []})
            entry["instances"] += [{"id": i.get("id"), "config": i.get("config") or {}} for i in m.get("loaded_instances") or []]
        models = sorted(merged.values(), key=lambda e: (not e["instances"], e["name"].lower()))
        self._cache = (time.time(), models)
        return models

    def loaded_model(self):
        """Key of the model the next LM Studio load test uses: the active one if loaded, else the first loaded."""
        try:
            loaded = [m["key"] for m in self.models() if m["instances"]]
        except RuntimeError:
            return None
        active = (self.store.get("active") or {}).get("model")
        return active if active in loaded else (loaded[0] if loaded else None)

    def request_settings(self, model):
        """Request settings of the active configuration if it belongs to ``model``."""
        active = self.store.get("active") or {}
        return dict(active.get("request") or {}) if active.get("model") == model else {}

    # --- State for the dashboard ---
    def state(self):
        try:
            models, error = self.models(), None
        except RuntimeError as exc:
            models, error = [], str(exc)
        with self.lock:
            return {"models": models, "error": error, "active": self.store.get("active"),
                    "configs": self.store.get("configs") or [], "progress": dict(self.progress),
                    "busy": bool(self.busy()), "settings": {"load": list(LOAD_SETTINGS), "request": list(REQUEST_SETTINGS)}}

    def _save(self):
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.store, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.store_path)

    def _set(self, **progress):
        with self.lock:
            self.progress = {"at": time.time(), **progress}
        print(f"[monitor] model: {progress.get('message', progress.get('phase'))}")

    # --- Actions ---
    def apply(self, params):
        """
        Apply a model choice from the dashboard. ``action`` is "keep" (use the loaded model as it is),
        "adjust" (load or reload with the given settings) or "defaults" (load or reload with LM Studio's
        defaults). Loading runs in the background; returns (ok, message).
        """
        if self.busy():
            return False, "a load test is running – models cannot be changed now"
        if self.progress.get("phase") == "working":
            return False, "a model action is already in progress"
        action, key = params.get("action"), str(params.get("model") or "")
        try:
            model = next((m for m in self.models(max_age=0) if m["key"] == key), None)
        except RuntimeError as exc:
            return False, str(exc)
        if model is None:
            return False, "unknown model"
        if action not in ("keep", "adjust", "defaults"):
            return False, "unknown action"
        load, error = _clean(params.get("load"), LOAD_SETTINGS) if action == "adjust" else ({}, None)
        request, error2 = clean_request_settings(params.get("request")) if action == "adjust" else ({}, None)
        if error or error2:
            return False, error or error2
        if load.get("context_length") and model.get("max_context") and load["context_length"] > model["max_context"]:
            return False, f"context length exceeds the model's maximum of {model['max_context']}"
        active = {"model": key, "action": action, "load": load, "request": request, "at": time.time(),
                  "config": str(params.get("config") or "")[:80] or None}
        if action == "keep":
            if not model["instances"]:
                return False, "the model is not loaded"
            with self.lock:
                self.store["active"] = active
                self._save()
            self._set(phase="done", message=f"{model['name']} is used as loaded")
            return True, "the loaded model is used as it is"
        self._set(phase="working", message=f"{'reloading' if model['instances'] else 'loading'} {model['name']} …")
        threading.Thread(target=self._load, args=(model, load, active), daemon=True).start()
        return True, "loading started"

    def _load(self, model, load, active):
        try:
            for instance in model["instances"]:
                self._api("POST", "/api/v1/models/unload", {"instance_id": instance["id"]}, timeout=120)
            result = self._api("POST", "/api/v1/models/load", {"model": model["key"], **load, "echo_load_config": True},
                               timeout=LOAD_TIMEOUT_S)
            with self.lock:
                # Only the settings this card offers; LM Studio also returns e.g. the whole prompt template.
                loaded = {k: v for k, v in (result.get("load_config") or {}).items() if k in LOAD_SETTINGS}
                self.store["active"] = {**active, "loaded_config": loaded}
                self._save()
            took = result.get("load_time_seconds")
            self._set(phase="done", message=f"{model['name']} loaded" + (f" in {took:.0f} s" if took else ""))
        except RuntimeError as exc:
            self._set(phase="error", message=f"loading {model['name']} failed: {exc}")
        finally:
            self._cache = (0.0, None)

    def unload(self, key):
        """Unload all instances of a model. Returns (ok, message)."""
        if self.busy():
            return False, "a load test is running – models cannot be changed now"
        try:
            model = next((m for m in self.models(max_age=0) if m["key"] == key), None)
            if model is None or not model["instances"]:
                return False, "the model is not loaded"
            for instance in model["instances"]:
                self._api("POST", "/api/v1/models/unload", {"instance_id": instance["id"]}, timeout=120)
        except RuntimeError as exc:
            return False, str(exc)
        finally:
            self._cache = (0.0, None)
        self._set(phase="done", message=f"{model['name']} unloaded")
        return True, "model unloaded"

    def save_config(self, params):
        """Save (or overwrite) a named configuration. Returns (ok, message)."""
        name = params.get("name")
        name = " ".join(name.split())[:80] if isinstance(name, str) else ""
        if not name:
            return False, "enter a name for the configuration"
        load, error = _clean(params.get("load"), LOAD_SETTINGS)
        request, error2 = clean_request_settings(params.get("request"))
        if error or error2:
            return False, error or error2
        entry = {"name": name, "model": str(params.get("model") or ""), "load": load, "request": request,
                 "saved": time.time()}
        with self.lock:
            configs = [c for c in self.store["configs"] if c.get("name") != name]
            if len(configs) >= MAX_CONFIGS:
                return False, f"at most {MAX_CONFIGS} configurations can be saved"
            self.store["configs"] = sorted(configs + [entry], key=lambda c: c["name"].lower())
            self._save()
        return True, f"configuration “{name}” saved"

    def delete_config(self, name):
        with self.lock:
            before = len(self.store["configs"])
            self.store["configs"] = [c for c in self.store["configs"] if c.get("name") != name]
            if len(self.store["configs"]) == before:
                return False, "configuration not found"
            self._save()
        return True, "configuration deleted"
