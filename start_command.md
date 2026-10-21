# Manual Locust run

## Campus workstation
```bash
python -m locust -f main.py --host http://localhost:3001
```

### Options
| Option | Type | Default | Description |
|---|---|---|---|
| `--wait-min` | float | 5.0 | Minimum think time in seconds |
| `--wait-max` | float | 15.0 | Maximum think time in seconds |
| `--mode` | `query` \| `chat` | `query` | `query` = answer from workspace documents only (RAG), `chat` = regular chat |
| `--request-timeout` | float | 300.0 | Timeout per request in seconds |
| `--messages-per-thread` | int | 1 | Questions per thread before a new one is started (1 = new thread per question, 0 = never) |
| `--run-id` | str | start time | Identifier included in every thread name |
| `--keep-threads` | flag | off | Do not delete threads after use |
| `--refusal-text` | str | AnythingLLM default | Query mode refusal response configured in the workspace |
| `--refusal-log` | path | – | JSON Lines file for refusal examples |

For automated capacity testing see `auto_loadtest.py` in the README.
