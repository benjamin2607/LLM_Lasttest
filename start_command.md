# Manual Locust run

Requires `ANYTHING_API_KEY` and `WORKSPACE_SLUG` (see [PARAMETERS.md](PARAMETERS.md#1-environment-variables)).

```bash
python -m locust -f main.py --host http://localhost:3001
```

With the Locust web UI (http://localhost:8089) you choose the number of users and start the test
yourself. Options for the simulated users are passed on the same command line, e.g.:

```bash
python -m locust -f main.py --host http://localhost:3001 --mode chat --wait-min 10 --wait-max 30
```

All options of `main.py` are described in [PARAMETERS.md](PARAMETERS.md#3-simulated-users--mainpy).
For automated capacity testing use `auto_loadtest.py` (see the README).
