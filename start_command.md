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
| `--messages-per-thread` | int | 5 | Start a new thread after this many messages (0 = never) |

For automated capacity testing see `auto_loadtest.py` in the README.
