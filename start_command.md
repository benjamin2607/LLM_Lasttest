## Workstation am Campus:
python -m locust -f "PFAD_ZUR_SKRIPT_DATEI" --host http://localhost:3001

### Flags
"--wait-min", type=float, default=5.0, help="Minimale Denkpause in Sekunden"
"--wait-max", type=float, default=15.0, help="Maximale Denkpause in Sekunden"
"--mode", choices=["query", "chat"], default="query", help="query = nur RAG/Dokumente, chat = normaler Chat"
"--request-timeout", type=float, default=300.0, help="Timeout pro Anfrage in Sekunden"
"--messages-per-thread", type=int, default=5, help="Nach so vielen Nachrichten startet ein neuer Thread (0 = nie)"