#!/usr/bin/env bash
# Start the NASDAQ agent in a detached tmux session.
# Usage:  ./start.sh          — start
#         ./start.sh stop     — stop
#         ./start.sh status   — show status
#         ./start.sh logs     — tail live logs

SESSION="nasdaq-agent"
LOG="/home/user/Pawan/nasdaq_agent/logs/agent.log"
DIR="/home/user/Pawan/nasdaq_agent"

case "${1:-start}" in
  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "Already running (tmux session: $SESSION)"
      exit 0
    fi
    mkdir -p "$(dirname "$LOG")"
    tmux new-session -d -s "$SESSION" \
      "cd '$DIR' && gunicorn -c gunicorn.conf.py main:app 2>&1 | tee -a '$LOG'"
    echo "Started — tmux session: $SESSION"
    echo "Logs:   $LOG"
    echo "Stop:   ./start.sh stop"
    ;;
  stop)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Stopped." || echo "Not running."
    ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "Running ✓" || echo "Stopped ✗"
    ;;
  logs)
    tail -f "$LOG"
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs}"
    exit 1
    ;;
esac
