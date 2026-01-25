#!/bin/bash
# Run experiments in background with logging

cd "$(dirname "$0")"

LOG_FILE="experiment_run_$(date +%Y%m%d_%H%M%S).log"

echo "Starting experiments..."
echo "Log file: $LOG_FILE"
echo "Monitor with: tail -f $LOG_FILE"

nohup python run_experiments.py > "$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Experiments running in background."
