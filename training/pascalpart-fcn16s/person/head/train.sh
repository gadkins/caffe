#!/bin/bash
# Usage: ./train.sh <GPU/CPU-device-ID> <resume-option> <iteration>
# Train from scratch example: ./train.sh 0
# Resume training from iteration 16000 example: ./train.sh 0 -resume 16000
# N.B. "2>&1 | tee "log/$(date +%Y-%m-%d_%H:%M.log)" combines stderr 
# and stdout into a training log with date/time

python solve.py "$1" "$2" "$3" 2>&1 | tee "log/$(date +%Y-%m-%d_%H:%M.log)"
