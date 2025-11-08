#!/bin/bash

# Run NCU profiling in tmux session
# Usage: ./run_ncu_tmux.sh <case_number>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <case_number>"
    echo "Example: $0 01"
    exit 1
fi

CASE=$1
SESSION_NAME="ncu_profile_case${CASE}"

# Check if tmux session already exists
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "Error: tmux session '${SESSION_NAME}' already exists"
    echo "Please use one of the following commands:"
    echo "  tmux attach -t ${SESSION_NAME}  # attach to existing session"
    echo "  tmux kill-session -t ${SESSION_NAME}  # kill existing session"
    exit 1
fi

echo "=========================================="
echo "Creating tmux session: ${SESSION_NAME}"
echo "Running NCU profiling for case${CASE}"
echo "=========================================="
echo ""
echo "Profiling will run in background. You can:"
echo "  1. Attach to session to view progress:"
echo "     tmux attach -t ${SESSION_NAME}"
echo ""
echo "  2. Detach from session (keep running):"
echo "     Press Ctrl+B then D"
echo ""
echo "  3. List all sessions:"
echo "     tmux ls"
echo ""
echo "  4. Kill session:"
echo "     tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "=========================================="
echo "Creating session..."

# Create new tmux session and execute profiling
tmux new-session -d -s ${SESSION_NAME} -c $(pwd)

# Execute profiling command in session
tmux send-keys -t ${SESSION_NAME} "echo 'Starting NCU Profiling - case${CASE}'" C-m
tmux send-keys -t ${SESSION_NAME} "echo 'Time: \$(date)'" C-m
tmux send-keys -t ${SESSION_NAME} "echo '========================================'" C-m
tmux send-keys -t ${SESSION_NAME} "./profile_ncu.sh ${CASE}" C-m

echo ""
echo "✓ tmux session '${SESSION_NAME}' created and started"
echo ""
echo "Attach to session to view progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
