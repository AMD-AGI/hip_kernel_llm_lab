#!/bin/bash
# Quick helper to inspect sandbox error logs.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ERROR_DIR=${HIP_ERROR_LOG_DIR:-"$ROOT_DIR/runtime/error_log"}

echo "=== HIP Kernel Error Log Viewer ==="
echo "Error directory: $ERROR_DIR"
echo ""

if ! ls "$ERROR_DIR"/*_error.log >/dev/null 2>&1 && ! ls "$ERROR_DIR"/*_prewarm_error.log >/dev/null 2>&1; then
    echo "No error logs found"
    exit 0
fi

echo "1. Error file counts"
echo "-------------------"
for log in "$ERROR_DIR"/*.log; do
    [ -f "$log" ] || continue
    count=$(grep -c "Kernel:" "$log" 2>/dev/null || echo "0")
    echo "  $(basename "$log"): $count entries"
done
echo ""

echo "2. Error stage summary"
echo "----------------------"
grep "Stage:" "$ERROR_DIR"/*.log 2>/dev/null | \
    awk -F'Stage: ' '{print $2}' | \
    sort | uniq -c | sort -rn || echo "  No stage entries"
echo ""

echo "3. Recent errors"
echo "----------------"
grep -h "\[20" "$ERROR_DIR"/*.log 2>/dev/null | tail -n 10 || echo "  No recent records"
echo ""

if [ "${1:-}" != "" ]; then
    echo "4. Filtered stage: $1"
    echo "--------------------"
    grep -A 10 "Stage: $1" "$ERROR_DIR"/*.log 2>/dev/null || echo "  No entries for stage $1"
    echo ""
fi

echo "Tips:"
echo "  - filter compile failures: grep 'COMPILATION_FAILED' $ERROR_DIR/*.log"
echo "  - inspect one kernel: grep 'kernel_name' $ERROR_DIR/*.log"
echo "  - view one full file: less $ERROR_DIR/<kernel>_error.log"
