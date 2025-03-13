#!/bin/bash

# Part B1: execute the PP implementation with micro-batches

cd "$(dirname "$0")" || return
START_TIME=$SECONDS

for ((i = 0; i < 3; i = i + 1)); do
    touch "out$i.txt"
    (
        sleep 1
        python -u "s01_b1_microbatches.py" $i >"out$i.txt"
    ) &
done

wait
echo "Elapsed time (s): $((SECONDS - START_TIME))"
