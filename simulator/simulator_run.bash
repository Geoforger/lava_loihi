#!/bin/bash
for l in $(seq 0 10)
do
    echo "Testing label: $l"
    for i in $(seq 1 1000)
    do
        echo "Iteration $i - $(date)"
        echo "$l" | python /home/george/Documents/lava_loihi/simulator/control_simulator.py
        sleep 0.2
    done
done