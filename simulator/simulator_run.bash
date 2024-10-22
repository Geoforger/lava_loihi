#!/bin/bash
declare -a arr=("v" "b")

# Cycle through modes
for m in "${arr[@]}"
do
    echo "Testing Mode: $m"
    # Cycle through all textures
    for l in $(seq 0 9)
    do
        echo "Testing Label: $l"
        # Implement all tests using these settings
        for i in $(seq 1 50)
        do
            echo "Iteration $i - $(date)"
            # Pass both mode and label as input to the python script
            printf "$m\n$l\n" | python /home/george/Documents/lava_loihi/simulator/control_simulator.py
            sleep 0.2
        done
    done
done