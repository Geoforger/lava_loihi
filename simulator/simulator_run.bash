# !/bin/bash
for i in $(seq 1 1000)
do
    echo "Iteration $i - $(date)"
    python /home/george/Documents/lava_loihi/simulator/control_simulator.py
    sleep 0.5
done

