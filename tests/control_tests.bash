# !/bin/bash
for i in $(seq 1 50)
do
    echo "Iteration $i - $(date)"
    python ~/Documents/PhD/lava_loihi/tests/control_integration_test.py
    sleep 1
done