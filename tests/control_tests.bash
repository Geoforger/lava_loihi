for i in {1..50};
    echo "Iteration $i - $(date)"
    do python control_integration_test.py;
    sleep 1
done;