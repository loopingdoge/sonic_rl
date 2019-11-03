#!/bin/bash

execute_test() {
    local test_id=$1
    local rank=$2
    local load_model=$3
    python test.py $test_id $rank --load-model $3 --hyper-opt
}

# Main
test_id=$1
load_model=$2

for i in {0..3}
do
    execute_test $test_id $i $load_model &
done

wait

for i in {4..7}
do
    execute_test $test_id $i $load_model &
done

wait

for i in {8..10}
do
    execute_test $test_id $i $load_model &
done

wait

python scores.py ./logs/$test_id