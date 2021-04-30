#!/bin/bash
set -e

log_dir=$(pwd)/log
mkdir -p ${log_dir}

./predict_resnet50_impl.sh 1 0 2 ${log_dir}/log0.log &
./predict_resnet50_impl.sh 2 1 2 ${log_dir}/log1.log

wait

