#! /usr/bin/bash

set -eufx -o pipefail

bench_cpu () {
    local MAX_PRIME="$1"
    local THREADS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
    local DURATION="$2"
    for i in "${THREADS[@]}"
    do
        local FILENAME=sysbench_cpu_"$i"_"$DURATION"_sec_"$MAX_PRIME"_prime.txt
        sysbench --threads="$i" \
                 --test=cpu \
                 --cpu-max-prime="$MAX_PRIME" \
                 --histogram=on \
                 --percentile=99 \
                 --time="$DURATION" \
                 --report-interval=1 \
                 run | tee "$FILENAME"
        python parse_sysbench.py "$FILENAME"
        sleep 15
    done
}

bench_cpu ${1:-2000} ${2:-20}
