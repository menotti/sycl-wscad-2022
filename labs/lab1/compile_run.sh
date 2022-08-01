#source /opt/intel/oneapi/setvars.sh
dpcpp -o bin/bitonic-sort lab/bitonic-sort.cpp
for i in {10..29}; do bin/bitonic-sort $i 42; done
