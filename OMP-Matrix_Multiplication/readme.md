### Khoa Nguyen - CS360 - Lab6 - 03/26/2022

Two main files in this folder are `mmm-naive.c` and `lab6.c`.
1. `mmm-naive.c` is the original code, but I also added some time records in different parts of the code to calculate the memory allocation (mallocTime), matrices initialization (initTime), and matrix multiplication (matmulTimeForAll and matmulTimeForOne). I also change the datatype of the matrices from int to long long int to capture the correct results.
2. `lab6.c` parallelized the matrix multiplication implementation in the `mmm-naive.c`. Also, I change arguments processing to getopt(), in which `-i` for `itemsPerDimension`, `-r` for `repeats`, `-p` for `platform`, `-c` for `coreSpeed`, and `-t` for `thread_count`.

To run, in Terminal:
1. Run `make clean`.
2. Run `make all`.
3. Run `./mmm-naive` to execute the serial matrix multiplication.
4. Run `./lab6` to execute the parallelized matrix multiplication (add options if needed).
