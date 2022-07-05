### Khoa Nguyen - CS360 - Lab8 - 04/21/2022

Two main files in this folder are `mmm-naive.c` and `lab8.c`.
1. `mmm-naive.c` is the original code, but I also added some time records in different parts of the code to calculate the memory allocation (mallocTime), matrices initialization (initTime), and matrix multiplication (matmulTimeForAll and matmulTimeForOne). I also change the datatype of the matrices from int to long long int to capture the correct results.
2. `lab8.c` parallelized the matrix multiplication implementation in the `mmm-naive.c` using mpi. Also, I change arguments processing to getopt(), in which `-i` is for `itemsPerDimension`.

To run, in Terminal:
1. Run `make clean`.
2. Run `make all`.   
3. Run `./mmm-naive` to execute the serial matrix multiplication.
4. Run `mpiexec -n 10 ./lab8 -i <itemsPerDimension>` to execute the mpi matrix multiplication, map by slot option.
5. Run `mpiexec -n 10 --hostfile whedon-hosts --map-by node ./lab8 -i <itemsPerDimension>` to execute the mpi matrix multiplication, map by node option.
