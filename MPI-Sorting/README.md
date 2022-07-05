### Khoa Nguyen - CS360 - Lab8 - 04/21/2022

To run, in Terminal:
1. Run `make clean`.
2. Run `make all`.   
4. Run `mpiexec -n 1 ./lab9 -n <num_elements_per_node>` to execute the code serially. To reproduce the test cases, run with `num_elements_per_node` = 8000, 80000, and 800000.
5. Run `mpiexec -n 8 ./lab9 -n <num_elements_per_node>` to execute the code with 8 processes on one machine. To reproduce the test cases, run with `num_elements_per_node` = 1000, 10000, and 100000.
6. Run `mpiexec -n 8 --hostfile whedon-hosts --map-by node ./lab9 -n <num_elements_per_node>` to execute the code with distributed system. To reproduce the test cases, run with `num_elements_per_node` = 1000, 10000, and 100000.
