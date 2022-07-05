### Khoa Nguyen - CS360 - Lab10 - 05/14/2022

To run, in Terminal:
1. Run `make clean`.
2. Run `make all`.
3. Run `mpirun -np n -hostfile whedon-hosts --map-by node lab10-prime -t k -e x` to find the number of primes between 3 and x, using n nodes, in which each node has k more threads.
4. Run `mpirun -np n -hostfile whedon-hosts --map-by node lab10-curve -a x -b y -n z -t k`, where x, y, and z are integers, to find the integral of the function f from x to y using trap rule and z trapezoids, paralleling with n nodes, in which each has k more threads.
