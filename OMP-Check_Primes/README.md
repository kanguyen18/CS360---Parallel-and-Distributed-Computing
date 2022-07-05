### Khoa Nguyen - CS360 - Lab3 - 02/21/2022

I added some print commands to make it easier to see the inputs and the parallelism performance.

To run, in Terminal:
1. Run `make clean`.
2. Run `make all`.
3. Run `./lab3 -e n -t m`, where n is an integer >= 3, to find the number of primes between 3 and n; t is the number of threads.

**Parallelism strategy:**

The idea is to split the range in half, and use `parallel for` to assign evenly the works for each thread. Then, if a thread first got the small numbers portion, it will have to handle the large numbers portion in the second half of the range, so that the sum of all the numbers being handle by a thread would be as close as possible. For example, if the range is (3,10) and we have 2 threads, thread 1 will take (3,4) and (9,10), while thread 2 will take (5,6) and (7,8).