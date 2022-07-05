// Khoa Nguyen
// CS360 - Lab 5 primes - 03/04/2022
// lab5-primes.c

#include <stdio.h>
#include <stdlib.h>
#include "check_prime_brute_force.c"
#include "check_prime_brute_force.h"
#include <getopt.h>
#include "omp.h"
#include <time.h>

int main(int argc, char *argv[]){
    int option;
    long long int count = 0;
    long long int c = 1000000; // The default range is between 3 and 1000000.
    char* endPtr;
    int thread_count = 16; // The default number of thread is 16.
    int chunk_size = 1;

    // The while loop handle the options in the command line.
    while ((option = getopt(argc, argv, ":e:t:c:")) != -1) {
        switch(option) {
            case 'e':
                c = strtoll(optarg, &endPtr, 10);
                if (c<3){
                    printf("Invalid input.\n");
                    exit(1);
                }
            break;

            case 't':
                thread_count = atof(optarg);
                if (thread_count<0){
                    printf("Invalid input.\n");
                    exit(1);
                }
            break;

            case 'c':
                chunk_size = atof(optarg);
                if (chunk_size<0){
                    printf("Invalid input.\n");
                    exit(1);
                }
            break;

            default :
                c = 1000000;
                thread_count = 16;
                chunk_size = 1;
        }
    }

    printf("The range is between 3 and %llu.\n", c); // Print the range.
    printf("The number of threads is %d.\n", thread_count); // Print the number of threads.

    printf("-----------------DEFAULT-----------------\n");
    time_t begin_default = time(NULL); // Start counting time.
    #pragma omp parallel for num_threads(thread_count) reduction(+:count)
    for (long long int i=3; i < c+1 ; i++)
    {
        if (check_prime_brute_force(i) == 1){
            count++;
        }
    }
    time_t end_default = time(NULL);
    printf("There are %llu primes between 3 and %llu.\n", count, c); // Print the result.
    printf("The elapsed time is %ld seconds.\n", (end_default-begin_default)); // Print the total run-walltime.

    printf("-----------------STATIC-----------------\n");
    count=0;
    time_t begin_static = time(NULL); // Start counting time.
    #pragma omp parallel for num_threads(thread_count) reduction(+:count) schedule(static,chunk_size)
    for (long long int i=3; i < c+1 ; i++)
    {
        if (check_prime_brute_force(i) == 1){
            count++;
        }
    }
    time_t end_static = time(NULL);
    printf("There are %llu primes between 3 and %llu.\n", count, c); // Print the result.
    printf("The elapsed time is %ld seconds.\n", (end_static-begin_static)); // Print the total run-walltime.

    printf("-----------------DYNAMIC-----------------\n");
    count=0;
    time_t begin_dynamic = time(NULL); // Start counting time.
    #pragma omp parallel for num_threads(thread_count) reduction(+:count) schedule(dynamic,chunk_size)
    for (long long int i=3; i < c+1 ; i++)
    {
        if (check_prime_brute_force(i) == 1){
            count++;
        }
    }
    time_t end_dynamic = time(NULL);
    printf("There are %llu primes between 3 and %llu.\n", count, c); // Print the result.
    printf("The elapsed time is %ld seconds.\n", (end_dynamic-begin_dynamic)); // Print the total run-walltime.

    printf("-----------------GUIDED-----------------\n");
    count=0;
    time_t begin_guided = time(NULL); // Start counting time.
    #pragma omp parallel for num_threads(thread_count) reduction(+:count) schedule(guided)
    for (long long int i=3; i < c+1 ; i++)
    {
        if (check_prime_brute_force(i) == 1){
            count++;
        }
    }
    time_t end_guided = time(NULL);
    printf("There are %llu primes between 3 and %llu.\n", count, c); // Print the result.
    printf("The elapsed time is %ld seconds.\n", (end_guided-begin_guided)); // Print the total run-walltime.

    return 0;
}
