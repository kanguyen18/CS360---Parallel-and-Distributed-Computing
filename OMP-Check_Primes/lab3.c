// Khoa Nguyen
// CS360 - Lab 3 - 02/21/2022
// lab3.c

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
    
    time_t begin = time(NULL); // Start counting time.

    // The while loop handle the options in the command line.
    while ((option = getopt(argc, argv, ":e:t:")) != -1) {
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

            default :
                c = 1000000;
                thread_count = 16;
        }
    }
    
    printf("The range is between 3 and %llu.\n", c); // Print the range.
    printf("The number of threads is %d.\n", thread_count); // Print the number of threads.
    
    
    // Parallelism strategy:
    // The idea is to split the range in half, and use parallel for to assign evenly the works for each thread.
    // Then, if a thread got the small numbers portion, it will have to handle the large numbers portion in the second half of the range, so that the sum of all the numbers being handle by a thread would be as close as possible.
    // For example, if the range is (3,10) and we have 2 threads, thread 1 will take (3,4) and (9,10), while thread 2 will take (5,6) and (7,8).
    
    // get half the range
    long long int range = c-3+1;
    long long int whole;
    if (range%2 == 0){
        whole = range/2;
    }
    else {
        whole = (range/2) + 1; // in case range is not divisible by 2
    }
    
    #pragma omp parallel num_threads(thread_count) 
    {
        double wtime = omp_get_wtime(); // start counting time for each thread
        
        int thread_id = omp_get_thread_num();
        long long int mini_count = 0;
        #pragma omp for nowait // Use parallel for to evenly split the works in the first half of the range, use nowait to assist with measuring the running time for each thread. Without nowait, all threads would run for the same time.
        for (long long int i=3; i < whole+3 ; i++)
        {
            if (check_prime_brute_force(i) == 1){
                mini_count++;
            }
            
            // also, split the works in the second half, depending on the index in the first half
            if (i!=(c-i+3)){ // avoid duplicate when the range is not divisible by 2.
                if (check_prime_brute_force(c-i+3) == 1){
                    mini_count++;
                }
            }
        }
        wtime = omp_get_wtime() - wtime;
        printf( "Time taken by thread %d is %f\n", thread_id, wtime ); // print running time of each thread
        
        #pragma omp critical // handle race condition
        {
        count += mini_count;
        } 
    }
    
    printf("There are %llu primes between 3 and %llu.\n", count, c); // Print the result.
    
    time_t end = time(NULL);
    printf("The elapsed time is %ld seconds.\n", (end-begin)); // Print the total run-walltime.
    return 0;
}








