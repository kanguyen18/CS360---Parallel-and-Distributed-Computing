// Run code with:
// mpirun -np 8 -hostfile whedon-hosts --map-by node lab10-prime -t 4
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "check_prime_brute_force.c"
#include "check_prime_brute_force.h"
#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {

  int my_rank, world_size, provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  long long int total_count = 0; // Save the result on total_count
  long long int c = 1000000; // The default range is between 3 and 1000000.
  int thread_count = 16; // The default number of thread is 16.

  // Handle command line arguments with getopt
  int option;
  char* endPtr;

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

  // Hybrid section
  long long int sub_count;
  #pragma omp parallel for num_threads(thread_count) reduction(+:sub_count) schedule(static,1)
  for (long long int i=3+my_rank; i<=c; i+=world_size) {
    if (check_prime_brute_force(i) == 1) {
        sub_count++;
    }
  }

  // Node 0 gather and print results
  long long int *client_count = NULL;
  if (my_rank == 0) {
    client_count = (long long int *)malloc(sizeof(long long int) * world_size);
  }

  MPI_Gather(&sub_count, 1, MPI_LONG_LONG_INT, client_count, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  if (my_rank == 0) {
    for (int i=0; i<world_size; i++) {
      total_count += client_count[i];
    }

    printf("There are %llu primes between 3 and %llu.\n", total_count, c); // Print the result.
  }

  MPI_Finalize();
  return(0);

}
