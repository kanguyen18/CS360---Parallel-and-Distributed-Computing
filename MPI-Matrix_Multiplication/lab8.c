// Khoa Nguyen
// CS360 - Lab 8 - 04/20/2022
// lab8.c

// This code is adapted from mmm-naive.c, where I added getopt() for arguments processing and parallelized the matmul implementation with MPI.
// I also use long long int for the matrices to correctly store the results.

// Compile:
// mpicc -g -std=c99 -Wall --pedantic -O0 lab8.c -o lab8

// Run these command lines for results:
// mpiexec -n 10 ./lab8
// mpiexec -n 10 --hostfile whedon-hosts --map-by node ./lab8

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>
#include <getopt.h>
#include <assert.h>

int main (int argc, char *argv[]) {
  int itemsPerDimension = 1000;

  // getopt() for arguments processing
  int option;
  while ((option = getopt(argc, argv, ":i:")) != -1) {
      switch(option) {
          case 'i':
              itemsPerDimension = atoi(optarg);
          break;

          default :
            itemsPerDimension = 1000;
      }
  }

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  long long int (*mOne)[itemsPerDimension] = malloc(sizeof(long long int[itemsPerDimension][itemsPerDimension]));
  long long int (*mTwo)[itemsPerDimension] = malloc(sizeof(long long int[itemsPerDimension][itemsPerDimension]));
  long long int (*mResult)[itemsPerDimension] = malloc(sizeof(long long int[itemsPerDimension][itemsPerDimension]));

  int itemsPerNode = (itemsPerDimension*itemsPerDimension)/world_size;
  int h, k, i, j, q;
  long long int sum;
  MPI_Barrier(MPI_COMM_WORLD);

  // node 0 populate the matrices and share with other nodes using MPI_Bcast
  double bcast_start, bcast_finish;
  bcast_start = MPI_Wtime();
  if (world_rank == 0) {
    for (i = 0; i < itemsPerDimension; i++) {
      for(j = 0; j < itemsPerDimension; j++) {
        mOne[i][j] = 333333;
        mTwo[i][j] = 777777;
        mResult[i][j] = 0; } }

    MPI_Bcast(mOne, itemsPerDimension*itemsPerDimension, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mTwo, itemsPerDimension*itemsPerDimension, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(mOne, itemsPerDimension*itemsPerDimension, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mTwo, itemsPerDimension*itemsPerDimension, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  bcast_finish = MPI_Wtime();
  printf("Process %d > Bcast time = %e seconds\n", world_rank, bcast_finish - bcast_start);
  //NOTE: The longest time is the total Bcast time for all nodes.

// During calculation
  double matmul_start, matmul_finish;
  matmul_start = MPI_Wtime();
  if (world_rank == world_size-1) {
    for (h = world_rank*itemsPerNode; h < itemsPerDimension*itemsPerDimension; h++) {
      sum = 0;
      i = h/itemsPerDimension;
      j = h%itemsPerDimension;
      for (k = 0; k < itemsPerDimension; k++) {
        sum = sum + mOne[i][k] * mTwo[k][j];
      }
      mResult[i][j] = sum;
    }
  }
  else {
    for (h = world_rank*itemsPerNode; h < (world_rank+1)*itemsPerNode; h++) {
      sum = 0;
      i = h/itemsPerDimension;
      j = h%itemsPerDimension;
      for (k = 0; k < itemsPerDimension; k++) {
        sum = sum + mOne[i][k] * mTwo[k][j];
      }
      mResult[i][j] = sum;
    }
  }
  matmul_finish = MPI_Wtime();
  // Showing the running time of doing matmul for each node
  printf("Process %d > Matmul time = %e seconds\n", world_rank, matmul_finish - matmul_start);
  MPI_Barrier(MPI_COMM_WORLD);

  // After calculation: Assembling the results
  double assemble_start, assemble_finish;
  assemble_start = MPI_Wtime();
  if (world_rank != 0) {
    // Sending back results to head node 0
     if (world_rank == world_size-1) {
       // special case of last process
       h = world_rank*itemsPerNode;
       i = h/itemsPerDimension;
       j = h%itemsPerDimension;
       MPI_Send(&(mResult[i][j]), (itemsPerDimension*itemsPerDimension) - h, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
     } else {
       h = world_rank*itemsPerNode;
       i = h/itemsPerDimension;
       j = h%itemsPerDimension;
       MPI_Send(&(mResult[i][j]), itemsPerNode, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
     }
  } else {
    // Node 0 receive the sub-results and assemble them
     for (q = 1; q < world_size-1; q++) {
        h = q*itemsPerNode;
        i = h/itemsPerDimension;
        j = h%itemsPerDimension;
        MPI_Recv(&(mResult[i][j]), itemsPerNode, MPI_LONG_LONG_INT, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     }
     // special case of last process
     h = (world_size-1)*itemsPerNode;
     i = h/itemsPerDimension;
     j = h%itemsPerDimension;
     MPI_Recv(&(mResult[i][j]), (itemsPerDimension*itemsPerDimension) - (world_size-1)*itemsPerNode, MPI_LONG_LONG_INT, world_size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  assemble_finish = MPI_Wtime();
  printf("Process %d > Assemble time = %e seconds\n", world_rank, assemble_finish - assemble_start);
  //NOTE: The longest time is the total assembling time for all nodes.

  // display mResult, for testing and debugging.
  // if (world_rank == 0) {
  //   for (i = 0; i < 2; i++) {
  //     for(j = 0; j < 2; j++) {
  //       fprintf(stderr, "mResult[%d][%d] = %lld\n", i, j, mResult[i][j]); }
  //     }
  //   }

  free(mOne); free(mTwo); free(mResult);
  MPI_Finalize();
  exit(0);
}
