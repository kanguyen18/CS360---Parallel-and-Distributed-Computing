// Run these command lines for results:
// mpiexec -n 10 ./lab9
// mpiexec -n 10 --hostfile whedon-hosts --map-by node ./lab9

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>

// Creates an array of random numbers. Each number has a value from 0 - 100
int *create_rand_nums(int num_elements) {
  int *rand_nums = (int *)malloc(sizeof(int) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = rand()%100;
  }
  return rand_nums;
}

// func to compare two number: one for ascending sort, and one for descending sort
int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

int inv_cmpfunc (const void * a, const void * b) {
   return ( *(int*)b - *(int*)a );
}

// main function
int main(int argc, char *argv[]) {
  int num_elements_per_proc = 10; // default is 10 elements per process
  // getopt() for arguments processing to input the num_elements_per_proc
  int option;
  while ((option = getopt(argc, argv, ":n:")) != -1) {
      switch(option) {
          case 'n':
              num_elements_per_proc = atoi(optarg);
          break;

          default :
            num_elements_per_proc = 10;
      }
  }

  // Seed the random number generator to get different results each time
  srand(time(NULL));

  // Start the MPI region
  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Create a random array of elements on the root process. Its total
  // size will be the number of elements per process times the number
  // of processes
  int *rand_nums = (int *)malloc(sizeof(int) * num_elements_per_proc * world_size);
  int *sorted_nums_1 = NULL; // resulting sorted array for the odd-even transposition sort
  int *sorted_nums_2 = NULL; // resulting sorted array for enumeration sort
  if (world_rank == 0) {
    rand_nums = create_rand_nums(num_elements_per_proc * world_size);
    // Un-comment for testing purpose
    // printf("The unsorted array is:\n");
    // for (int i = 0; i < num_elements_per_proc*world_size; i++) {
    //   printf("%d ", rand_nums[i]);
    // }
    // printf("\n");
    sorted_nums_1 = (int *)malloc(sizeof(int) * num_elements_per_proc*world_size);
    sorted_nums_2 = (int *)calloc(num_elements_per_proc*world_size, sizeof(int));

  }

  // allocate necessary memory in each process to receive info from head node and carry out the sorting task
  // for efficient odd-even transposition sort
  int *sub_rand_nums = (int *)malloc(sizeof(int) * num_elements_per_proc * 2);
  // for enumeration sort
  int *index_list = (int *)malloc(sizeof(int) * num_elements_per_proc);

  // Head node sends information for odd-even transposition sort
  MPI_Scatter(rand_nums, num_elements_per_proc, MPI_INT, sub_rand_nums, num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
  // Head node sends information for enumeration sort
  MPI_Bcast(rand_nums, num_elements_per_proc * world_size, MPI_INT, 0, MPI_COMM_WORLD);

  // EFFICIENT ODD-EVEN TRANSPOSITION SORT
  double oe_start, oe_finish;
  oe_start = MPI_Wtime();

  if (world_size % 2 == 0) {
    for (int phase = 0; phase < world_size-1; phase ++){
      // at even phase
      if (phase % 2 == 0) {
        // passing values
        if (world_rank % 2 == 0) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);
        } else {
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank % 2 != 0) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD);
        } else {
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // sort
        if (world_rank % 2 == 0) {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), cmpfunc);
        } else {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), inv_cmpfunc);
          qsort(sub_rand_nums, num_elements_per_proc, sizeof(int), cmpfunc);
        }
      } else { // at odd phase
        // passing values
        if ((world_rank % 2 != 0) && (world_rank != world_size-1)) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);
        } else if ((world_rank % 2 == 0) && (world_rank != 0)) {
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if ((world_rank % 2 == 0) && (world_rank != 0)) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD);
        } else if ((world_rank % 2 != 0) && (world_rank != world_size-1)) {
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // sort
        if ((world_rank % 2 == 0) && (world_rank != 0)) {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), inv_cmpfunc);
          qsort(sub_rand_nums, num_elements_per_proc, sizeof(int), cmpfunc);
        } else if ((world_rank % 2 != 0) && (world_rank != world_size-1)) {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), cmpfunc);
        }
      }
    }
  } else { //world_size is odd
    // at even phase
    for (int phase = 0; phase < world_size+2; phase ++) {
      if (phase % 2 == 0) {
        // passing values
        if ((world_rank % 2 == 0) && (world_rank != world_size-1)) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);
        } else if (world_rank % 2 != 0){
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank % 2 != 0) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD);
        } else if ((world_rank % 2 == 0) && (world_rank != world_size-1)) {
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // sort
        if ((world_rank % 2 == 0) && (world_rank != world_size-1)) {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), cmpfunc);
        } else if (world_rank % 2 != 0) {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), inv_cmpfunc);
          qsort(sub_rand_nums, num_elements_per_proc, sizeof(int), cmpfunc);
        }
      } else { // at odd phase
        if (world_rank % 2 != 0) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);
        } else if ((world_rank % 2 == 0) && (world_rank != 0)) {
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if ((world_rank % 2 == 0) && (world_rank != 0)) {
          MPI_Send(sub_rand_nums, num_elements_per_proc, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD);
        } else if (world_rank % 2 != 0) {
          MPI_Recv(&sub_rand_nums[num_elements_per_proc], num_elements_per_proc, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // sort
        if ((world_rank % 2 == 0) && (world_rank != 0)) {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), inv_cmpfunc);
          qsort(sub_rand_nums, num_elements_per_proc, sizeof(int), cmpfunc);
        } else if ((world_rank % 2 != 0) && (world_rank != world_size-1)) {
          qsort(sub_rand_nums, num_elements_per_proc*2, sizeof(int), cmpfunc);
        }
      }
    }
  }

  MPI_Gather(sub_rand_nums, num_elements_per_proc, MPI_INT, sorted_nums_1, num_elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  oe_finish = MPI_Wtime();

  if (world_rank == 0) {
    printf("Total time for efficient odd-even transposition sort = %e seconds\n", oe_finish - oe_start);
    // Un-comment for debugging
    // printf("The sorted array using odd-even transposition sort is:\n");
    // for (int i = 0; i < num_elements_per_proc*world_size; i++) {
    //   printf("%d ", sorted_nums_1[i]);
    // }
    // printf("\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ENUMERATION SORT:
  // Each process works on a part of the full array depending on its rank.
  // For each value, the process will assign it an index of 0 and increment it as it goes through
  // the full list where there are smaller values.
  // There is a small issue of equal values where they will have the same index count at the end.
  // Head node will handle the equal values with the same index at the end.
  double enum_start, enum_finish;
  enum_start = MPI_Wtime();

  for (int i = world_rank*num_elements_per_proc; i < (world_rank+1)*num_elements_per_proc; i++) {
    int index = 0;
    for (int k = 0; k < num_elements_per_proc * world_size; k++) {
      if (rand_nums[i] > rand_nums[k]) {
        index += 1;
      }
    index_list[i-world_rank*num_elements_per_proc] = index;
    }
  }
  if (world_rank != 0) {
    MPI_Send(index_list, num_elements_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  if (world_rank == 0) {
    for (int i = 0; i < num_elements_per_proc; i++) {
      sorted_nums_2[index_list[i]] = rand_nums[i];
    }
    for (int j = 1; j < world_size; j++) {
      MPI_Recv(index_list, num_elements_per_proc, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int k = j*num_elements_per_proc; k < (j+1)*num_elements_per_proc; k++) {
        sorted_nums_2[index_list[k-j*num_elements_per_proc]] = rand_nums[k];
      }
    }
    for (int h = 1; h < num_elements_per_proc*world_size; h++) {
      if (sorted_nums_2[h] == 0) {
        sorted_nums_2[h] = sorted_nums_2[h-1];
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  enum_finish = MPI_Wtime();

  if (world_rank == 0) {
    printf("Total time for enumeration sort = %e seconds\n", enum_finish - enum_start);
    // Un-comment for debugging
    // printf("The sorted array using enumeration sort is:\n");
    // for (int i = 0; i < num_elements_per_proc*world_size; i++) {
    //   printf("%d ", sorted_nums_2[i]);
    // }
    // printf("\n");
  }

  // Clean up
  free(rand_nums);
  free(sub_rand_nums);
  free(index_list);
  free(sorted_nums_1);
  free(sorted_nums_2);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return(0);
}
