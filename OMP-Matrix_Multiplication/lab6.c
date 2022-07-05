// Khoa Nguyen
// CS360 - Lab 6 - 03/26/2022
// lab6.c

/*
    This code is adapted from mmm-naive.c, where I added getopt() for arguments processing,
recorded the running time of each main section, and parallelized the matmul implementation.
I also use long long int for the matrices to correctly store the results.
*/
// The code implements the matmul of two `itemsPerDimension` by `itemsPerDimension` matrices for a `repeats` #of MatrixItemsPerSecond.
// Then, it records the running times for analysis.

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include "omp.h"
#include <getopt.h>

int main (int argc, char *argv[]) {
  int counter, i, j, k, h;
  long long int sum;
  unsigned long totalBytes;
  struct timeval matmul_startTime, matmul_stopTime, malloc_startTime, malloc_stopTime, init_startTime, init_stopTime;
  double matmulTimeForAll, matmulTimeForOne, MatrixKBPerSecond, mallocTime, initTime;
  // Default values
  int itemsPerDimension = 1000;
  int repeats = 10;
  char platform[32] = "unknown";
  float coreSpeed = 1.0;
  int thread_count = 16;
  // getopt() for arguments processing
  int option;
  while ((option = getopt(argc, argv, ":i:r:c:t:p:")) != -1) {
      switch(option) {
          case 'i':
              itemsPerDimension = atoi(optarg);
          break;

          case 'r':
              repeats = atoi(optarg);
          break;

          case 'c':
              coreSpeed = atof(optarg);
          break;

          case 't':
              thread_count = atoi(optarg);
          break;

          case 'p':
              strcpy(platform, optarg);
          break;

          default :
            itemsPerDimension = 1000;
            repeats = 10;
            strcpy(platform, "unknown");
            coreSpeed = 1.0;
            thread_count = 16;
      }
  }

  fprintf(stderr, "# itemsPerDimension: %d, repeats: %d, platform: %s, coreSpeed: %.3f, threadCount: %d\n", itemsPerDimension, repeats, platform, coreSpeed, thread_count);

  totalBytes = itemsPerDimension * itemsPerDimension * sizeof(int);

// get time for malloc of the three matrices
  if (gettimeofday(&malloc_startTime, NULL) != 0) {
		perror("gettimeofday() malloc_startTime failed");
		exit(-1); }
  // malloc for three matrices
  long long int (*mOne)[itemsPerDimension] = malloc(sizeof(long long int[itemsPerDimension][itemsPerDimension]));
  long long int (*mTwo)[itemsPerDimension] = malloc(sizeof(long long int[itemsPerDimension][itemsPerDimension]));
  long long int (*mResult)[itemsPerDimension] = malloc(sizeof(long long int[itemsPerDimension][itemsPerDimension]));

  if (gettimeofday(&malloc_stopTime, NULL) != 0) {
		perror("gettimeofday() malloc_stopTime failed");
		exit(-1); }

  mallocTime = (double)(malloc_stopTime.tv_sec - malloc_startTime.tv_sec) +
                   (double)((malloc_stopTime.tv_usec - malloc_startTime.tv_usec) *
                   (double)0.000001);

  if ((mOne == NULL) || (mTwo == NULL) || (mResult == NULL)) {
    perror("initial malloc() of mOne, mTwo, and/or mResult failed");
    exit(-1); }

// get time for initialize the three matrices
  if (gettimeofday(&init_startTime, NULL) != 0) {
    perror("gettimeofday() init_startTime failed");
    exit(-1); }

  // initialize the three matrices
  for (i = 0; i < itemsPerDimension; i++) {
    for(j = 0; j < itemsPerDimension; j++) {
      mOne[i][j] = 333333;
      mTwo[i][j] = 777777;
      mResult[i][j] = 0; } }

  if (gettimeofday(&init_stopTime, NULL) != 0) {
    perror("gettimeofday() init_stopTime failed");
    exit(-1); }

  initTime = (double)(init_stopTime.tv_sec - init_startTime.tv_sec) +
                   (double)((init_stopTime.tv_usec - init_startTime.tv_usec) *
                   (double)0.000001);

// get time for doing the matmul
	if (gettimeofday(&matmul_startTime, NULL) != 0) {
		perror("gettimeofday() startTime failed");
		exit(-1); }

  // do the matmul repeats number of times.
  for (counter = 0; counter < repeats; counter++) {

    // PARALELLIZE the matmul process
    #pragma omp parallel for num_threads(thread_count) private(k, i, j, sum, h)
    for (h=0; h < itemsPerDimension*itemsPerDimension; h++) {
      sum = 0;
      i = h/itemsPerDimension;
      j = h%itemsPerDimension;
      for (k = 0; k < itemsPerDimension; k++) {
        sum = sum + mOne[i][k] * mTwo[k][j];
      }
      mResult[i][j] = sum;
    }

  }

	if (gettimeofday(&matmul_stopTime, NULL) != 0) {
		perror("gettimeofday() stopTime failed");
		exit(-1); }

	matmulTimeForAll = (double)(matmul_stopTime.tv_sec - matmul_startTime.tv_sec) +
  					       (double)((matmul_stopTime.tv_usec - matmul_startTime.tv_usec) *
  					       (double)0.000001);

  // factor out repeats, generate rate in MatrixItemsPerSecond
  matmulTimeForOne = matmulTimeForAll / (double)repeats;
  MatrixKBPerSecond = ((double)totalBytes / (double)1024.0) / matmulTimeForOne;

  // display a portion of mResult, for testing and debugging, enabled via gcc -DDISPLAY ...
  #ifdef DISPLAY
    for (i = 0; i < 2; i++) {
      for(j = 0; j < 2; j++) {
        fprintf(stderr, "mResult[%d][%d] = %lld\n", i, j, mResult[i][j]); }
      }
  #endif
  // print out the resulting running times
  fprintf(stderr, "# platform, totalBytes, matmulTimeForAll, matmulTimeForOne, MatrixKBPerSecond, mallocTime, initTime\n");
  fprintf(stderr, "%s, %lu, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf\n", platform, totalBytes, matmulTimeForAll, matmulTimeForOne, MatrixKBPerSecond, mallocTime, initTime);
  // free memory
  free(mOne); free(mTwo); free(mResult);
  exit(0);
}
