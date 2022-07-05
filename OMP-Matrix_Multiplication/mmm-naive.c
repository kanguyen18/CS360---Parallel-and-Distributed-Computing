// mmm-naive.c - naive implementation of matrix matrix multiplication.
// Originally authored by charliep, updated by barbeda Spring 2022
// I added and adjusted the time records to measure the running time for memory allocation, initialization, and matmul
// I also use long long int for the matrices to correctly store the results.
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>

int main (int argc, char *argv[]) {
  int itemsPerDimension, repeats, counter, i, j, k;
  long long int sum;
  // The code implements the matmul of two `itemsPerDimension` by `itemsPerDimension` matrices for a `repeats` #of MatrixItemsPerSecond
  // Then, it records the running times for analysis
  unsigned long totalBytes;
  struct timeval matmul_startTime, matmul_stopTime, malloc_startTime, malloc_stopTime, init_startTime, init_stopTime;
  double matmulTimeForAll, matmulTimeForOne, MatrixKBPerSecond, mallocTime, initTime;
  char platform[32] = "";
  float coreSpeed = 1.0;

  // arguments processing
  if (argc == 5) {
    sscanf(argv[1], "%d", &itemsPerDimension);
    sscanf(argv[2], "%d", &repeats);
    sscanf(argv[3], "%f", &coreSpeed);
    sscanf(argv[4], "%s", platform); }

  else if (argc == 4) {
    sscanf(argv[1], "%d", &itemsPerDimension);
    sscanf(argv[2], "%d", &repeats);
    sscanf(argv[3], "%f", &coreSpeed);
    strcpy(platform, "unknown"); }

  else if (argc == 3) {
    sscanf(argv[1], "%d", &itemsPerDimension);
    sscanf(argv[2], "%d", &repeats);
    coreSpeed = 1.0;
    strcpy(platform, "unknown"); }

  else if (argc == 2) {
    sscanf(argv[1], "%d", &itemsPerDimension);
    repeats = 10;
    coreSpeed = 1.0;
    strcpy(platform, "unknown"); }

  else {
    itemsPerDimension = 1000;
    repeats = 10;
    coreSpeed = 1.0;
    strcpy(platform, "unknown"); }

  fprintf(stderr, "# itemsPerDimension: %d, repeats: %d, platform: %s, coreSpeed: %.3f\n", itemsPerDimension, repeats, platform, coreSpeed);

  totalBytes = itemsPerDimension * itemsPerDimension * sizeof(int);

// get time for malloc of the three matrices
  if (gettimeofday(&malloc_startTime, NULL) != 0) {
		perror("gettimeofday() malloc_startTime failed");
		exit(-1); }

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

  // do the math repeats number of times. do not
  for (counter = 0; counter < repeats; counter++) {
    sum = 0;

    for (i = 0; i < itemsPerDimension; i++) {
      for (j = 0; j < itemsPerDimension; j++) {
        for (k = 0; k < itemsPerDimension; k++) {
          sum = sum + mOne[i][k] * mTwo[k][j]; }

        mResult[i][j] = sum;
        sum = 0; } }
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

  // display running times
  fprintf(stderr, "# platform, totalBytes, matmulTimeForAll, matmulTimeForOne, MatrixKBPerSecond, mallocTime, initTime\n");
  fprintf(stderr, "%s, %lu, %.3lf, %.3lf, %.3lf, %.3lf, %.3lf\n", platform, totalBytes, matmulTimeForAll, matmulTimeForOne, MatrixKBPerSecond, mallocTime, initTime);

  // free memory
  free(mOne); free(mTwo); free(mResult);
  exit(0);
}
