#include <stdlib.h>
#include <stdio.h>

__global__ void add(int* a_arr, int* b_arr, int* c_arr) {
	int i;

	i =  threadIdx.x + blockDim.x * blockIdx.x;
  c_arr[i] = a_arr[i] + b_arr[i];

}

__host__ void usage() {
	fprintf(stderr, "Usage: vec-add <number thread blocks> <number threads per block>\n");
	exit(1);
}

int main(int argc, char** argv) {
  int size, vec_size, numThreadBlocks, numThreadsPerBlock;
  int *a_arr, *b_arr, *c_arr;
  int *g_a_arr, *g_b_arr, *g_c_arr;
  cudaError_t status = (cudaError_t)0;

  if (argc != 3)
		usage();
	if (sscanf(argv[1], "%d", &numThreadBlocks) != 1)
		usage();
	if (sscanf(argv[2], "%d", &numThreadsPerBlock) != 1)
		usage();

  size = numThreadBlocks * numThreadsPerBlock;
  /* Allocate CPU memory. */
  vec_size= numThreadBlocks * numThreadsPerBlock * sizeof(int);

  if (!(a_arr = (int*) malloc(vec_size))) {
		fprintf(stderr, "malloc() FAILED (a_arr)\n");
		exit(0);
	}

	if (!(b_arr = (int*) malloc(vec_size))) {
		fprintf(stderr, "malloc() FAILED (b_arr)\n");
		exit(0);
	}

	if (!(c_arr = (int*) malloc(vec_size))) {
		fprintf(stderr, "malloc() FAILED (c_arr)\n");
		exit(0);
	}

  /* Allocate GPGPU memory. */
  if ((status = cudaMalloc ((void**) &g_a_arr, vec_size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (g_a_arr), status = %d (%s)\n", status,
		  cudaGetErrorString(status));
		exit(1);
	}

	if ((status = cudaMalloc ((void**) &g_b_arr, vec_size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (g_b_arr), status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1);
	}

	if ((status = cudaMalloc ((void**) &g_c_arr, vec_size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (g_c_arr), status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1);
	}

  /* Populate CPU arrays. */
  for (int i = 0; i < size; i++) {
    a_arr[i] = i;
    b_arr[i] = i;
  }

  /* Send values from CPU to GPU */
  cudaMemcpy(g_a_arr, a_arr, vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(g_b_arr, b_arr, vec_size, cudaMemcpyHostToDevice);

  /* Call the kernel function to run on the GPGPU chip. */
	add <<<numThreadBlocks, numThreadsPerBlock>>>
	  (g_a_arr, g_b_arr, g_c_arr);

  /* Copy the result arrays from the GPU's memory to the CPU's memory. */
	cudaMemcpy(c_arr, g_c_arr, vec_size, cudaMemcpyDeviceToHost);

  /* Print the results */
  printf("The a_arr and b_arr are:\n");
  for (int i = 0; i < size; i++) {
    printf("%d ", a_arr[i]);
  }

  printf("\n");

  printf("The result array is:\n");
  for (int j = 0; j < size; j++) {
    printf("%d ", c_arr[j]);
  }

  printf("\n");

  /* Free CPU and GPU memory. */
  free(a_arr);
  free(b_arr);
  free(c_arr);
  cudaFree(g_a_arr);
  cudaFree(g_b_arr);
  cudaFree(g_c_arr);

  exit(0);

}
