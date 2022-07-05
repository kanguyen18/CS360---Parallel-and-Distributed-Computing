// Adapted from NVIDIA's examples, scaler value with one block, one thread
#include <stdio.h>

__global__ void add(int *a, int *b, int *c) { 
  *c = *a + *b;
}

__host__ void usage() {
	fprintf(stderr, "Usage: vec-add-start a b\n");
	exit(1);
}


int main(int argc, char** argv) {
  int a, b, c; // host copies of a, b, c 
  int *d_a, *d_b, *d_c; // device copies of a, b, c 
  int size = sizeof(int);
               
  // Setup input values  
	if (argc != 3) 
		usage();
	if (sscanf(argv[1], "%d", &a) != 1) 
		usage();
	if (sscanf(argv[2], "%d", &b) != 1) 
		usage();

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size); 
  cudaMalloc((void **)&d_b, size); 
  cudaMalloc((void **)&d_c, size);

  // Copy inputs to device
  cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU
  add<<< 1, 1 >>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
  fprintf(stdout, "%d + %d = %d\n", a, b, c); 

  // Cleanup
  cudaFree(d_a); 
  cudaFree(d_b); 
  cudaFree(d_c);
  
  return(0);
}
