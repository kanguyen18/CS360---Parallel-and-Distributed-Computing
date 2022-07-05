/*
 * CUDA Hello World Program
 *
 * Usage: hello-cuda <number thread blocks> <number threads per block>
 *
 * charliep	09-April-2011		First pass, based on the example by Alan Kaminsky
 * charliep	01-July-2011		Improved error handling.
 * charliep 21-March-2016		Fixed to work with CUDA 6.x
 * charliep 15-April-2020		Fixed to work with CUDA 10.2
*/
#include <stdlib.h>
#include <stdio.h>

/*
  * Kernel function, this runs on the GPGPU card. This thread's element of blockArray 
  * is set to this thread's block index. This thread's element of threadArray is set 
  * to this thread's thread index within the block. The appropriate position in 
  * globalThreadArray is set to the sum of the three identifiers (which should be 
  * unique). Note the use of the builtin variables blockDim, blockIdx and threadIdx.
*/
__global__ void hello(int* blockArray, int* threadArray, int* globalThreadArray) {
	int i; 
	
	i =  threadIdx.x + blockDim.x * blockIdx.x;
	blockArray[i] = blockIdx.x;
	threadArray[i] = threadIdx.x;
	globalThreadArray[i] = threadIdx.x + blockDim.x * blockIdx.x;
}

__host__ void usage() {
	fprintf(stderr, "Usage: hello-cuda <number thread blocks> <number threads per block>\n");
	exit(1);
}

int main(int argc, char** argv) {
	int numThreadBlocks, numThreadsPerBlock, totalNumThreads, size, i;
	int *cpuBlockArray, *cpuThreadArray, *cpuGThreadArray; 
	int *gpuBlockArray, *gpuThreadArray, *gpuGThreadArray;
	cudaError_t status = (cudaError_t)0; 

	if (argc != 3) 
		usage();
	if (sscanf(argv[1], "%d", &numThreadBlocks) != 1) 
		usage();
	if (sscanf(argv[2], "%d", &numThreadsPerBlock) != 1) 
		usage();
	
	totalNumThreads = numThreadBlocks * numThreadsPerBlock; 

	/* Allocate CPU memory. */ 
	size = totalNumThreads * sizeof(int);
	
	if (!(cpuBlockArray = (int*) malloc(size))) {
		fprintf(stderr, "malloc() FAILED (Block)\n"); 
		exit(0);
	}
	
	if (!(cpuThreadArray = (int*) malloc(size))) {
		fprintf(stderr, "malloc() FAILED (Thread)\n"); 
		exit(0);
	}
	
	if (!(cpuGThreadArray = (int*) malloc(size))) {
		fprintf(stderr, "malloc() FAILED (GThread)\n"); 
		exit(0);
	}
	
	/* Allocate GPGPU memory. */ 
	if ((status = cudaMalloc ((void**) &gpuBlockArray, size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (Block), status = %d (%s)\n", status,     
		  cudaGetErrorString(status));
		exit(1); 
	}

	if ((status = cudaMalloc ((void**) &gpuThreadArray, size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (Thread), status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1); 
	}
	
	if ((status = cudaMalloc ((void**) &gpuGThreadArray, size)) != cudaSuccess) {
		printf("cudaMalloc() FAILED (GThread), status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1); 
	}
	
	/* Call the kernel function to run on the GPGPU chip. */ 
	hello <<<numThreadBlocks, numThreadsPerBlock>>> 
	  (gpuBlockArray, gpuThreadArray, gpuGThreadArray);
	
	/* Copy the result arrays from the GPU's memory to the CPU's memory. */ 
	cudaMemcpy(cpuBlockArray, gpuBlockArray, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuThreadArray, gpuThreadArray, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuGThreadArray, gpuGThreadArray, size, cudaMemcpyDeviceToHost);
	
	/* Display the results. */ 
	printf("blockID\t\tthreadID\tglobalThreadID\n");
	
	for (i = 0; i < totalNumThreads; ++i) {
		printf("%d\t\t%d\t\t%d\n", cpuBlockArray[i], cpuThreadArray[i], cpuGThreadArray[i]);
	}
	
	printf("Total number of hellos: %d\n", totalNumThreads); 
	
	/* Free CPU and GPU memory. */
	free(cpuBlockArray);
	free(cpuThreadArray);
	free(cpuGThreadArray);
	cudaFree(gpuBlockArray);
	cudaFree(gpuThreadArray);
	cudaFree(gpuGThreadArray);
	
	exit(0); 
}
