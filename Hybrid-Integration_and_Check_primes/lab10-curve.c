// Run code with:
// mpirun -np 8 -hostfile whedon-hosts --map-by node lab10-curve -t 4
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <sys/time.h>

double f(double x);
double Trap(double a, double b, int n, double h, int thread_count);

int main(int argc, char *argv[]) {

  int my_rank, world_size, provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  double  total_integral;   /* Store result in total_integral   */
  double  a=0, b=100;       /* Left and right endpoints   */
  int     n=100;          /* Number of trapezoids       */
  double  h;          /* Height of trapezoids       */
  int thread_count = 16; // thread count

  // Handle command line arguments with getopt
  char* endPtr;
  int option;

  while ((option = getopt(argc, argv, ":a:b:n:t:")) != -1) {
      switch(option) {
          case 'a':
              a = strtod(optarg, &endPtr);
          break;

          case 'b':
              b = strtod(optarg, &endPtr);
          break;

          case 'n':
              n = atoi(optarg);
          break;

          case 't':
              thread_count = atoi(optarg);
          break;

          default :
              a=0;
              b=100;
              n=100;
              thread_count = 16;
      }
  }

  total_integral = (f(a) + f(b))/2.0;
  h = (b-a)/n;

  // Hybrid section
  double sub_integral = 0;
  #pragma omp parallel for num_threads(thread_count) reduction(+:sub_integral)
  for (int k = 1+my_rank; k <= n-1; k+=world_size) {
      sub_integral += f(a+k*h);
  }

  // Node 0 gather and print results
  double *client_integral = NULL;
  if (my_rank == 0) {
    client_integral = (double *)malloc(sizeof(double) * world_size);
  }

  MPI_Gather(&sub_integral, 1, MPI_DOUBLE, client_integral, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (my_rank == 0) {
    for (int i=0; i<world_size; i++) {
      total_integral += client_integral[i];
    }
    total_integral = total_integral*h;
    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15f\n", a, b, total_integral);
  }

  MPI_Finalize();
  return(0);
}


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x)

{
    const double A1 = 8000,               a1 =  2;
    const double A3 =  100,  x03 =   1,   a3 =  2;
    const double A4 =   80,  x04 =   1,   a4 =  2;
    const double A5 =   60,  x05 = .01,   a5 =  2;
    const double A6 =   40,  x06 = .01,   a6 =  2;
    const double A7 =   20,  x07 = .01,   a7 =  2;

     return A1 * sin( x / a1 )
         +  A3 * exp( -pow( x - x03, 2.0 / a3 ) )
         +  A4 * exp( -pow( x - x04, 2.0 / a4 ) )
         +  A5 * exp( -pow( x - x05, 2.0 / a5 ) )
         +  A6 * exp( -pow( x - x06, 2.0 / a6 ) )
         +  A7 * exp( -pow( x - x07, 2.0 / a7 ) );
}  /* f */
