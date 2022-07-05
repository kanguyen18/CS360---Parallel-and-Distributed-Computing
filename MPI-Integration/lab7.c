/*
Khoa Nguyen - Lab 7 - 03/31/2022
The code was mostly from our text book (p.98). I added the Trap/f functions and getopt() argument processing.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <getopt.h>

double f(double x);    /* Function we're integrating */
double Trap(double a, double b, int n, double h);

int main(int argc, char *argv[]) {
   int        t = 1000; // number of trap
   int        comm_sz, my_rank, local_t;
   double     a = 0, b = 100, h, local_a, local_b;
   double     local_int, total_int;

   char*      endPtr;
   int        option;
   while ((option = getopt(argc, argv, ":a:b:t:")) != -1) {
       switch(option) {
           case 'a':
               a = strtod(optarg, &endPtr);
           break;

           case 'b':
               b = strtod(optarg, &endPtr);
           break;

           case 't':
               t = atoi(optarg);
           break;

           default :
               a=0;
               b=100;
               t=1000;
       }
   }

   /* Start up MPI */
   MPI_Init(NULL, NULL);

   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   h = (b-a)/t;
   local_t = t/comm_sz;

   local_a = a + my_rank*local_t*h;
   local_b = local_a + local_t*h;
   local_int = Trap(local_a, local_b, local_t, h);

   if(my_rank!=0) {
     MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
   }
   else {
     total_int = local_int;
     for (int q = 1; q < comm_sz; q++) {
        MPI_Recv(&local_int, 1, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total_int += local_int;
     }
   }

   if(my_rank == 0) {
     printf("With t = %d trapezoids, our estimate\n", t);
     printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
   }

   MPI_Finalize();
   return 0;
}  /* main */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral
 */
double Trap(double a, double b, int n, double h) {
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;
   for (k = 1; k <= n-1; k++) {
     integral += f(a+k*h);
   }
   integral = integral*h;

   return integral;
}  /* Trap */

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
