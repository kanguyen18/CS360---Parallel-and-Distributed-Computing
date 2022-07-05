/* File:    trap.c
 * Purpose: Calculate definite integral using trapezoidal 
 *          rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -o trap trap.c
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * IPP:     Section 3.2.1 (pp. 94 and ff.) and 5.2 (p. 216)
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include "omp.h"
#include <time.h>

double f(double x);    /* Function we're integrating */
double Trap(double a, double b, int n, double h, int thread_count); // add thread count arguments

int main(int argc, char *argv[]) {
    double  integral;   /* Store result in integral   */
    double  a=0, b=100;       /* Left and right endpoints   */
    int     n=100;          /* Number of trapezoids       */
    double  h;          /* Height of trapezoids       */
    int thread_count = 16; // thread count
    
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

    h = (b-a)/n;
    integral = Trap(a, b, n, h, thread_count); // add thread_count to arguments

    printf("With n = %d trapezoids, our estimate\n", n);
    printf("of the integral from %f to %f = %.15f\n", a, b, integral);

    return 0;
}  /* main */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double Trap(double a, double b, int n, double h, int thread_count) {
    double integral;

    integral = (f(a) + f(b))/2.0;
    
    // Parallelize this part from the original serial code
    # pragma omp parallel num_threads(thread_count) 
    {
    #pragma omp for reduction(+:integral)  
    for (int k = 1; k <= n-1; k++)
    {
        integral += f(a+k*h);
    }
        
    }
    integral = integral*h;
    
    return integral;
}  /* Trap */

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
