#include <math.h>

int check_prime_brute_force(unsigned long long number) {

	unsigned long long upper_bound = 0, remainder = 0, divisor = 3;
//	double u_b_d, n_d;

	remainder = number % 2;
	if (remainder == 0)
		return((int)0);

	/* Say number is not prime, then it factors as n = ab.  Consequently
	one of {a, b} has to be <= sqrt(n), and the other has to be >= sqrt(n).
	That is if both a, b are < sqrt(n) then ab < n.  Thanks Tim McLarnan. */

 	upper_bound = (unsigned long long)sqrt((double)number);

/* 	n_d = (double)number;
	asm ("fsqrt %[ubd],%[nd]" : [ubd] "=f" (u_b_d) : [nd] "f" (n_d));
		: "=d" (u_b_d) : "d" (n_d));
	upper_bound = (unsigned long)u_b_d; */

	while (divisor <= upper_bound) {
		remainder = number % divisor;

		if (remainder == 0)
			return((int)0);

		/* Since we filter all the even numbers we don't need to divide by
		even divisors.  If we initialize divisor to 3 we can increment by 2
		each time and cut our work in half.  Thanks Tim McLarnan.  */
		divisor += 2;
	}

	return((int)1);
}
