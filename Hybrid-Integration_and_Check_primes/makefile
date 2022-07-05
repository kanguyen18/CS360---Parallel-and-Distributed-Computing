all: lab10-prime lab10-curve

lab10-prime: lab10-prime.c
	mpicc -g -std=c99 -Wall --pedantic -O0 -fopenmp -lm lab10-prime.c -o lab10-prime

lab10-curve: lab10-curve.c
	mpicc -g -std=c99 -Wall --pedantic -O0 -fopenmp -lm lab10-curve.c -o lab10-curve

clean:
	rm -f lab10-prime lab10-curve
