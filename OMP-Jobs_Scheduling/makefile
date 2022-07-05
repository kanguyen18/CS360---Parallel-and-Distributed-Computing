CXX=gcc -Wall -Werror -std=c99

all: lab5-primes lab5-curve

lab5-primes: lab5-primes.c
	$(CXX) lab5-primes.c -lm -fopenmp -o lab5-primes

lab5-curve: lab5-curve.c
	$(CXX) lab5-curve.c -lm -fopenmp -o lab5-curve

clean:
	rm -f lab5-primes lab5-curve
