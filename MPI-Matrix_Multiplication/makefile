all: lab8 mmm-naive

mmm-naive: mmm-naive.c
	gcc -std=c99 -Wall --pedantic -O0 mmm-naive.c -o mmm-naive

lab8: lab8.c
	mpicc -g -std=c99 -Wall --pedantic -O0 lab8.c -o lab8

clean:
	rm -f lab8 mmm-naive
