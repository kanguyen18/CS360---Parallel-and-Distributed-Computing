CXX=gcc -DDISPLAY -std=c99 -Wall --pedantic -O0

all: mmm-naive lab6

mmm-naive: mmm-naive.c
	$(CXX) mmm-naive.c -o mmm-naive

lab6: lab6.c
	$(CXX) -fopenmp lab6.c -o lab6

clean:
	rm -f mmm-naive lab6
