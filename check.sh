gcc -c alexnet.c data.c train.c
gcc -o test test.c alexnet.o data.o train.o -w -lm

