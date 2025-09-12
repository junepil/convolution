# Parallel Convolution Library
This repository contains the code for the matrix parallel convolution library. The code is written in C and uses OpenMP for parallelization.
## Structure
The repository is organized as follows:
- `conv.c`: The main file containing the convolution function.
- `test/`: The test directory containing the test code.
- `makefile`: The makefile for the project.

## How to run the code
If you want to build the code using Kaya, you can use the following command.
```sh
make hpc
```
If you want to run the test using $10^4\times10^4$ random input and $3 \times 3$ random kernel in Kaya, simply run. 
```sh
make test-hpc
```

It will generate the output file `test.txt` with the value of $f * g$.

Use the following command to clean up the directory.
```sh
make clean
```