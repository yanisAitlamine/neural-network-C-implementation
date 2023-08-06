#ifndef MATRIX_H
#define MATRIX_H
#include <stdbool.h>




typedef struct Matrix matrix;
struct Matrix{
	bool failFlag;
	size_t len;
	size_t depth;
	float** data;
};

matrix createMatrix(size_t len, size_t depth);
void freeMatrix (matrix mtrx);
void fillMatrix (matrix* mtrx);
void printMtrx (matrix* mtrx);
#endif
