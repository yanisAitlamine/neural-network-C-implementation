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

typedef struct nNetwork nNetwork;
struct nNetwork{
	bool failFlag;
	size_t len;
	matrix* weights;
	float** bias;
};

matrix createMatrix(size_t len, size_t depth);
void freeMatrix (matrix mtrx);
void fillMatrix (float** mtrx, size_t len, size_t* depth);
nNetwork createNN(size_t len, size_t* depths);
void freeNN (nNetwork);
#endif
