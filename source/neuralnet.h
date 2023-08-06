#ifndef NN_H
#define NN_H
#include <stdbool.h>
#include "matrix.h"

typedef struct nNetwork nNetwork;
struct nNetwork{
	bool failFlag;
	size_t len;
	matrix* weights;
	matrix* bias;
};

nNetwork createNN(size_t len, size_t* depths);
void fillNN(nNetwork* NN);
void printNN(nNetwork* NN);
void freeNN (nNetwork);
#endif
