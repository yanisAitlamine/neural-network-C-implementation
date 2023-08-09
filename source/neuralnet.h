#ifndef NN_H
#define NN_H
#include <stdbool.h>

typedef struct nNetwork nNetwork;
struct nNetwork{
	bool failFlag;
	size_t len;
	size_t* depths;
	double*** weights;
	double*** bias;
};

nNetwork* createNN(size_t len, size_t* depths);
void fillNN(nNetwork* NN);
void printNN(nNetwork* NN);
void freeNN (nNetwork);
#endif
