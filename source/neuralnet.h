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
bool alloc_mtrx(double ***mtrx, size_t len, size_t depth);
void fillNN(nNetwork* NN);
void printNN(nNetwork* NN);
void freeNN (nNetwork* NN);
void free_mtrx(double ***data, size_t depth);
#endif
