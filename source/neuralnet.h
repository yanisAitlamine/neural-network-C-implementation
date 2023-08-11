#ifndef NN_H
#define NN_H
#include <stdbool.h>
//actual activation
#define AN 0
//not smoothed activation
#define	ZN 1
//dC/dAn
#define DERIV 2
//prime of smoothing func on zn
#define ZNPRIME

typedef struct nNetwork nNetwork;
struct nNetwork{
	bool failFlag;
	size_t len;
	size_t* depths;
	double*** weights;
	double** bias;
	double*** weightsGrd;
	double** biasGrd;
	double*** activations;
};

nNetwork* createNN(size_t len, size_t* depths);
bool alloc_mtrx(double ***mtrx, size_t len, size_t depth);
bool alloc_table(double** mtrx, size_t len);
void fillNN(nNetwork* NN);
void printNN(nNetwork* NN);
void freeNN (nNetwork* NN);
void free_mtrx(double ***data, size_t depth);
#endif
