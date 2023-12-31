#ifndef NN_H
#define NN_H
#include <stdbool.h>
#include "mtrx.h"
#define DEBUG false

#define SIG 4
#define RELU 5
#define SOFT 6

#define FF(nn) (nn->failFlag)
#define LEN(nn) (nn->len)
#define DPTH(nn) (nn->depths)
#define W(nn) (nn->weights)
#define B(nn) (nn->bias)
#define WGRD(nn) (nn->weightsGrd)
#define BGRD(nn) (nn->biasGrd)
#define ACT(nn) (nn->activations)
#define FUNC(nn) (nn->functions)
#define ZN(nn) (nn->zn)
#define ERR(nn) (nn->error)
#define ZNP(nn) (nn->znprime)

typedef struct nNetwork nNetwork;
struct nNetwork{
	bool failFlag;
	size_t len;
	size_t* depths;
	size_t *functions;
	mtrx_vector* weights;
	mtrx_vector* bias;
	mtrx_vector* weightsGrd;
	mtrx_vector* biasGrd;
	mtrx_vector* activations;
	mtrx_vector* error;
	mtrx_vector* zn;
	mtrx_vector* znprime;

};

void copy_size_list(size_t *source, size_t *target,size_t len);
nNetwork* createNN(size_t len, size_t* depths, size_t* functions);
bool alloc_mtrx(double ***mtrx, size_t len, size_t depth);
bool alloc_table(double** mtrx, size_t len);
void fillNN(nNetwork* NN);
void initGRD(nNetwork* NN);
void updateNN(nNetwork* NN, double learning_rate);
void printNN(nNetwork* NN);
void printGrd(nNetwork* NN);
void printACT(nNetwork* NN);
void printZN(nNetwork* NN);
void printZNP(nNetwork* NN);
void printERR(nNetwork* NN);
void freeNN (nNetwork* NN);
#endif
