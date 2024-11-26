/*
 * neuralnet.h
 * Purpose: Implements neuralnet-related functionality.
 * Auto-commented by GPT.
 */
#ifndef NN_H
#define NN_H
#include <stdbool.h> // Include library for required functionality.
#include "mtrx.h" // Include library for required functionality.
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

void copy_size_list(size_t *source, size_t *target,size_t len); // Function definition.
nNetwork* createNN(size_t len, size_t* depths, size_t* functions);
void fillNN(nNetwork* NN); // Function definition.
void initGRD(nNetwork* NN); // Function definition.
void updateNN(nNetwork* NN, double learning_rate); // Function definition.
void printNN(nNetwork* NN); // Function definition.
void printGrd(nNetwork* NN); // Function definition.
void printACT(nNetwork* NN); // Function definition.
void printZN(nNetwork* NN); // Function definition.
void printZNP(nNetwork* NN); // Function definition.
void printERR(nNetwork* NN); // Function definition.
void freeNN (nNetwork* NN); // Function definition.
#endif