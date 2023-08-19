#ifndef NN_H
#define NN_H
#include <stdbool.h>
#include "mtrx.h"
#define DEBUG false

//actual activation
#define AN 0
//not smoothed activation
#define	ZN 1
//dC/dAn
#define DERIV 2
//prime of smoothing func on zn
#define ZNPRIME	3

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

typedef struct nNetwork nNetwork;
struct nNetwork{
	bool failFlag;
	size_t len;
	size_t* depths;
	int *functions;
	mtrx_vector* weights;
	mtrx_vector* bias;
	mtrx_vector* weightsGrd;
	mtrx_vector* biasGrd;
	mtrx_vector* activations;
};

void copy_int_list(int *source, int *functions,int len);
nNetwork* createNN(size_t len, size_t* depths, int* functions);
bool alloc_mtrx(double ***mtrx, size_t len, size_t depth);
bool alloc_table(double** mtrx, size_t len);
void fillNN(nNetwork* NN);
void initGRD(nNetwork* NN);
void updateNN(nNetwork* NN, double learning_rate);
void printNN(nNetwork* NN);
void printGrd(nNetwork* NN);
void printACT(nNetwork* NN);
void printERROR(nNetwork* NN);
void freeNN (nNetwork* NN);
void free_mtrx(double **data, size_t depth);
void multiply_grd(nNetwork* NN, double value);
void printTrainData(double** expected, double** input,int len_data,int depthinput, double depthoutput);
void free3D_mtrx(double ***data, size_t len, size_t* depths);
void normalize(double **input,int size_data,int len_row,double max);

#endif
