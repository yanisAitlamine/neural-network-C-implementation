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
#define ZNPRIME	3

#define FF(nn) (nn->failFlag)
#define LEN(nn) (nn->len)
#define DPTH(nn) (nn->depths)
#define W(nn) (nn->weights)
#define B(nn) (nn->bias)
#define WGRD(nn) (nn->weightsGrd)
#define BGRD(nn) (nn->biasGrd)
#define ACT(nn) (nn->activations)

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
void updateNN(nNetwork* NN, double learning_rate, bool debug);
void printNN(nNetwork* NN);
void printNNGrd(nNetwork* NN);
void freeNN (nNetwork* NN);
void free_mtrx(double **data, size_t depth);
void multiply_grd(nNetwork* NN, double value);
void printTrainData(double** expected, double** input,int len_data,int depthinput, double depthoutput);
void free3D_mtrx(double ***data, size_t len, size_t* depths);
#endif
