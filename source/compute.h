#ifndef CMPT
#define CMPT
#include "utils.h"
#include "mtrx.h"
#include "neuralnet.h"

// Softmax activation function
void softmax(nNetwork* NN, int layer);
void softmaxPrime(nNetwork *NN,int layer);
void activation(nNetwork *NN, int layer);
void derivActivation(nNetwork *NN,int layer);
void predict(mtrx *input, int x, nNetwork *NN);
double sum_cost(double *expected, double *output, int x, int len, int function);
double MSE_cost(double* expected, double* output, int x, int len);
double MAE_cost(double* expected, double* output, int x, int len);
double multiclass_cost(double* expected, double* output, int x, int len);
double multnode_cost(double *expected, mtrx_vector *v, int function);
mtrx* sum_W_Zn_Deriv(int layer, nNetwork* NN);
void compute_grd(mtrx *expected, nNetwork *NN, int rank, int function);
double test(nNetwork *NN, mtrx* test_input,mtrx *test_expected,size_t function);
void batch(mtrx *train_expected, mtrx *train_input, int rank, nNetwork* NN, int size_batch, double learning_rate, int function);
void train(mtrx *train_expected, mtrx *train_input,mtrx *test_expected, mtrx *test_input, nNetwork* NN, int size_batch, double learning_rate, int function, int epochs);
#endif
