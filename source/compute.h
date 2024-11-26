/*
 * compute.h
 * Purpose: Implements compute-related functionality.
 * Auto-commented by GPT.
 */
#ifndef CMPT
#define CMPT
#include "utils.h" // Include library for required functionality.
#include "mtrx.h" // Include library for required functionality.
#include "neuralnet.h" // Include library for required functionality.

// Softmax activation function
void softmax(nNetwork* NN, int layer); // Function definition.
void softmaxPrime(nNetwork *NN,int layer); // Function definition.
void activation(nNetwork *NN, int layer); // Function definition.
void derivActivation(nNetwork *NN,int layer); // Function definition.
void predict(mtrx *input, int y, nNetwork *NN); // Function definition.
double sum_cost(double *expected, double **output,  int len, int function);
double MSE_cost(double* expected, double** output,  int len);
double MAE_cost(double* expected, double** output,  int len);
double multiclass_cost(double* expected, double** output, int len);
double multnode_cost(double *expected, mtrx *m, int function);
mtrx* sum_W_Zn_Deriv(int layer, nNetwork* NN);
void compute_grd(mtrx *expected, nNetwork *NN, int rank, int function); // Function definition.
double test(nNetwork *NN, mtrx* test_input,mtrx *test_expected,size_t function);
void batch(mtrx *train_expected, mtrx *train_input, int rank, nNetwork* NN, int size_batch, double learning_rate, int function); // Function definition.
void train(mtrx *train_expected, mtrx *train_input,mtrx *test_expected, mtrx *test_input, nNetwork* NN, int size_batch, double learning_rate, int function, int epochs); // Function definition.
#endif