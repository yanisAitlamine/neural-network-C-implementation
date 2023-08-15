#ifndef CMPT
#define CMPT
#define EULER_NUMBER 2.71828182845904523536
#define REGRESSION 0
#define BINARY 1
#define SQR_REG 2
#define MULTICLASS 3
#define MAE 4
#define MSE 5
#define ERR 10000000000000000.0
#include "neuralnet.h"

double sigmoid(double n);
double sigmoidprime(double n);
void splitData(int num_obj, int len_in, int len_out, double ***data, double*** input, double*** expected);
void compute(double *input, nNetwork *NN, bool debug);
double regression_cost(double expected, double output);
double sqr_regression(double expected, double output);
double binary_prime(double expected, double output);
double binary_cost(double expected, double output);
double cost( double expected, double output, int function);
double sum_cost(double *expected, double **output, int len, int function);
double MSE_cost(double *expected, double **output, int len);
double MAE_cost(double *expected, double **output, int len);
double multiclass_cost(double *expected, double **output, int len);
double multnode_cost(double *expected, double **output, int len, int function);
void compute_grd(double *expected, nNetwork *NN, int function,bool debug);
double sum_W_Zn_Deriv(int rank, int ndnum, nNetwork* NN);
void batch(double **expected, double **input, nNetwork* NN, int size_batch, double learning_rate, int function,bool debug);
void train(double **expected, double **input, nNetwork* NN, int size_batch, double learning_rate, int function, int epochs, bool debug); 
void printTrainData(double** expected, double** input,int len_data,int depthinput, double depthoutput);
void shuffle(double*** data,int len,int depth_in,int depth_out,int rounds);
void swapTables(double ***data,int base,int target,int depth_in,int depth_out);
#endif
