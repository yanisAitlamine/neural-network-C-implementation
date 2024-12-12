#ifndef UT
#define UT
#define EULER_NUMBER 2.71828182845904523536
#define REGRESSION 0
#define BINARY 1
#define SQR_REG 2
#define MULTICLASS 3
#define MAE 4
#define MSE 5
#define PRODUCT 6
#define ERR_RETURN 10000000000000000.0
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include "errors.h"

double pow_double_int(double x, int n);
double rand_decimal();
double sigmoid(double n);
double sigmoidprime(double n);
double Relu(double n);
double Reluprime(double n);
double regression_cost(double expected, double output);
double sqr_regression(double expected, double output);
double binary_prime(double expected, double output);
double sqr_prime(double expected, double output);
double binary_cost(double expected, double output);
double cost (double expected, double output, int function);
double sum_double(double* data, int size_data);
double mean_double(double* data,int size_data);
void swapTables(double ***data,int base,int target,int depth_in,int depth_out);
void shuffle(double*** data,int len,int depth_in,int depth_out,int rounds);
#endif
