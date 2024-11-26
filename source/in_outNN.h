/*
 * in_outNN.h
 * Purpose: Implements in_outNN-related functionality.
 * Auto-commented by GPT.
 */
#ifndef READ_H_INCLUDED
#define READ_H_INCLUDED
#include <stdio.h> // Include library for required functionality.
#include "neuralnet.h" // Include library for required functionality.

int read(FILE *stream, char *chain, size_t len);
void freeBuffer(); // Function definition.
bool writeNN(char* filename, nNetwork* NN);
nNetwork* readNN(char* filename);
bool readMtrx (FILE* file, double** mtrx, size_t len, size_t depth);
bool writeMtrx (FILE* file, double** mtrx, size_t len, size_t depth);
double*** init_data_matrix(int nb_sample,int depth_in, int depth_out);
void free_data_mtrx(double*** data, int nb_sample); // Function definition.
bool readMnistIMG(double ***data,int len_data, bool mode);
bool readMnistLabels(double ***data,int len_data, bool mode);
#endif // READ_H_INCLUDED