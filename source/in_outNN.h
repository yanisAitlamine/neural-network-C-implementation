#ifndef READ_H_INCLUDED
#define READ_H_INCLUDED
#include "neuralnet.h"

int read(FILE *stream, char *chain, size_t len);
void freeBuffer();
bool writeNN(char* filename, nNetwork* NN);
nNetwork* readNN(char* filename);
bool readMtrx (FILE* file, double** mtrx, size_t len, size_t depth);
bool writeMtrx (FILE* file, double** mtrx, size_t len, size_t depth);
double*** init_data_matrix(int nb_sample,int depth_in, int depth_out);
void free_data_mtrx(double*** data, int nb_sample);
bool readMnistIMG(double ***data, bool debug,int len_data, bool mode);
bool readMnistLabels(double ***data, bool debug,int len_data, bool mode);
#endif // READ_H_INCLUDED
