#ifndef READ_H_INCLUDED
#define READ_H_INCLUDED
#include "neuralnet.h"

int read(FILE *stream, char *chain, size_t len);
void freeBuffer();
bool writeNN(char* filename, nNetwork* NN);
nNetwork* readNN(char* filename);
bool readMtrx (FILE* file, double** mtrx, size_t len, size_t depth);
bool writeMtrx (FILE* file, double** mtrx, size_t len, size_t depth);
#endif // READ_H_INCLUDED
