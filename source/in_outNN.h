#ifndef READ_H_INCLUDED
#define READ_H_INCLUDED
#include "matrix.h"
#include "neuralnet.h"

int read(FILE *stream, char *chain, size_t len);
void freeBuffer();
bool writeNN(char* filename, nNetwork* NN);
nNetwork* readNN(char* filename);
bool readMtrx (FILE* file, matrix* mtrx);
bool writeMtrx (FILE* file, matrix* mtrx);
#endif // READ_H_INCLUDED
