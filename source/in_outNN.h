#ifndef READ_H_INCLUDED
#define READ_H_INCLUDED
#include "matrix.h"
#include "neuralnet.h"

int read(FILE *stream, char *chain, size_t len);
void freeBuffer();
bool writeNN(char* filename, nNetwork* NN, int total);
nNetwork* readNN(char* filename, int total);
#endif // READ_H_INCLUDED
