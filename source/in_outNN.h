#ifndef READ_H_INCLUDED
#define READ_H_INCLUDED
#include "matrix.h"
#include "neuralnet.h"

int read(FILE *stream, char *chain, size_t len);
void freeBuffer();
void writeNN(FILE* toSave, nNetwork NN);
void writeMtrx(FILE* toSave, matrix* mtrx);
#endif // READ_H_INCLUDED
