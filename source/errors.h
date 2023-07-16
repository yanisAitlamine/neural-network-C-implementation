#ifndef __ERRORS__
#define __ERRORS__
#include <stdbool.h>
#include "matrix.h"
#define ERROR(message) printf(message);return 1

void testERROR(int testInt, char* message);
int check_Malloc_Table(float* data,char* message);
int check_Malloc_2Table(float** data,char* message);
int check_Malloc_Mtrx_Table(matrix* data,char* message);

#endif
