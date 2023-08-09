#ifndef __ERRORS__
#define __ERRORS__
#include <stdbool.h>
#include "neuralnet.h"

#define ERROR(message) printf(message);return true

bool check_Malloc_Table(double* data,char* message);
bool check_Malloc_2Table(double** data,char* message);
#endif
