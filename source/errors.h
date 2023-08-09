#ifndef __ERRORS__
#define __ERRORS__
#include <stdbool.h>
#include "neuralnet.h"

#define ERROR(message) printf(message);return true

bool check_malloc(void* data,char* message);
#endif
