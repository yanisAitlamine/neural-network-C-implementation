#ifndef __ERRORS__
#define __ERRORS__
#include <stdbool.h>

#define ERROR(message) printf("%s",message)

bool check_malloc(void* data,char* message);
#endif
