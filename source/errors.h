/*
 * errors.h
 * Purpose: Implements errors-related functionality.
 * Auto-commented by GPT.
 */
#ifndef __ERRORS__
#define __ERRORS__
#include <stdbool.h> // Include library for required functionality.

#define ERROR(message) printf("%s",message)

bool check_malloc(void* data,char* message);
#endif