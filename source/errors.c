/*
 * errors.c
 * Purpose: Implements errors-related functionality.
 * Auto-commented by GPT.
 */
#include <stdio.h> // Include library for required functionality.
#include <stdlib.h> // Include library for required functionality.
#include <stddef.h> // Include library for required functionality.
#include "errors.h" // Include library for required functionality.

// Return ERROR value if table is null
bool check_malloc(void* data,char* message) {
    if(data==NULL){
        ERROR(message);
        return true;
    }
    return false;
}

