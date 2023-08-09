#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "errors.h"

// Return ERROR value if single row table is null
bool check_Malloc_Table(double* data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return false;
}

// Return ERROR value if a 2 rows table is null
bool check_Malloc_2Table(double** data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return false;
}


