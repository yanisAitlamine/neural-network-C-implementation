#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "matrix.h"
#include "errors.h"

// Return ERROR value if single row table is null
int check_Malloc_Table(float* data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return 0;
}

// Return ERROR value if a 2 rows table is null
int check_Malloc_2Table(float** data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return 0;
}

// Return ERROR value if a matrix object is null
int check_Malloc_Mtrx(matrix* data,char* message) {
    if(data==NULL){
        return ERROR(message);
    }
    return 0;
}
