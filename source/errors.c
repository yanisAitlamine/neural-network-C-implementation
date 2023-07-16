#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "matrix.h"
#include "errors.h"

void testERROR(int testInt, char* message){
    ERROR(message);
}

int check_Malloc_Table(float* data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return 0;
}

int check_Malloc_2Table(float** data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return 0;
}

int check_Malloc_Mtrx_Table(matrix* data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return 0;
}
