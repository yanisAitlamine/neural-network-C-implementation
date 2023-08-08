#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "neuralnet.h"
#include "matrix.h"
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

// Return ERROR value if a matrix object is null
bool check_Malloc_Mtrx(matrix* data,char* message) {
    if(data==NULL){
        return ERROR(message);
    }
    return false;
}

bool check_weights(nNetwork* NN){
    if (check_Malloc_Mtrx(NN->weights,"Allocation failed for weights matrix!\n")){ 
        NN->failFlag=true; 
        return true;
    }
    return false;
}

bool check_bias(nNetwork* NN){
    if (check_Malloc_Mtrx(NN->bias,"Allocation failed for biass matrix!\n")){ 
        NN->failFlag=true; 
        return true;
    }
    return false;
}
