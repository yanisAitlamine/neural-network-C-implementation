#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "matrix.h"
#include "errors.h"

// Create an object matrix of a given length and size
matrix createMatrix(size_t len, size_t depth){
	printf("Creating matrix of size %ld, depth %ld!\n",len,depth);
	matrix mtrx;
	mtrx.len=len;
	mtrx.depth=depth;
	mtrx.failFlag=false;
	mtrx.data=(float**)malloc(len * sizeof(float*));
	if (check_Malloc_2Table(mtrx.data,"Allocation failed for data matrix!\n")){
        mtrx.failFlag=true;
        return mtrx;
	}
	for (int i=0;i<len;i++){
		mtrx.data[i]=(float*)malloc(depth*sizeof(float));
		if (check_Malloc_Table(mtrx.data[i],"Allocation failed for data table ")){
            printf("%d!\n",i);
            mtrx.failFlag=true;
            return mtrx;
       		 }
	}
	return mtrx;
}

// Free an object matrix
void freeMatrix (matrix mtrx){
	for (int i=0;i<mtrx.len;i++){
		free(mtrx.data[i]);
	}
	free(mtrx.data);
}



// Initialize mtrx data with random numbers
void fillMatrix (matrix* mtrx, size_t len, size_t depth){
	printf("Filling Matrix of size %ld, depth %ld!\n",len,depth);
	for (int i=0;i<len;i++){
		for (int y=0;y<depth;y++){
			(mtrx->data)[i][y]=0.5;	
		}	
	}
}

//Initialize weights and bias with rand numbers


