#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "matrix.h"
#include "errors.h"

// Create an object matrix of a given length and size
matrix createMatrix(size_t len, size_t depth){
	matrix mtrx;
	mtrx.len=len;
	mtrx.depth=depth;
	mtrx.failFlag=false;
	allocData (&mtrx);
	return mtrx;
}

//Allocate space for data of a matrix
void allocData (matrix* mtrx){
	mtrx->data=(double**)malloc(mtrx->len * sizeof(double*));
	if (check_Malloc_2Table(mtrx->data,"Allocation failed for data matrix!\n")){
        mtrx->failFlag=true;
	return;
	}
	for (int i=0;i<mtrx->len;i++){
		mtrx->data[i]=(double*)malloc(mtrx->depth*sizeof(double));
		if (check_Malloc_Table(mtrx->data[i],"Allocation failed for data table ")){
            printf("%d!\n",i);
            mtrx->failFlag=true;
            return;
       		 }
	}
}

// Free an object matrix
void freeMatrix (matrix mtrx){
	for (int i=0;i<mtrx.len;i++){
		free(mtrx.data[i]);
	}
	free(mtrx.data);
}



// Initialize mtrx data with random numbers
void fillMatrix (matrix* mtrx){
	double x=0.1;
	for (int i=0;i<mtrx->len;i++){
		for (int y=0;y<mtrx->depth;y++){
			(mtrx->data)[i][y]=x;
			x+=0.1;
			if (x>=1.0){x=0.1;}
		}
	}
}

void printMtrx (matrix* mtrx){
	printf("[");
	for (int i=0;i<mtrx->len;i++){
		printf("[");
		for (int y=0;y<mtrx->depth;y++){
			printf("%.1f",(mtrx->data)[i][y]);
			if (y<mtrx->depth-1){printf(",");}
		}
		printf("]");
	}
	printf("]");
}
