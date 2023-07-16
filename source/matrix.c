#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "matrix.h"
#include "errors.h"

matrix createMatrix(size_t len, size_t depth){
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
		if (check_Malloc_Table,mtrx.data[i],("Allocation failed for data table ")){
            printf("%d!\n",i);
            mtrx.failFlag=true;
            return mtrx;
        }
	}
	return mtrx;
}

void freeMatrix (matrix mtrx){
	for (int i=0;i<mtrx.len;i++){
		free(mtrx.data[i]);
	}
	free(mtrx.data);
}

nNetwork createNN(size_t len, size_t* depths){
	nNetwork NN;
	NN.failFlag=false;
	NN.len=len;
	NN.weights=(matrix*)malloc(len*sizeof(matrix));
	if (check_Malloc_Mtrx_Table(NN.weights,"Allocation failed for weights matrix!\n")){
        NN.failFlag=true;
        return NN;
    }
	NN.bias=(float**)malloc(len*sizeof(float*));
	if (check_Malloc_2Table(NN.bias,"Allocation failed for bias matrix!\n")){
        NN.failFlag=true;
        return NN;
    }
	for (int i=0;i<len;i++){
		NN.weights[i]=createMatrix(i,i+1);
		if (NN.weights[i].failFlag){
			NN.failFlag=true;
			return NN;
		}
		NN.bias[i]=(float*)malloc(len*sizeof(float));
		if (check_Malloc_Table(NN.bias[i],"Allocation failed for bias table ")){
            printf("%d!\n",i);
            NN.failFlag=true;
            return NN;
        }
	}
	return NN;
}

void fillMatrix (float** mtrx, size_t len, size_t* depth){}

void freeNN(nNetwork NN){
	for (int i=0;i<NN.len;i++){
		freeMatrix(NN.weights[i]);
		free(NN.bias[i]);
	}
	free(NN.weights);
	free(NN.bias);
}
