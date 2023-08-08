#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "errors.h"
#include "neuralnet.h"

// Create an object neural network of a given length with given layer lenghts
nNetwork createNN(size_t len, size_t* depths){
	printf("Creating Network of size %ld!\n",len);
	len-=1;
	nNetwork NN;
	NN.failFlag=false;
	NN.len=len;
	NN.weights=(matrix*)malloc(len*sizeof(matrix));
	if (check_weights(&NN)){
	    return NN;
	}
	NN.bias=(matrix*)malloc(len*sizeof(matrix));
	if (check_bias(&NN)){
	    return NN;
	}
	printf("depths: ");
	for (int i=0;i<len;i++){
		printf ("layer %d:%ld",i,depths[i]);
		if (i<len-1){printf(",");}
		NN.weights[i]=createMatrix(*(depths+i),*(depths+i+1));
		if (NN.weights[i].failFlag){
			NN.failFlag=true;
			return NN;
		}
		NN.bias[i]=createMatrix(*(depths+i),*(depths+i+1));
		if (NN.bias[i].failFlag){
			NN.failFlag=true;
			return NN;
		}
	}
	printf("\n");
	return NN;
}

//Initialize weights and bias with random numbers
void fillNN(nNetwork* NN){
    printf ("Filling neural net of size %ld!\n",NN->len+1);
    for (int i=0;i<NN->len;i++){
        printf ("Filling weights of row %d!\n",i);
        fillMatrix(&((NN->weights)[i]));
        printf ("Filling bias of row %d!\n",i);
        fillMatrix(&((NN->bias)[i]));
    }
}

//Print weights and bias
void printNN(nNetwork* NN){
    printf ("Printing neural net of size %ld!\n",NN->len);
    for (int i=0;i<NN->len;i++){
	printf ("===================================================================\n");
	printf("Layer %d\tlen: %ld\tdepth: %ld\n",i,NN->weights[i].len,NN->bias[i].depth);
	printf ("===================================================================\n");
    	printMtrx(&(NN->weights[i]));
    	printf("\n");
    	printMtrx(&(NN->bias[i]));
	printf("\n");
	} 
}


// Free a neural network object
void freeNN(nNetwork NN){
	for (int i=0;i<NN.len;i++){
		freeMatrix(NN.weights[i]);
		freeMatrix(NN.bias[i]);
	}
	free(NN.weights);
	free(NN.bias);
}
