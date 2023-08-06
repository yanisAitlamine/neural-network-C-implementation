#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "errors.h"
#include "neuralnet.h"

// Create an object neural network of a given length with given layer lenghts
nNetwork createNN(size_t len, size_t* depths){
	printf("Creating Network of size %ld!\n",len);
	nNetwork NN;
	NN.failFlag=false;
	NN.len=len;
	NN.weights=(matrix*)malloc(len*sizeof(matrix));
	if (check_Malloc_Mtrx(NN.weights,"Allocation failed for weights matrix!\n")){
        NN.failFlag=true;
        return NN;
    }
	NN.bias=(matrix*)malloc(len*sizeof(matrix));
	if (check_Malloc_Mtrx(NN.bias,"Allocation failed for bias matrix!\n")){
        NN.failFlag=true;
        return NN;
    }
	for (int i=0;i<len;i++){
		NN.weights[i]=createMatrix(1,*(depths+i));
		if (NN.weights[i].failFlag){
			NN.failFlag=true;
			return NN;
		}
		NN.bias[i]=createMatrix(1,*(depths+i));
		if (NN.bias[i].failFlag){
			NN.failFlag=true;
			return NN;
		}
	}
	return NN;
}

//Initialize weights and bias with random numbers
void fillNN(nNetwork* NN, size_t len, size_t* depths){
    printf ("Filling neural net of size %ld!\n",len);
    for (int i=0;i<len;i++){
        printf ("Filling weights of row %d!\n",i);
        fillMatrix(&((NN->weights)[i]),1,*(depths+i));
        printf ("Filling bias of row %d!\n",i);
        fillMatrix(&((NN->bias)[i]),1,*(depths+i));
    }
}

//Print weights and bias
void printNN(nNetwork* NN, size_t len, size_t* depths){
    printf ("Printing neural net of size %ld!\n",len);
    
    int maxD=0;
    for (int i=0;i<len;i++){
        if (*(depths+i)>maxD){maxD=*(depths+i);}
	printf ("\t%d\t",i);
    }
    printf ("\n");
    for (int x=0;x<maxD;x++){
	printf ("%d\t",x);
        for (int i=0;i<len;i++){
            if (x<((maxD-depths[i])/2)||x>=(((maxD-depths[i])/2)+depths[i])){
                printf("\t\t");
            } else {
		int rank = x-((maxD-depths[i])/2);
		printf("[%.1f,%.1f]\t",NN->weights[i].data[0][rank],NN->bias[i].data[0][rank]);
		//printf("%d\t",x);
	    } 
        }
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
