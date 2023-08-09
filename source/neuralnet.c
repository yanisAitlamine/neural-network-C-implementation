#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "errors.h"
#include "neuralnet.h"

// Create an object neural network of a given length with given layer lenghts
nNetwork* createNN(size_t len, size_t* depths){
	printf("Creating Network of size %ld!\n",len);
	nNetwork* NN=(nNetwork*)malloc(sizeof(nNetwork));
	NN->failFlag=false;
	NN->len=len;
	NN->depths=(size_t*)malloc(len*sizeof(size_t));
	NN->weights=(double***)malloc(len*sizeof(double**));
	//if (check_weights(&NN)){
//	    return NN;
	//}
	NN->bias=(double***)malloc(len*sizeof(double**));
	//if (check_bias(&NN)){
//	    return NN;
	//}
	printf("depths\t ");
	NN->depths[0]=depths[0];
	for (int i=0;i<len-1;i++){
		printf ("layer %d:%ld\t",i,depths[i]);
		NN->depths[i+1]=depths[i+1];
		NN->weights[i]=(double**)malloc(*(depths+i)*sizeof(double));
		for (int y=0;y<*(depths+i);y++){
		    NN->weights[i][y]=(double*)malloc(*(depths+i+1)*sizeof(double));
		}
		//if (NN.weights[i].failFlag){
	//		NN.failFlag=true;
//			return NN;
		//}
		NN->bias[i]=(double**)malloc(*(depths+i)*sizeof(double));
		for (int y=0;y<*(depths+i);y++){
		    NN->bias[i][y]=(double*)malloc(*(depths+i+1)*sizeof(double));
		}		
	//	if (NN.bias[i].failFlag){
		//	NN.failFlag=true;
	//		return NN;
		//}
	}
	printf("\nSuccesfully created!\n");
	return NN;
}

//Initialize weights and bias with random numbers
void fillNN(nNetwork* NN){
    printf ("Filling neural net of size %ld!\n",NN->len);
    for (int i=0;i<NN->len-1;i++){
        printf ("Filling weights of row %d!\n",i);
	double buff=0.1;
        for (int x=0;x<NN->depths[i];x++){
	    for (int y=0; y<NN->depths[i+1];y++){	
		NN->weights[i][x][y]=buff;
		buff= buff+0.1;
		if (buff>=1.0) buff=0.1;
	    }
	}
	buff=0.1;
        printf ("Filling bias of row %d!\n",i);
        for (int x=0;x<NN->depths[i];x++){
	    for (int y=0; y<NN->depths[i+1];y++){
		NN->bias[i][x][y]=buff;
		buff= buff+0.1;
		if (buff>=1.0) buff=0.1;
	    }
	}
    }
    printf ("Filling done!\n");
}

//Print weights and bias
void printNN(nNetwork* NN){
    printf ("Printing neural net of size %ld!\n",NN->len);
    for (int i=0;i<NN->len-1;i++){
	printf ("===================================================================\n");
	printf("Layer %d\tlen: %ld\tdepth: %ld\n",i,NN->depths[i],NN->depths[i+1]);
	printf ("===================================================================\n");
    	for (int x=0;x<NN->depths[i];x++){
	    printf("[");
	    for (int y=0; y<NN->depths[i+1];y++){
		printf("%.1f",NN->weights[i][x][y]);
		if (y<NN->depths[i+1]-1) printf(", ");
	    }
	    printf("]");
	}    	
	printf("\n");
    	for (int x=0;x<NN->depths[i];x++){
	    printf("[");
	    for (int y=0; y<NN->depths[i+1];y++){
		printf("%.1f",NN->bias[i][x][y]);
		if (y<NN->depths[i+1]-1) printf(", ");
	    }
	    printf("]");
	}
	printf("\n");
    } 
}


// Free a neural network object
void freeNN(nNetwork NN){
	free(NN.weights);
	free(NN.bias);
}
