#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include "neuralnet.h"
#include "errors.h"
#include "in_outNN.h"
#include "compute.h"


int main()
{	
	srand(time(0));
	double train_data[][3]={{1,0,1},{0,1,1},{1,1,0},{0,0,0}};
	char* file="NNtest.nn";
/*	size_t len=3;
	size_t depths[]={2,2,1};
	char* file="NNtest.nn";
	nNetwork* NN = createNN( len, depths);
	if (NN==NULL||NN->failFlag){
		ERROR("NN is NULL!\n");
		freeNN(NN);
		return 1;
	}
	fillNN(NN);
	printNN(NN);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);*/
	nNetwork* NN = readNN(file);
	if (NN==NULL||NN->failFlag){
		ERROR("NN 2 is NULL!\n");
		freeNN(NN);
		return 1;
	}
	printNN(NN);
	double input[]={1,0};
	double** output=(double**)malloc(sizeof(double*));
	*output=(double*)malloc(NN->depths[NN->len-1]*sizeof(double));
	compute (input, output, NN);
	printf ("output: [");
	for (int i=0;i<NN->depths[NN->len-1];i++){
		printf("%f",(*output)[i]);
		if (i<NN->depths[NN->len-1]-1){
			printf (", ");
		} else {
			printf ("]\n");
		}
	}
	freeNN(NN);
	return 0;
}
