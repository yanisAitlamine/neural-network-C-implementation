#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include "neuralnet.h"
#include "errors.h"
#include "in_outNN.h"
#include "compute.h"
#define NB_IN 4
#define DP_IN 2
#define DP_OUT 1
int main()
{	
	srand(time(0));
	double train_data[NB_IN][DP_IN+DP_OUT]={{1,0,1},{0,1,1},{1,1,0},{0,0,0}};
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
	double** input=(double**)malloc(NB_IN*sizeof(double*));
	for (int i=0;i<NB_IN;i++){
		input[i]=(double*)malloc(DP_IN*sizeof(double));
		for (int y=0;y<DP_IN;y++){input[i][y]=train_data[i][y];}
	}
	double** expected=(double**)malloc(NB_IN*sizeof(double*));
	for (int i=0;i<NB_IN;i++){
		expected[i]=(double*)malloc(DP_OUT*sizeof(double));
		for (int y=DP_IN;y<DP_IN+DP_OUT;y++){expected[i][y-DP_IN]=train_data[i][y];}
	}
	double** output=(double**)malloc(NB_IN*sizeof(double*));
	*output=(double*)malloc(DP_OUT*sizeof(double));
	for (int i=0;i<NB_IN;i++){
		output[i]=(double*)malloc(DP_OUT*sizeof(double));
		if (input[i]==NULL||output[i]==NULL){
			ERROR("input or output null!\n");
			return 1;
		}
		compute (input[i], &(output[i]), NN);
	}
	printf ("output: [");
	for (int i=0;i<NB_IN;i++){
		printf ("[");
		for (int y=0;y<DP_OUT;y++){
			printf("%f",output[i][y]);
			if (y<DP_OUT-1){
				printf (", ");
			}
		}
		if (i<NB_IN-1){
			printf ("],");
		} else {
			printf ("]");
		}
	}
	printf ("]\n");
	printf ("costs: [");
	for (int i=0;i<NB_IN;i++){
		printf ("[");
		for (int y=0;y<DP_OUT;y++){
			printf("%f",cost(expected[i][y],output[i][y],BINARY));
			if (y<DP_OUT-1){
				printf (", ");
			}
		}
		if (i<NB_IN-1){
			printf ("],");
		} else {
			printf ("]");
		}
	}
	printf ("]\n");
	freeNN(NN);
	return 0;
}
