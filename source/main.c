#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "neuralnet.h"
#include "errors.h"
#include "in_outNN.h"
#include "compute.h"
#define NB_IN 100 
#define DP_IN 784
#define DP_OUT 10
#define LR 0.10000
#define EPOCHS 100
#define DEBUG false
#define TRAIN true
#define TEST false

int main()
{
	srand(time(0));
	double*** train_data=init_data_matrix(NB_IN,DP_IN,DP_OUT);	
	if (train_data==NULL){
		ERROR("train_data is null!\n");
		return 1;
	}
	if (readMnistIMG(train_data,DEBUG,NB_IN,TRAIN)){
		ERROR("read failed!\n");
		free_data_mtrx(train_data,NB_IN);
		return 1;
	}
	if (readMnistLabels(train_data,DEBUG,NB_IN,TRAIN)){
		ERROR("read failed!\n");
		free_data_mtrx(train_data,NB_IN);
		return 1;
	}
	char* file="NNtest.nn";
	nNetwork* NN=NULL;
	if (fopen(file,"r")==NULL){
		size_t len=4;
		size_t depths[]={DP_IN,28,16,DP_OUT};
		NN = createNN( len, depths);
		if (NN==NULL||NN->failFlag){
			ERROR("NN is NULL!\n");
			freeNN(NN);
		return 1;
		}
		fillNN(NN);
		printNN(NN);
		if (!writeNN (file, NN)){ERROR("failed to write");}
		freeNN(NN);
	}
	NN = readNN(file);
	if (NN==NULL||NN->failFlag){
		ERROR("NN 2 is NULL!\n");
		freeNN(NN);
		return 1;
	}
	//printNN(NN);
	double** input=(double**)malloc(NB_IN*sizeof(double*));
	double** expected=(double**)malloc(NB_IN*sizeof(double*));
	splitData(NB_IN,DP_IN,DP_OUT,train_data,&input,&expected);
	printf ("data splitted\n");
	free_data_mtrx(train_data,NB_IN);
	
	train(expected, input, NN, NB_IN, LR, MSE,EPOCHS,DEBUG);

	for (int i=0;i<NB_IN;i++){
		compute (input[i], NN,DEBUG);
		printf ("output: [");
		fflush(stdout);
		for (int y=0;y<NN->depths[NN->len-1];y++){
			printf("%f",NN->activations[NN->len-1][y][AN]);
			if (y<NN->depths[NN->len-1]-1){
				printf (", ");
			}
		}
		printf ("]\ncosts: [");
		printf("%f",multnode_cost(expected[i],NN->activations[NN->len-1],NN->depths[NN->len-1],MSE));
		printf ("]\n");
	}
	printNN(NN);
	free_mtrx(input, NB_IN);
	free_mtrx(expected, NB_IN);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);
	return 0;
}
