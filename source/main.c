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
#define SIZE_DATA 60000 
#define DP_IN 784
#define DP_OUT 10
#define LR 0.20000
#define EPOCHS 10
#define BATCH_SIZE 50
#define DEBUG false
#define TRAIN true
#define TEST false

int main()
{
	srand(time(0));
	double*** train_data=init_data_matrix(SIZE_DATA,DP_IN,DP_OUT);	
	if (train_data==NULL){
		ERROR("train_data is null!\n");
		return 1;
	}
	if (readMnistIMG(train_data,false,SIZE_DATA,TRAIN)){
		ERROR("read failed!\n");
		free_data_mtrx(train_data,SIZE_DATA);
		return 1;
	}
	if (readMnistLabels(train_data,false,SIZE_DATA,TRAIN)){
		ERROR("read failed!\n");
		free_data_mtrx(train_data,SIZE_DATA);
		return 1;
	}
	char* file="NNtest.nn";
	nNetwork* NN=NULL;
	if (fopen(file,"r")==NULL){
		size_t len=4;
		size_t depths[]={DP_IN,512,512,DP_OUT};
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
	double** input=(double**)malloc(SIZE_DATA*sizeof(double*));
	double** expected=(double**)malloc(SIZE_DATA*sizeof(double*));
	splitData(SIZE_DATA,DP_IN,DP_OUT,train_data,&input,&expected);
	printf ("data splitted\n");
	free_data_mtrx(train_data,SIZE_DATA);
	
	train(expected, input, NN, SIZE_DATA, BATCH_SIZE, LR, MSE,EPOCHS,DEBUG);

	for (int i=0;i<10;i++){
		compute (input[i], NN,!DEBUG);
		printf("costs: [");
		printf("%f",multnode_cost(expected[i],NN->activations[NN->len-1],NN->depths[NN->len-1],MSE));
		printf ("]\n");
	}
	printNN(NN);
	free_mtrx(input, SIZE_DATA);
	free_mtrx(expected, SIZE_DATA);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);
	return 0;
}
