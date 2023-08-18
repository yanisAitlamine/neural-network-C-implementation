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
#define SIZE_DATA 3 
#define DP_IN 784
#define DP_OUT 10
#define LR 0.01
#define EPOCHS 2
#define BATCH_SIZE 3
#define TRAIN true
#define TEST false
#define SIZE_TEST 10

int main()
{
	srand(time(0));
	double*** train_data=init_data_matrix(SIZE_DATA,DP_IN,DP_OUT);	
	if (train_data==NULL){
		ERROR("train_data is null!\n");
		return 1;
	}
	if (readMnistIMG(train_data,SIZE_DATA,TRAIN)){
		ERROR("read train img failed!\n");
		free_data_mtrx(train_data,SIZE_DATA);
		return 1;
	}
	if (readMnistLabels(train_data,SIZE_DATA,TRAIN)){
		ERROR("read train labels failed!\n");
		free_data_mtrx(train_data,SIZE_DATA);
		return 1;
	}
	double*** test_data=init_data_matrix(SIZE_TEST,DP_IN,DP_OUT);	
	if (test_data==NULL){
		ERROR("test_data is null!\n");
		return 1;
	}
	if (readMnistIMG(test_data,SIZE_TEST,TEST)){
		ERROR("read test img failed!\n");
		free_data_mtrx(test_data,SIZE_TEST);
		return 1;
	}
	if (readMnistLabels(test_data,SIZE_TEST,TEST)){
		ERROR("read test labels failed!\n");
		free_data_mtrx(test_data,SIZE_TEST);
		return 1;
	}
	char* file="NNtest.nn";
	nNetwork* NN=NULL;
	if (fopen(file,"r")==NULL){
		size_t len=4;
		size_t depths[]={DP_IN,128,64,DP_OUT};
		int functions[]={RELU,RELU,RELU,SOFT};
		NN = createNN( len, depths,functions);
		if (NN==NULL||NN->failFlag){
			ERROR("NN is NULL!\n");
			freeNN(NN);
		return 1;
		}
		fillNN(NN);
		if (!writeNN (file, NN)){ERROR("failed to write");}
		freeNN(NN);
	}
	NN = readNN(file);
	if (NN==NULL||NN->failFlag){
		ERROR("NN 2 is NULL!\n");
		freeNN(NN);
		return 1;
	}
	printNN(NN);
	double** input=(double**)malloc(SIZE_DATA*sizeof(double*));
	double** expected=(double**)malloc(SIZE_DATA*sizeof(double*));

	shuffle(train_data,SIZE_DATA,DP_IN,DP_OUT,3);	
	splitData(SIZE_DATA,DP_IN,DP_OUT,train_data,&input,&expected);
	normalize(input,SIZE_DATA,DP_IN,255);
	printf ("data splitted\n");
	free_data_mtrx(train_data,SIZE_DATA);
	double** test_input=(double**)malloc(SIZE_TEST*sizeof(double*));
	double** test_expected=(double**)malloc(SIZE_TEST*sizeof(double*));

	splitData(SIZE_TEST,DP_IN,DP_OUT,test_data,&test_input,&test_expected);
	normalize(test_input,SIZE_TEST,DP_IN,255);
	free_data_mtrx(test_data,SIZE_TEST);
	train(expected, input,test_expected,test_input, NN, SIZE_DATA, BATCH_SIZE, SIZE_TEST, LR, MULTICLASS,EPOCHS);
	double costs[SIZE_TEST];
	for (int i=0;i<SIZE_TEST;i++){
		compute (test_input[i], NN);
		costs[i]=multnode_cost(test_expected[i],NN->activations[NN->len-1],NN->depths[NN->len-1],MULTICLASS);
		printf ("Testing\noutput[");
		for (int y=0;y<NN->depths[NN->len-1];y++){
		printf("%.1f",NN->activations[NN->len-1][y][AN]);
			if (y<NN->depths[NN->len-1]-1){
				printf (", ");
			}
		}
		printf ("]\nExpected [");
		for (int y=0;y<DPTH(NN)[LEN(NN)-1];y++){
			printf("%.1f ",test_expected[i][y]);
		}
		printf("]\ncosts: ");
		printf("%f\n",costs[i]);
	}
	free_mtrx(input, SIZE_DATA);
	free_mtrx(expected, SIZE_DATA);
	free_mtrx(test_input, SIZE_TEST);
	free_mtrx(test_expected,SIZE_TEST);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);
	return 0;
}
