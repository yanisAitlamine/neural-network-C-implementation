#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "utils.h"
#include "neuralnet.h"
#include "errors.h"
#include "in_outNN.h"
#include "compute.h"
#define SIZE_DATA 3000
#define DP_IN 28*28
#define DP_OUT 10
#define LR 0.01
#define EPOCHS 30
#define SIZE_BATCH 10
#define TRAIN true
#define TEST false
#define SIZE_TEST 100

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
		size_t depths[]={DP_IN,8,4,DP_OUT};
		size_t functions[]={RELU,RELU,RELU,SOFT};
		NN = createNN( len, depths,functions);
		if (NN==NULL||NN->failFlag){
			ERROR("NN is NULL!\n");
			freeNN(NN);
		return 1;
		}
		fillNN(NN);
		printf("filled correctly!\n");
		//printNN(NN);
		if (!writeNN (file, NN)){ERROR("failed to write!\n");}
		freeNN(NN);
	}
	NN = readNN(file);
	if (NN==NULL||NN->failFlag){
		ERROR("NN 2 is NULL!\n");
		freeNN(NN);
		return 1;
	}
	//printNN(NN);
	mtrx* input=create_mtrx(SIZE_DATA,DP_IN);
	mtrx* expected=create_mtrx(SIZE_DATA,DP_OUT);
	shuffle(train_data,SIZE_DATA,DP_IN,DP_OUT,3);	
	splitData(SIZE_DATA,DP_IN,DP_OUT,train_data,input,expected);
	normalize(input,255);
	printf ("data splitted\n");
	free_data_mtrx(train_data,SIZE_DATA);
	mtrx* test_input=create_mtrx(SIZE_TEST,DP_IN);
	mtrx* test_expected=create_mtrx(SIZE_TEST,DP_OUT);
	splitData(SIZE_TEST,DP_IN,DP_OUT,test_data,test_input,test_expected);
	normalize(test_input,255);
	free_data_mtrx(test_data,SIZE_TEST);
	train(expected ,input,test_expected, test_input, NN, SIZE_BATCH, LR, MULTICLASS, EPOCHS);
	
	double costs[SIZE_TEST];
	double* expect; 
	for (int i=0;i<X(test_expected);i++){
		predict (test_input,i, NN);
		printf("\ninputs:\n");
		print_mtrx_v(ACT(NN),0);
		expect=get_list_from_m(test_expected,i);
		costs[i]=multnode_cost(expect,ACT(NN),MULTICLASS);
		printf ("testing\noutput:");
		print_mtrx_v(ACT(NN),X(ACT(NN))-1);
		printf ("Expected [");
		for (int y=0;y<DPTH(NN)[LEN(NN)-1];y++){
			printf("%.1f ",expect[y]);
		}
		printf("]\ncosts: ");
		printf("%f\n",costs[i]);
		free(expect);
	}
	free_mtrx(input);
	free_mtrx(expected);
	free_mtrx(test_input);
	free_mtrx(test_expected);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);
	return 0;
}
