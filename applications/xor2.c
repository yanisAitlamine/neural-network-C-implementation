#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../source/utils.h"
#include "../source/neuralnet.h"
#include "../source/errors.h"
#include "../source/in_outNN.h"
#include "../source/compute.h"
#define SIZE_DATA 32
#define DP_IN 6
#define DP_OUT 2
#define LR 0.01
#define EPOCHS 1000
#define SIZE_BATCH 1
#define TRAIN true
#define TEST false
#define SIZE_TEST 32

int main()
{
	srand(time(0));
	double*** train_data=init_data_matrix(SIZE_DATA,DP_IN,DP_OUT);	
	if (train_data==NULL){
		ERROR("train_data is null!\n");
		return 1;
	}
	for (int i = 0;i<SIZE_DATA;i++){
		for (int j=0;j<DP_IN;j++){
			train_data[i][0][j]=(( i & (1 << j ) ) ? 1 : 0 );
		}
		train_data[i][1][0]=1;
		for (int j=0;j<DP_IN-1;j+=2){
			if ((train_data[i][0][j]&&train_data[i][0][j+1])||(!(train_data[i][0][j])&&!(train_data[i][0][j+1]))) train_data[i][1][0]=0;
		}
		if (train_data[i][1][0]){
			train_data[i][1][1]=0;
		} else {
			train_data[i][1][1]=1;
		}
	}
	double*** test_data=init_data_matrix(SIZE_TEST,DP_IN,DP_OUT);	
	if (test_data==NULL){
		ERROR("test_data is null!\n");
		return 1;
	}
	for (int i = 0;i<SIZE_TEST;i++){
		for (int j=0;j<DP_IN;j++){
			test_data[i][0][j]=(( i & (1 << j ) ) ? 1 : 0 );
		}
		test_data[i][1][0]=1;
		for (int j=0;j<DP_IN-1;j+=2){
			if ((test_data[i][0][j]&&test_data[i][0][j+1])||(!(test_data[i][0][j])&&!(test_data[i][0][j+1]))) test_data[i][1][0]=0;
		}
		if (test_data[i][1][0]){
			test_data[i][1][1]=0;
		} else {
			test_data[i][1][1]=1;
		}
	}
	
	char* file="NNtest.nn";
	nNetwork* NN=NULL;
	if (fopen(file,"r")==NULL){
		size_t len=3;
		size_t depths[]={DP_IN,3,DP_OUT};
		size_t functions[]={RELU,RELU,SOFT};
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
	printNN(NN);
	mtrx* input=create_mtrx(SIZE_DATA,DP_IN);
	mtrx* expected=create_mtrx(SIZE_DATA,DP_OUT);
	shuffle(train_data,SIZE_DATA,DP_IN,DP_OUT,3);	
	splitData(SIZE_DATA,DP_IN,DP_OUT,train_data,input,expected);
	printf ("data splitted\n");
	free_data_mtrx(train_data,SIZE_DATA);
	mtrx* test_input=create_mtrx(SIZE_TEST,DP_IN);
	mtrx* test_expected=create_mtrx(SIZE_TEST,DP_OUT);
	splitData(SIZE_TEST,DP_IN,DP_OUT,test_data,test_input,test_expected);
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
		printf ("\ntesting\noutput:");
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
