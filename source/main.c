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
#define NB_IN 28 
#define DP_IN 4
#define DP_OUT 1
#define LR 0.10000
#define EPOCHS 100000
#define DEBUG false

int main()
{
	srand(time(0));
	double*** train_data=init_data_matrix(NB_IN,DP_IN,DP_OUT);	
	if (train_data==NULL){
		ERROR("train_data is null");
		return 1;
	}
	//init the data for the specific problem of xor gate
	for (int i=0;i<DP_IN;i++){
		train_data[0][0][i]=0;
	}
	train_data[0][1][0]=0;
	for (int i=1;i<NB_IN;i++){
		int rest=1;
		for (int y=0;y<DP_IN;y++){
			if (rest){
				if (train_data[i-1][0][y]){
					rest=1;
					train_data[i][0][y]=0;
				}else{
					train_data[i][0][y]=1;
					rest=0;
				}
			} else {train_data[i][0][y]=train_data[i-1][0][y];}

		}
		if (((int)train_data[i][0][0]^(int)train_data[i][0][1])&&((int)train_data[i][0][2]^(int)train_data[i][0][3])){
			train_data[i++][1][0]=1;
			if (i<NB_IN){
				train_data[i][1][0]=1;
				for (int x=0;x<DP_IN;x++)train_data[i][0][x]=train_data[i-1][0][x];
			}
		}else{
			train_data[i][1][0]=0;
		}
	}
	char* file="NNtest.nn";
	nNetwork* NN=NULL;
	if (fopen(file,"r")==NULL){
		size_t len=4;
		size_t depths[]={4,4,2,1};
		NN = createNN( len, depths);
		if (NN==NULL||FF(NN)){
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
	if (NN==NULL||FF(NN)){
		ERROR("NN 2 is NULL!\n");
		freeNN(NN);
		return 1;
	}
	printNN(NN);
	double** input=(double**)malloc(NB_IN*sizeof(double*));
	double** expected=(double**)malloc(NB_IN*sizeof(double*));
	splitData(NB_IN,DP_IN,DP_OUT,train_data,&input,&expected);
	printf ("data splitted\n");
	free_data_mtrx(train_data,NB_IN);
	train(expected, input, NN, NB_IN,28, LR, MSE,EPOCHS,DEBUG);

	for (int i=0;i<NB_IN;i++){
		compute (input[i], NN,!DEBUG);
		printf ("output: [");
		fflush(stdout);
		for (int y=0;y<DPTH(NN)[LEN(NN)-1];y++){
			printf("%f",ACT(NN)[LEN(NN)-1][y][AN]);
			if (y<DPTH(NN)[LEN(NN)-1]-1){
				printf (", ");
			}
		}
		printf ("]\ncosts: [");
		printf("%f",multnode_cost(expected[i],ACT(NN)[LEN(NN)-1],DPTH(NN)[NN->len-1],MSE));
		printf ("]\n");
	}
	printNN(NN);
	free_mtrx(input, NB_IN);
	free_mtrx(expected, NB_IN);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);
	return 0;
}
