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
#define NB_IN 4 
#define DP_IN 2
#define DP_OUT 1
#define LR 0.1
#define EPOCHS 10000
#define DEBUG false

int main()
{	
	srand(time(0));
	double*** train_data = (double **[]){
        (double *[]){(double[]){1.0, 0.0}, (double[]){1.0}},
        (double *[]){(double[]){0.0, 1.0}, (double[]){1.0}},
        (double *[]){(double[]){1.0, 1.0}, (double[]){0.0}},
	(double *[]){(double[]){0.0, 0.0}, (double[]){0.0}}
//        (double *[]){(double[]){1.0, 1.0, 1.0, 0.0}, (double[]){0.0, 1.0}},
//        (double *[]){(double[]){1.0, 0.0, 1.0, 1.0}, (double[]){1.0, 0.0}},
	};	
	char* file="NNtest.nn";
	nNetwork* NN=NULL;
	if (fopen(file,"r")==NULL){
		size_t len=3;
		size_t depths[]={2,2,1};
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
	printNN(NN);
	double** input=(double**)malloc(NB_IN*sizeof(double*));
	double** expected=(double**)malloc(NB_IN*sizeof(double*));
	splitData(NB_IN,DP_IN,DP_OUT,train_data,&input,&expected);
	printf ("data splitted\n");
	double** output=(double**)malloc(NB_IN*sizeof(double*));
	*output=(double*)malloc(DP_OUT*sizeof(double));
	for (int i=0;i<NB_IN;i++){
		output[i]=(double*)malloc(DP_OUT*sizeof(double));
		if (input[i]==NULL||output[i]==NULL){
			ERROR("input or output null!\n");
			return 1;
		}
	}
	train(expected, input, output, NN, NB_IN, LR, MSE,EPOCHS,DEBUG);

	for (int i=0;i<NB_IN;i++){
		compute (input[i], &(output[i]), NN,DEBUG);
		printf ("output: [[");
		for (int y=0;y<NN->depths[NN->len-1];y++){
			printf("%f",output[i][y]);
			if (y<NN->depths[NN->len-1]-1){
				printf (", ");
			}
		}
		printf ("]");
		printf ("]\ncosts: [");
		printf("%f",multnode_cost(expected[i],output[i],NN->depths[NN->len-1],MSE));
		printf ("]\n");
	}
	printNN(NN);
	free_mtrx(&output, NB_IN);
	free_mtrx(&input, NB_IN);
	free_mtrx(&expected, NB_IN);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);
	return 0;
}
