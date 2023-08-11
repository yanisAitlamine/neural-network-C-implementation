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
#define DP_IN 4
#define DP_OUT 2
int main()
{	
	srand(time(0));
	double*** train_data = (double **[]){
        (double *[]){(double[]){1.0, 0.0, 1.0, 0.0}, (double[]){1.0, 1.0}},
        (double *[]){(double[]){0.0, 1.0, 0.0, 1.0}, (double[]){1.0, 1.0}},
        (double *[]){(double[]){1.0, 1.0, 1.0, 1.0}, (double[]){0.0, 0.0}},
        (double *[]){(double[]){0.0, 0.0, 0.0, 0.0}, (double[]){0.0, 0.0}}};	
	char* file="NNtest.nn";
	size_t len=4;
	size_t depths[]={4,4,4,2};
	nNetwork* NN = createNN( len, depths);
	if (NN==NULL||NN->failFlag){
		ERROR("NN is NULL!\n");
		freeNN(NN);
		return 1;
	}
	fillNN(NN);
	printNN(NN);
	if (!writeNN (file, NN)){ERROR("failed to write");}
	freeNN(NN);
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
	double** output=(double**)malloc(NB_IN*sizeof(double*));
	*output=(double*)malloc(DP_OUT*sizeof(double));
	for (int i=0;i<NB_IN;i++){
		output[i]=(double*)malloc(DP_OUT*sizeof(double));
		if (input[i]==NULL||output[i]==NULL){
			ERROR("input or output null!\n");
			return 1;
		}
	}
	for  (int i=0;i<NB_IN;i++){
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
		printf("%f",avg_cost(expected[i],output[i],DP_OUT,MULTICLASS));
		if (i<NB_IN-1){
			printf (", ");
		}
	}
	printf ("]\n");
	free_mtrx(&output, NB_IN);
	free_mtrx(&input, NB_IN);
	free_mtrx(&expected, NB_IN);
	freeNN(NN);
	return 0;
}
