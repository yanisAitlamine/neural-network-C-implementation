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
	nNetwork* nn = readNN(file);
	if (nn==NULL||nn->failFlag){
		ERROR("NN 2 is NULL!\n");
		freeNN(nn);
		return 1;
	}
	printNN(nn);
	freeNN(nn);
	double input[]={1,0};
	double output[1];
	compute (input, output, nn);
	printf ("output %ld",*output);
	return 0;
}
