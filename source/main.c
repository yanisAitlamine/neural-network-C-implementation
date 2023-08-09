#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "neuralnet.h"
#include "errors.h"
#include "in_outNN.h"

int main()
{
	size_t len=6;
	size_t depths[]={2,5,5,5,5,3};
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
	freeNN(NN);
	nNetwork* nn = readNN(file);
	if (nn==NULL||nn->failFlag){
		ERROR("NN 2 is NULL!\n");
		freeNN(nn);
		return 1;
	}
	printNN(nn);
	freeNN(nn);
	return 0;
}
