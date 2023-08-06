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
	if (len >10){
		ERROR("Too many layers");
		return 1;
	}
	size_t depths[len];
	for (int i=0;i<len;i++){ 
		depths[i]=3;
		if (depths[i] >5){
			ERROR("Too deep"); 
           		 return 1;
		}
	}
	depths[0]=2;
	depths[2]=4;
	depths[3]=5;
	nNetwork NN = createNN(len, depths);
	if (NN.failFlag){
		ERROR("NN is NULL!\n");
	}
	fillNN(&NN);
	printNN(&NN);
	FILE* toSave=fopen ("NNtest.nn", "w+");
	writeNN (toSave, NN);
	freeNN(NN);
	printf("freed NN\n");
	return 0;
}
