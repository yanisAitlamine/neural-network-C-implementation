#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "matrix.h"
#include "errors.h"
#include "read.h"

int main()
{
	size_t len=4;
	if (len >5){
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
	nNetwork NN = createNN(len, depths);
	if (NN.failFlag){
		ERROR("Bias is NULL!\n");
	}
	freeNN(NN);
	printf("freed NN\n");
	return 0;
}
