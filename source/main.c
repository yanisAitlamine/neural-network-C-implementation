#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "matrix.h"
#include "errors.h"
#include "read.h"
#define READ_SIZE(addr,message) \
	printf("%s:\n",message);\
	char buffer[3];\
	*addr=strtoi(read(stdin,buffer,3));\
	free(buffer)


int main()
{
    int testInt=1;
    testERROR(testInt,"Test Error\n");
	size_t len;
	READ_SIZE(&len,"Enter the number of layer");
	if (len >5){
		ERROR("Too many layers");
		return 1;
	}
	size_t depths[len];
	char *numStr[2];
	for (int i=0;i<len;i++){
        sprintf(numStr,"%d",i);
		READ_SIZE(&(depths+i),(strcat("Enter the depth of layer ",numStr)));
		if (depths[i] >5){
		ERROR("Too deep");
            free(numStr);
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
