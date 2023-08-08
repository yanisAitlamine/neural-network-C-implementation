#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "errors.h"
#include "in_outNN.h"

// read from a stream into a chain for a set number of characters
int read(FILE *stream, char *chain, size_t len) {

    char *returnChar = NULL;

    if (fgets(chain, len, stream)==NULL){
        return 1;
    }
    returnChar = strchr(chain, '\n');
    if (returnChar!=NULL){
            *returnChar='\0';
    }
    freeBuffer();
    return 0;
}

// read characters in the buffer until the end of the fil or \n
void freeBuffer() {
    int a=0;
    while (a!='\n' && a!=EOF) {
        a=getchar();
    }
}

bool writeNN(char* filename, nNetwork* NN){
    printf ("Saving neural net of size %ld!\n",NN->len);
    FILE* file=NULL;
    file = fopen(filename, "wb+");
    if (file==NULL){ return false;}
    if (fwrite (NN, sizeof(nNetwork), 1, file)!= 1){ 
	return false;
    } else {
	printf ("=");
    }
    for (int i=0; i<NN->len; i++){
	if (fwrite (&(NN->weights[i]), sizeof(matrix), 1, file)!= 1){ 
	    return false;
	} else {
	    printf ("=");
	    for (int x=0;x<NN->weights[i].len;x++){
		if (fwrite (NN->weights[i].data[x], sizeof(double), NN->weights[i].depth, file)!= NN->weights[i].depth){ 
		    return false;
		}
	    }
	}
	if (fwrite (&(NN->bias[i]), sizeof(matrix), 1, file)!= 1){ 
	    return false;
	} else {
	    printf ("=");
	    for (int x=0;x<NN->bias[i].len;x++){
		if (fwrite (NN->bias[i].data[x], sizeof(double), NN->bias[i].depth, file)!= NN->bias[i].depth){ 
		    return false;
		}
	    }
	}
    }
    if (fclose (file) == EOF){return false;}
    printf (">Saved!\n");
    return true;
}

nNetwork* readNN(char* filename){
    printf ("Loading neural networks!\n");
    FILE* file=NULL;
    file = fopen(filename, "rb");
    if (file==NULL){ return NULL;}
    nNetwork* NN = (nNetwork*) malloc (sizeof(nNetwork));

    if (fread (NN, sizeof(nNetwork), 1, file)!= 1){ 
    	free(NN);
	return NULL;
    } else {
	printf ("=");
    }
    NN->weights=(matrix*)malloc(NN->len*sizeof(matrix));
    if (check_weights(NN)){
	free(NN);
        return NULL;
    }
    NN->bias=(matrix*)malloc(NN->len*sizeof(matrix));
    if (check_bias(NN)){
	free(NN);
        return NULL;
    }
    for (int i=0; i<NN->len; i++){
	if (readMtrx (file, &(NN->weights[i]))){ 	    
	    free(NN);
	    return NULL;
	}
	if (readMtrx(file, &(NN->bias[i]))){ 	    
	    free(NN);
	    return NULL;
	}
    }
    printf (">Loaded!\n");
    if (fclose (file)== EOF){
	free(NN);
	return NULL;
    }
    return NN;
}

bool readMtrx (FILE* file, matrix* mtrx){
    if (fread (mtrx, sizeof(matrix), 1, file)!= 1){ 	    
	return true;
    } else {
	printf ("=");
        for (int x=0;x<mtrx->len;x++){
	    allocData(mtrx);
	    if (fread (mtrx->data[x], sizeof(double), mtrx->depth, file)!= mtrx->depth || mtrx->failFlag){ 
		    return true;
	   }
	}
    return false;
    }
}
