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
    if (fwrite (NN->depths, sizeof(size_t), NN->len, file)!=NN->len){
	return false;
    }
    for (int i=0; i<NN->len-1; i++){
	if (!writeMtrx(file, NN->weights[i], NN->depths[i], NN->depths[i+1])){ 
	    return false;
	}
	if (!writeMtrx(file, NN->bias[i], NN->depths[i], NN->depths[i+1])){ 
	    return false;
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
    nNetwork* buff = (nNetwork*) malloc (sizeof(nNetwork));
    if (fread (buff, sizeof(nNetwork), 1, file)!= 1){
	freeNN(*buff);
	return NULL;
    }
    printf ("=");
    buff->depths=(size_t*)malloc(buff->len*sizeof(size_t));
    if (fread(buff->depths, sizeof(size_t), buff->len,file)!=buff->len){
	freeNN(*buff);
	return NULL;
    }
    nNetwork* NN=createNN(buff->len,buff->depths);
    freeNN(*buff);
    for (int i=0; i<NN->len-1; i++){
	if (!readMtrx (file, NN->weights[i], NN->depths[i], NN->depths[i+1])){ 	    
	    freeNN(*NN);
	    return NULL;
	}
	if (!readMtrx(file, NN->bias[i], NN->depths[i], NN->depths[i+1])){ 	    
	    freeNN(*NN);
	    return NULL;
	}
    }
    printf (">Loaded!\n");
    if (fclose (file)== EOF){
	freeNN(*NN);
	return NULL;
    }
    return NN;
}

bool readMtrx (FILE* file, double** mtrx, size_t len, size_t depth){
    printf ("=");
    for (int x=0;x<len;x++){
        if (fread (mtrx[x], sizeof(double), depth, file)!= depth){ 
		return false;
	}
    }
    return true;
}

bool writeMtrx (FILE* file, double** mtrx, size_t len, size_t depth){
	printf ("=");
        for (int x=0;x<len;x++){
	    if (fwrite (mtrx[x], sizeof(double), depth, file)!= depth){
    	    return false;
	    }
        }
    return true;
}

