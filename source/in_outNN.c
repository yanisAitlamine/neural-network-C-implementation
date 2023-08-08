#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

bool writeNN(char* filename, nNetwork* NN, int total){
    printf ("Saving neural net of size %ld!\n",NN->len);
    FILE* file=NULL;
    file = fopen(filename, "wb+");
    if (file==NULL){ return false;}
    if (fwrite (&total, sizeof(int), 1, file)!=1){ 
	return false;
    } else {
	printf ("=");
    }
    if (fwrite (NN, sizeof(nNetwork), total, file)!= total){ 
	return false;
    } else {
	printf ("=");
    }
    if (fwrite (&(NN->len), sizeof(size_t), 1, file)!=1){ 
	return false;
    } else {
	printf ("=");
    }
    if (fwrite (NN->weights, sizeof(matrix), NN->len, file)!= NN->len){ 
	return false;
    } else {
	printf ("=");
    }
    if (fwrite (NN->bias, sizeof(matrix), NN->len, file)!= NN->len){ 
	return false;
    } else {
	printf ("=");
    }
    if (fclose (file) == EOF){return false;}
    printf (">Saved!\n");
    return true;
}

nNetwork* readNN(char* filename, int total){
    printf ("Loading %d neural networks!\n",total);
    FILE* file=NULL;
    file = fopen(filename, "rb");
    if (file==NULL){ return NULL;}
    if (fread (&total, sizeof(int), 1, file)!=1){ return NULL;}
    nNetwork* data = malloc (total * sizeof(nNetwork));
    if (fread (data, sizeof(nNetwork), total, file)!= total){
	free(data);
	return NULL;
    }
    if (fclose (file)== EOF){
	free(data);
	return NULL;
    }
    return data;
}
