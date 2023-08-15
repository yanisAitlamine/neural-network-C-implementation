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

double*** init_data_matrix(int nb_sample,int depth_in, int depth_out){
    double*** data = (double***)calloc(nb_sample,sizeof(double**));
    if (check_malloc(data,"data matrix alloc failed")) return NULL;
    for (int i=0;i<nb_sample;i++){
	    data[i]=(double**)calloc(2,sizeof(double*));
	    data[i][0]=(double*)calloc(depth_in,sizeof(double));
	    data[i][1]=(double*)calloc(depth_out,sizeof(double));
    }
    return data;
}

void free_data_mtrx(double*** data, int nb_sample){
    for (int i=0;i<nb_sample;i++){
	    free(data[i][0]);
	    free(data[i][1]);
	    free(data[i]);
    }
    free(data);
}

//write a NN to a file
bool writeNN(char* filename, nNetwork* NN){
    printf ("Saving neural net of size %ld!\n",LEN(NN));
    FILE* file=NULL;
    file = fopen(filename, "wb+");
    if (file==NULL){ return false;}
    if (fwrite (&(LEN(NN)), sizeof(size_t), 1, file)!= 1){ 
	return false;
    }
    printf ("=");
    if (fwrite (DPTH(NN), sizeof(size_t), LEN(NN), file)!=NN->len){
	return false;
    }
    printf ("=");
    for (int i=0; i<LEN(NN)-1; i++){
	if (!writeMtrx(file, W(NN)[i], DPTH(NN)[i], NN->depths[i+1])){ 
	    return false;
	}
	if (fwrite (B(NN)[i], sizeof(double*), DPTH(NN)[i+1], file)!= NN->depths[i+1]){
    	    return false;
	}
    }
    if (fclose (file) == EOF){return false;}
    printf (">Saved!\n");
    return true;
}

//read a nNetwork from a file
nNetwork* readNN(char* filename){
    printf ("Loading neural networks!\n");
    FILE* file=NULL;
    file = fopen(filename, "rb");
    if (file==NULL){ return NULL;}
    size_t len;
    if (fread (&len, sizeof(size_t), 1, file)!= 1){
	return NULL;
    }
    printf ("=");
    size_t* depths=(size_t*)malloc(len*sizeof(size_t));
    if (fread(depths, sizeof(size_t), len,file)!=len){
	free(depths);
	return NULL;
    }
    printf ("=");
    nNetwork* NN=createNN(len,depths);
    free(depths);
    for (int i=0; i<LEN(NN)-1; i++){
	if (!readMtrx (file, W(NN)[i], DPTH(NN)[i], NN->depths[i+1])){ 	    
	    freeNN(NN);
	    return NULL;
	}
	if (fread (B(NN)[i], sizeof(double*), DPTH(NN)[i+1], file)!= NN->depths[i+1]){ 	    
	    freeNN(NN);
	    return NULL;
	}
    }
    printf (">Loaded!\n");
    if (fclose (file)== EOF){
	freeNN(NN);
	return NULL;
    }
    return NN;
}

//read matrix values from a file
bool readMtrx (FILE* file, double** mtrx, size_t len, size_t depth){
    printf ("=");
    for (int x=0;x<len;x++){
        if (fread (mtrx[x], sizeof(double), depth, file)!= depth){ 
		return false;
	}
    }
    return true;
}

//write mtrx values to a file
bool writeMtrx (FILE* file, double** mtrx, size_t len, size_t depth){
	printf ("=");
        for (int x=0;x<len;x++){
	    if (fwrite (mtrx[x], sizeof(double), depth, file)!= depth){
    	    return false;
	    }
        }
    return true;
}

