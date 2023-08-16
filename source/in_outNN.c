#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include "errors.h"
#include "in_outNN.h"
#define IMG_INFO 4
#define LBL_INFO 2
#define PIXEL_IMG 784
#define SQR_IMG 28
#define SIZE_OUT 10

bool readMnistLabels(double ***data, bool debug,int len_data,bool mode){
    char* labelfile=mode?"/home/yanis/projects/train-labels-idx1-ubyte":"t/home/yanis/projects/10k-labels-idx1-ubyte";
    FILE* file=NULL;
    file = fopen(labelfile, "rb");
    if (file==NULL) return true;
    if (debug)printf("Reading labels, ");
    unsigned char* bytes=malloc(sizeof(int)*(LBL_INFO));
    if (fread (bytes, sizeof(int), LBL_INFO, file)!= LBL_INFO){ 
	fclose(file);
	return true;
    }
    if (debug)printf("file info:\t");
    fflush(stdout);
    for (int i=0;i<LBL_INFO;i++){
	int info=(bytes[i*4]<<24)|(bytes[i*4+1]<<16)|(bytes[i*4+2]<<8)|bytes[i*4+3];
	if (debug)printf("%d\t", info);
    }
    for (int i=0;i<len_data;i++){
	if(debug)printf ("\nimage %d:\t",i);

	    if (fread (bytes, 1, 1, file)!= 1){ 
		fclose(file);
		return true;
	    }
	printf ("%d\t",bytes[0]);
	for (int y=0;y<SIZE_OUT;y++){
	    if (y==bytes[0]){data[i][1][y]=1;}else{data[i][1][y]=0;}
	    if (debug){
		    printf ("%.1f\t",data[i][1][y]);
	    }
	}
    }
    
    printf ("\nFile read with success!\n");
    fclose(file);
    return false;
}

bool readMnistIMG(double ***data, bool debug,int len_data,bool mode){
    char* imagefile=mode?"/home/yanis/projects/train-images-idx3-ubyte":"/home/yanis/projects/t10k-images-idx3-ubyte";
    FILE* file=NULL;
    file = fopen(imagefile, "rb");
    if (file==NULL) return true;
    if (debug)printf("Reading images, ");
    unsigned char* bytes=malloc(sizeof(int)*(IMG_INFO));
    if (fread (bytes, sizeof(int), IMG_INFO, file)!= IMG_INFO){ 
	fclose(file);
	return true;
    }
    if (debug)printf("file info:\t");
    fflush(stdout);
    for (int i=0;i<IMG_INFO;i++){
	int info=(bytes[i*4]<<24)|(bytes[i*4+1]<<16)|(bytes[i*4+2]<<8)|bytes[i*4+3];
	if (debug)printf("%d\t", info);
    }
    for (int i=0;i<len_data;i++){
	if(debug)printf ("\nimage %d\n",i);
	for (int y=0;y<PIXEL_IMG;y++){
	    if (fread (bytes, 1, 1, file)!= 1){ 
		fclose(file);
		return true;
	    }
	    data[i][0][y]=(double)(bytes[0]);
	    if (data[i][0][y]<100&&debug)printf ("0 ");
	    if (100<data[i][0][y]&&data[i][0][y]<200&&debug)printf("- ");
	    if (data[i][0][y]>200&&debug)printf("1 ");
	    if (y%28==0&&debug)printf("\n");
	}
    }
    
    printf ("\nFile read with success!\n");
    fclose(file);
    return false;
}

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
    printf ("Saving neural net of size %ld!\n",NN->len);
    FILE* file=NULL;
    file = fopen(filename, "wb+");
    if (file==NULL){ return false;}
    if (fwrite (&(NN->len), sizeof(size_t), 1, file)!= 1){ 
	return false;
    }
    printf ("=");
    if (fwrite (NN->depths, sizeof(size_t), NN->len, file)!=NN->len){
	return false;
    }
    printf ("=");
    for (int i=0; i<NN->len-1; i++){
	if (!writeMtrx(file, NN->weights[i], NN->depths[i], NN->depths[i+1])){ 
	    return false;
	}
	if (fwrite (NN->bias[i], sizeof(double*), NN->depths[i+1], file)!= NN->depths[i+1]){
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
    for (int i=0; i<NN->len-1; i++){
	if (!readMtrx (file, NN->weights[i], NN->depths[i], NN->depths[i+1])){ 	    
	    freeNN(NN);
	    return NULL;
	}
	if (fread (NN->bias[i], sizeof(double*), NN->depths[i+1], file)!= NN->depths[i+1]){ 	    
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

