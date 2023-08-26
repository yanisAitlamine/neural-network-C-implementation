#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "errors.h"
#include "in_outNN.h"
#define IMG_INFO 4
#define LBL_INFO 2
#define PIXEL_IMG 784
#define SQR_IMG 28
#define SIZE_OUT 10
#define DEBUGIO !true
#define DEBUGIONN !true

bool readMnistLabels(double ***data,int len_data,bool mode){
    char* labelfile=mode?"./data/train-labels-idx1-ubyte":"./data/t10k-labels-idx1-ubyte";
    FILE* file=NULL;
    file = fopen(labelfile, "rb");
    if (file==NULL) return true;
#if DEBUGIO
    printf("Reading labels, ");
#endif
    unsigned char* bytes=malloc(sizeof(int)*(LBL_INFO));
    if (fread (bytes, sizeof(int), LBL_INFO, file)!= LBL_INFO){ 
	fclose(file);
	return true;
    }
#if DEBUGIO
    printf("file info:\t");
    fflush(stdout);
#endif
    for (int i=0;i<LBL_INFO;i++){
	int info=(bytes[i*4]<<24)|(bytes[i*4+1]<<16)|(bytes[i*4+2]<<8)|bytes[i*4+3];
#if DEBUGIO
	 printf("%d\t", info);
#endif
    }
    for (int i=0;i<len_data;i++){
#if DEBUGIO
	printf ("\nimage %d:\t",i);
#endif
	    if (fread (bytes, 1, 1, file)!= 1){ 
		fclose(file);
		return true;
	    }
#if DEBUGIO
	printf ("%d\t",bytes[0]);
#endif
	for (int y=0;y<SIZE_OUT;y++){
	    if (y==bytes[0]){data[i][1][y]=1;}else{data[i][1][y]=0;}
	}
    }
    fclose(file);
    return false;
}

bool readMnistIMG(double ***data,int len_data,bool mode){
    char* imagefile=mode?"./data/train-images-idx3-ubyte":"./data/t10k-images-idx3-ubyte";
    FILE* file=NULL;
    file = fopen(imagefile, "rb");
    if (file==NULL) return true;
#if DEBUGIO
    printf("Reading images, ");
#endif
    unsigned char* bytes=malloc(sizeof(int)*(IMG_INFO));
    if (fread (bytes, sizeof(int), IMG_INFO, file)!= IMG_INFO){ 
	fclose(file);
	return true;
    }
#if DEBUGIO
    printf("file info:\t");
    fflush(stdout);
#endif
    for (int i=0;i<IMG_INFO;i++){
	int info=(bytes[i*4]<<24)|(bytes[i*4+1]<<16)|(bytes[i*4+2]<<8)|bytes[i*4+3];
#if DEBUGIO
	printf("%d\t", info);
#endif
    }
    for (int i=0;i<len_data;i++){
#if DEBUGIO
	printf ("\nimage %d\n",i);
#endif
	for (int y=0;y<PIXEL_IMG;y++){
	    if (fread (bytes, 1, 1, file)!= 1){ 
		fclose(file);
		return true;
	    }
	    data[i][0][y]=(double)(bytes[0]);
#if DEBUGIO
	    if (data[i][0][y]<100)printf ("0 ");
	    if (100<data[i][0][y]&&data[i][0][y]<200)printf("- ");
	    if (data[i][0][y]>200)printf("1 ");
	    if (y%28==0)printf("\n");
#endif
	}
    }

    printf ("\nIMG file read with success!\n");
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
#if DEBUGIONN
    printf ("Saving neural net of size %ld!\n",LEN(NN));
#endif
    size_t written=0;
    FILE* file=NULL;
    file = fopen(filename, "wb+");
    if (file==NULL){ return false;}
    if (fwrite (&(LEN(NN)), sizeof(size_t), 1, file)!= 1){ 
	return false;
    }
    written+=sizeof(size_t);
#if DEBUGIONN
    printf ("=");
#endif
    if (fwrite (DPTH(NN), sizeof(size_t), LEN(NN), file)!=NN->len){
	printf ("failed to write depths!\n");
	return false;
    }
    written+=sizeof(size_t)*LEN(NN)*2;
    if (fwrite (FUNC(NN), sizeof(size_t), LEN(NN), file)!=NN->len){
	printf("failed to write functions!\n");
	return false;
    }
#if DEBUGIONN
    printf ("=");
#endif
    size_t toWrite=0;
    for (int x=0;x<X(W(NN));x++){
	toWrite+=Y(W(NN),x)*Z(W(NN),x);
#if DEBUGIONN
	printf ("%ld, %ld, %ld toWrite\n",Y(W(NN),x),Z(W(NN),x),toWrite);
#endif

    }
    if (fwrite (W(NN)->data, sizeof(double), toWrite, file)!=toWrite){
	printf("failed to write weights!\n");
	return false;
    }
    written+=toWrite*sizeof(double);
    toWrite=0;
    for (int x=0;x<X(B(NN));x++)toWrite+=Y(B(NN),x);
    if (fwrite (B(NN)->data, sizeof(double), toWrite, file)!=toWrite){
	printf ("failed to write Bias!\n");
	return false;
    }
    written+=toWrite*sizeof(double);
    if (fclose (file) == EOF){return false;}
#if DEBUGIONN
    printf (">Saved %ld bytes!\n",written);
#endif
    return true;
}
//read a nNetwork from a file
nNetwork* readNN(char* filename){
#if DEBUGIONN
    printf ("Loading neural networks!\n");
    fflush(stdout);
#endif
    size_t loaded=0;
    FILE* file=NULL;
    file = fopen(filename, "rb");
    if (file==NULL){ return NULL;}
    size_t len;
    if (fread (&len, sizeof(size_t), 1, file)!= 1){
	return NULL;
    }
    loaded=sizeof(size_t);
#if DEBUGIONN
    printf ("=");
    fflush(stdout);
#endif
    size_t* depths=(size_t*)malloc(len*sizeof(size_t));
    if (fread(depths, sizeof(size_t), len,file)!=len){
	free(depths);
	return NULL;
    }
    size_t* functions=malloc(len*sizeof(size_t));
    if (fread(functions, sizeof(size_t), len,file)!=len){
	free(depths);
	free(functions);
	return NULL;
    }
    loaded+=2*len*sizeof(size_t);
#if DEBUGIONN
    printf ("=");
    fflush(stdout);
#endif
    nNetwork* NN=createNN(len,depths,functions);
    size_t toRead=0;
    for (int x=0;x<X(W(NN));x++){
	toRead+=Y(W(NN),x)*Z(W(NN),x);
#if DEBUGIONN
	printf ("%ld, %ld, %ld toRead\n",Y(W(NN),x),Z(W(NN),x),toRead);
	fflush(stdout);
#endif
    }
    if (fread (W(NN)->data, sizeof(double), toRead, file)!=toRead){
	free(NN);
	free(depths);
	free(functions);
	return NULL;
    }
    loaded+=toRead*sizeof(double);
    toRead=0;
    for (int x=0;x<X(B(NN));x++){
	toRead+=Y(B(NN),x);
#if DEBUGIONN
	printf ("%ld, %ld, %ld toRead\n",Y(B(NN),x),Z(B(NN),x),toRead);
	fflush(stdout);
#endif
    }
    if (fread (B(NN)->data, sizeof(double), toRead, file)!=toRead){
	printf("failed to read Bias read %ld bytes!\n",loaded);
	free(NN);
	free(depths);
	free(functions);
	return NULL;
    }
    loaded+=toRead*sizeof(double);
    free(depths);
    free(functions);
    printf (">Loaded %ld bytes!\n",loaded);
    if (fclose (file)== EOF){
	freeNN(NN);
	return NULL;
    }
    return NN;
}

