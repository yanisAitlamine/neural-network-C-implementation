#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "errors.h"
#include "neuralnet.h"


void copy_size_list(size_t *source,size_t* target, size_t len){
    for (int i=0;i<len; i++){
	target[i]=source[i];
    }
}
#define DEBUGINIT !true
// Create an object neural network of a given length with given layer lenghts
nNetwork* createNN(size_t len, size_t* depths,size_t* functions){
#if DEBUGINIT
	printf("Creating Network of size %ld!\n",len);
	fflush(stdout);
#endif	
	nNetwork* NN=(nNetwork*)malloc(sizeof(nNetwork));
	FF(NN)=false;
	LEN(NN)=len;
	DPTH(NN)=(size_t*)malloc(len*sizeof(size_t));
	if (check_malloc(DPTH(NN),"Depths init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	copy_size_list(depths,DPTH(NN),len);
	FUNC(NN)=malloc(len*sizeof(size_t));
	if (check_malloc(FUNC(NN),"function init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	copy_size_list(functions,FUNC(NN),len);
	mtrx_vector* v=create_vector(len-1,depths,&(depths[1]));
	W(NN)=v;
	if (check_malloc(W(NN),"Weights init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	WGRD(NN)=create_vector(len-1,depths,&(depths[1]));
	if (check_malloc(WGRD(NN),"WeightsGrd init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	size_t depth_static[len];
	for (int i=0;i<len;i++)depth_static[i]=1;
	B(NN)=create_vector(len-1,&(depths[1]),depth_static);
	if (check_malloc(B(NN),"Bias init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	BGRD(NN)=create_vector(len-1,&(depths[1]),depth_static);
	if (check_malloc(BGRD(NN),"biasGrd init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	ACT(NN)=create_vector(len,depths,depth_static);
	if (check_malloc(ACT(NN),"activations init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	ERR(NN)=create_vector(len,depths,depth_static);
	if (check_malloc(ACT(NN),"error init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	ZN(NN)=create_vector(len,depths,depth_static);
	if (check_malloc(ACT(NN),"zn init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
	ZNP(NN)=create_vector(len,depths,depth_static);
	if (check_malloc(ACT(NN),"znprim init failed!\n")){
	    FF(NN)=true;
	    return NN;
	}
//the activation table store the activation, the activation without the smoothing function, the derivative of C with regard to that node activation
#if DEBUGINIT
    printf("\nSuccesfully created!\n");
    fflush(stdout);
#endif
	return NN;
}

//Initialize weights and bias with random numbers
void fillNN(nNetwork* NN){
    init_vector_rand(W(NN));
    init_vector_rand(B(NN));
    init_vector(WGRD(NN));
    init_vector(BGRD(NN));
    init_vector(ACT(NN));
    init_vector(ERR(NN));
    init_vector(ZN(NN));
    init_vector(ZNP(NN));
}

//Initialize weights and bias with random numbers
void initGRD(nNetwork* NN){
    init_vector(WGRD(NN));
    init_vector(BGRD(NN));
}

//Update weights and bias with Grd and learing rate
void updateNN(nNetwork* NN, double learning_rate){
    multiply_vector(WGRD(NN),learning_rate);
    for (int x=0;x<LEN(NN)-1;x++)add_mtrx_mtrx(WGRD(NN),W(NN),x,x);
    multiply_vector(BGRD(NN),learning_rate);
    for (int x=0;x<LEN(NN)-1;x++)add_mtrx_mtrx(BGRD(NN),B(NN),x,x);
}


//Print weights and bias
void printNN(nNetwork* NN){
    printf ("\nPrinting neural net of size %ld!\n",NN->len);
    print_vector(W(NN));
    print_vector(B(NN));
}


//Print weights and bias Grd
void printGrd(nNetwork* NN){
    printf ("\nPrinting neural net Grd of size %ld!\n",LEN(NN));
    print_vector(WGRD(NN));
    print_vector(BGRD(NN)); 
}

void printACT(nNetwork* NN){
    printf ("\nPrinting neural net activations of size %ld!\n",NN->len);
    print_vector(ACT(NN));
}

void printERR(nNetwork* NN){
    printf ("\nPrinting neural net activations of size %ld!\n",NN->len);
    print_vector(ERR(NN));
}

void printZN(nNetwork* NN){
    printf ("\nPrinting neural net activations of size %ld!\n",NN->len);
    print_vector(ZN(NN));
}

void printZNP(nNetwork* NN){
    printf ("\nPrinting neural net activations of size %ld!\n",NN->len);
    print_vector(ZNP(NN));
}

#define DEBUGFREE false
// Free a neural network object
void freeNN(nNetwork* NN){
    if (NN==NULL){return;}
#if DEBUGFREE
    printf ("Free network of size %ld!\n",LEN(NN));
#endif
    free_vector(W(NN));
    free_vector(WGRD(NN));
    free_vector(B(NN));
    free_vector(BGRD(NN));
    free_vector(ACT(NN));
    free_vector(ERR(NN));
    free_vector(ZN(NN));
    free_vector(ZNP(NN));
    free(DPTH(NN));
    free(FUNC(NN));
    free(NN);
}

