#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "compute.h"
#define change
double sigmoid(double n){
    return (1/(1+pow(EULER_NUMBER, -n)));
}

double sigmoidprime(double n){
    return (n)*(1-(n));
}

void compute(double *input, nNetwork *NN,bool debug){
    size_t maxsize=0;
    if (debug){printf ("Computing with inputs: [");
        for ( int i=0;i<DPTH(NN)[0];i++){
            printf ("%.1f",input[i]);
            if (i<DPTH(NN)[0]-1){
                printf (", ");
            } else {
                printf ("]");
            }
        }
    }
    if (debug)printf ("=");
    for (int i=0;i<DPTH(NN)[0];i++){ACT(NN)[0][i][0]=input[i];}
    for (int i=0;i<LEN(NN)-1;i++){
        for (int y=0;y<DPTH(NN)[i+1];y++){
            ACT(NN)[i+1][y][ZN]=B(NN)[i][y];
            for (int x=0;x<DPTH(NN)[i];x++){
                ACT(NN)[i+1][y][ZN]+=(NN->activations[i][x][0]*W(NN)[i][x][y]); 
            }
            ACT(NN)[i+1][y][AN]=sigmoid(NN->activations[i+1][y][ZN]);
        } 
    }
    if (debug){
            for (int x=0;x<LEN(NN);x++){
                printf ("\nlayer %d: ",x);
                for (int y=0;y<DPTH(NN)[x];y++){
                    printf("%.1f\t",ACT(NN)[x][y][AN]);
                }
                printf ("|");
            }
        }
    if (debug)printf (">Finished\n");
}

//split data set if entries and expected are a simple list of activations
void splitData(int num_obj, int len_in, int len_out, double ***train_data, double*** input, double*** expected){
    	for (int i=0;i<num_obj;i++){
	    (*input)[i]=(double*)malloc(len_in*sizeof(double));
	    for (int y=0;y<len_in;y++){
                (*input)[i][y]=train_data[i][0][y];
            }
	}
	for (int i=0;i<num_obj;i++){
		(*expected)[i]=(double*)malloc(len_out*sizeof(double));
		for (int y=0;y<len_out;y++){(*expected)[i][y]=train_data[i][1][y];}
	}
}

void printTrainData(double** expected, double** input,int len_data,int depthinput, double depthoutput){
	for (int i=0;i<len_data;i++){
		printf ("\nInput %d: ",i);
		for (int y=0;y<depthinput;y++) printf("%.1f\t",input[i][y]);
		printf ("\nExpected %d: ",i);
		for (int y=0;y<depthoutput;y++) printf("%.1f\t",expected[i][y]);
	}
}

// compute cost
double regression_cost(double expected, double output){
    return expected-output;
}

double sqr_regression(double expected, double output){
    return pow(expected-output,2)/2;
}

// case expected = 0 output=1 handled, would divide by 0
double binary_prime(double expected, double output){
    if (expected == output){ return 0;}
    if (output!=1&&output!=0){return (expected/output)-((1-expected)/(1-output));}
    return (expected/output+0.000001)-((1-expected)/(1-output+0.000001));
}

double sqr_prime(double expected, double output){
    return -(expected-output);
}

// compute binary cost
double binary_cost(double expected, double output){
    return -((expected*log(output))+((1-expected)*log(1-output)));
}

double cost (double expected, double output, int function){
    switch(function){
        case REGRESSION:
            return regression_cost(expected,output);
        break;
        case SQR_REG:
            return sqr_regression(expected,output);
            break;
        case BINARY:
            return binary_cost(expected,output);
        break;
    }
    return ERR;
}

double sum_cost(double *expected, double **output, int len, int function){
    double local_cost=0;
    for (int i=0;i<len;i++){
        local_cost+=cost(expected[i],output[i][AN],function);
    }
    return local_cost;
}

double MSE_cost(double* expected, double** output, int len){
    return sum_cost(expected, output,len,SQR_REG)/len;
}

double MAE_cost(double* expected, double** output, int len){
    return sum_cost(expected, output,len,REGRESSION)/len;
}

double multiclass_cost(double* expected, double** output, int len){
    return sum_cost(expected, output,len,BINARY);
}

double multnode_cost(double *expected, double **output, int len, int function){
    switch (function){
        case MSE:
            return MSE_cost(expected,output,len);
            break;
        case MAE:
            return MAE_cost(expected,output,len);
            break;
        case MULTICLASS:
            return multiclass_cost(expected,output,len);
            break;
    }
    return ERR;
}

void compute_grd(double *expected, nNetwork *NN, int function, bool debug){
    if (debug)printf ("Computing Gradient!\n");
    for (int i=LEN(NN)-2;i>-1;i--){
        if (debug)printf ("layer %d\t",i+1);
        for (int x=0;x<DPTH(NN)[i+1];x++){
            ACT(NN)[i+1][x][DERIV]=0;
            ACT(NN)[i+1][x][ZNPRIME]=sigmoidprime(NN->activations[i+1][x][AN]);
            if (i==LEN(NN)-2){ 
                switch (function){
                    case MULTICLASS:
                    case BINARY:
                        ACT(NN)[i+1][x][DERIV]=NN->activations[i+1][x][ZNPRIME]*binary_prime(expected[x],NN->activations[i+1][x][AN]);
                    break;
                    case MSE:
                    case MAE:
                    case REGRESSION:
                    case SQR_REG:
                        ACT(NN)[i+1][x][DERIV]=-(NN->activations[i+1][x][ZNPRIME]*sqr_prime(expected[x],NN->activations[i+1][x][AN]));
                    break;
                }
            }else{
                ACT(NN)[i+1][x][DERIV]=sum_W_Zn_Deriv(i+1,x,NN)*NN->activations[i+1][x][ZNPRIME];
            }
            if (debug)printf ("Error%d: %f, ZnPRIME:%f\t",x,ACT(NN)[i+1][x][DERIV],NN->activations[i+1][x][ZNPRIME]);
            BGRD(NN)[i][x]+=ACT(NN)[i+1][x][DERIV];
        }
        if (debug)printf ("\n");
        for (int x=0;x<DPTH(NN)[i];x++){
            for (int y=0;y<DPTH(NN)[i+1];y++){
                WGRD(NN)[i][x][y]+=ACT(NN)[i][x][AN]*NN->activations[i+1][y][DERIV];               
            }
        }
        if (debug)printf ("\n");
    }
    if (debug)printf ("\n");
}

double sum_W_Zn_Deriv(int rank, int ndnum, nNetwork* NN){
    double result = 0;
    for (int i=0;i<DPTH(NN)[rank+1];i++){
                 result+=W(NN)[rank][ndnum][i]*ACT(NN)[rank+1][i][DERIV];   
    }
    return result;
}

void batch(double **expected, double **input, nNetwork* NN, int size_batch, double learning_rate, int function, bool debug){
	multiply_grd(NN, 0);
	for  (int i=0;i<size_batch;i++){
		compute (input[i], NN,debug);
		compute_grd(expected[i],NN,function,debug);
	}
	multiply_grd(NN, pow(size_batch,-1));
        if (debug) printNNGrd(NN);
	updateNN(NN,learning_rate,debug);
	if (debug)printNN(NN);
}

void train(double **expected, double **input, nNetwork* NN, int size_batch, double learning_rate, int function, int epochs, bool debug){
   printf ("training for %d epochs over batch of size %d\n",epochs, size_batch);
    for (int i=0;i<=epochs;i++){
        batch(expected,input,NN,size_batch,learning_rate,function,false);
        for (int y=0;y<=10;y++) {
            if (y*(epochs/10)==i) {
                printf("=");
                if (debug){
                    printf("epochs nb %d\n",i);
                    batch(expected,input,NN,size_batch,learning_rate,function,true);
                }
            }
        }
    }
    printf(">Finished\n");
}

void shuffle(double*** data,int len,int depth_in,int depth_out,int rounds){
    for (int a=0;a<rounds;a++){
        for (int i=0;i<len;i++){
            swapTables(data,i,len*(rand()/RAND_MAX),depth_in,depth_out);
        }
    }
}

void swapTables(double ***data,int base,int target,int depth_in,int depth_out){
    double buffer;
    for (int i=0;i<depth_in;i++){
        buffer=data[base][0][i];
        data[base][0][i]=data[target][0][i];
        data[target][0][i]=buffer;
    }
    for (int i=0;i<depth_out;i++){
        buffer=data[base][1][i];
        data[base][1][i]=data[target][1][i];
        data[target][1][i]=buffer;
    }

}
