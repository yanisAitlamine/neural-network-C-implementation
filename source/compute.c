#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "compute.h"

double sigmoid(double n){
    return (1/(1+pow(EULER_NUMBER, -n)));
}

double sigmoidprime(double n){
    if (n==1)return 0.0001;
    if (n==0)return -0.0001;
    return (n)*(1-(n));
}

double Relu(double n){
    if(n<0)return 0;
    return n;
}

double ReluPrime(double n){
    if (n)return 1;
    return 0;
}

// Softmax activation function
void softmax(nNetwork* NN, int rank) {
    double max_val = ACT(NN)[rank][0][ZN];
    for (int i = 1; i < DPTH(NN)[rank]; i++) {
        if (ACT(NN)[rank][i][ZN] > max_val) {
            max_val = ACT(NN)[rank][i][ZN];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < DPTH(NN)[rank]; i++) {
        ACT(NN)[rank][i][AN] = exp( ACT(NN)[rank][i][ZN]- max_val);
        sum_exp +=  ACT(NN)[rank][i][AN];
    }

    for (int i = 0; i < DPTH(NN)[rank]; i++) {
         ACT(NN)[rank][i][AN] /= sum_exp;
    }
}

// Derivative of softmax
void softmaxPrime(nNetwork *NN,int rank) {
    for (int i = 0; i < DPTH(NN)[rank]; i++) {
        ACT(NN)[rank][i][DERIV] = ACT(NN)[rank][i][AN] * (1.0 - ACT(NN)[rank][i][AN]);
    }
}

void activation(nNetwork *NN, int rank){
    int y;
    switch (FUNC(NN)[rank]){
        case SIG:
            for(y=0;y<DPTH(NN)[rank];y++){
                ACT(NN)[rank][y][AN]=sigmoid(ACT(NN)[rank][y][ZN]);

            }
        break;
        case RELU:
            for(y=0;y<DPTH(NN)[rank];y++){
                ACT(NN)[rank][y][AN]=Relu(ACT(NN)[rank][y][ZN]);
            }
        break;
        case SOFT:
            softmax(NN,rank);
        break;
    }
}

void derivActivation(nNetwork *NN,int rank){
    int y;
    switch (FUNC(NN)[rank]){
        case SIG:
            for(y=0;y<DPTH(NN)[rank];y++){
                ACT(NN)[rank][y][ZNPRIME]=sigmoidprime(ACT(NN)[rank][y][AN]);

            }
        break;
        case RELU:
            for(y=0;y<DPTH(NN)[rank];y++){
                ACT(NN)[rank][y][ZNPRIME]=ReluPrime(ACT(NN)[rank][y][AN]);
            }
        break;
        case SOFT:
            softmaxPrime(NN,rank);
        break;
    }
}

void compute(double *input, nNetwork *NN){
    int i,y,x;
    for (i=0;i<DPTH(NN)[0];i++){ACT(NN)[0][i][0]=input[i];}
    for (i=1;i<LEN(NN);i++){
        for (y=0;y<DPTH(NN)[i];y++){
            ACT(NN)[i][y][ZN]=B(NN)[i-1][y];
            for (x=0;x<DPTH(NN)[i-1];x++){
                ACT(NN)[i][y][ZN]+=(NN->activations[i-1][x][0]*W(NN)[i-1][x][y]); 
            }
        } 
        activation(NN,i);
    }
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

void normalize(double** input,int size_data,int len_row, double max){
    int i,y;
    for (i=0;i<size_data;i++){
        for (y=0;y<len_row;y++){
         input[i][y]/=max;
        }
    }
}

void printData(double** expected, double** input,int len_data,int depthinput, double depthoutput){
    for (int i=0;i<len_data;i++){
	printf ("\nInput %d: ",i);
	for (int y=0;y<depthinput;y++) printf("%.1f\t",input[i][y]);
	printf ("\nExpected %d: ",i);
        for (int y=0;y<depthoutput;y++) printf("%.1f\t",expected[i][y]);
    }
    printf("\n");
}

// compute cost
double regression_cost(double expected, double output){
    return expected-output;
}

double sqr_regression(double expected, double output){
    return pow(expected-output,2);
}

// case expected = 0 output=1 handled, would divide by 0
double binary_prime(double expected, double output){
    if (expected == output){ return 0;}
    if (output!=1&&output!=0){return (expected/output)-((1-expected)/(1-output));}
    return (expected/output+0.000001)-((1-expected)/(1-output+0.000001));
}

double sqr_prime(double expected, double output){
    return -2*(expected-output);
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
        case MULTICLASS:
            if(output==0)output=0.0000001;
            return -expected*log(output);
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
    return sum_cost(expected, output,len,MULTICLASS);
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

void compute_grd(double *expected, nNetwork *NN, int function){
    int i,x,y;
    derivActivation(NN,LEN(NN)-1);
    for (i=0;i<DPTH(NN)[LEN(NN)-1];i++){
        switch (function){
            case MULTICLASS:
                ACT(NN)[LEN(NN)-1][i][DERIV]=expected[i]-NN->activations[LEN(NN)-1][i][AN];
            break;
            case BINARY:
                ACT(NN)[LEN(NN)-1][i][DERIV]=NN->activations[LEN(NN)-1][i][ZNPRIME]*binary_prime(expected[i],NN->activations[LEN(NN)-1][i][AN]);
            break;
            case MSE:
            case MAE:
            case REGRESSION:
            case SQR_REG:
                ACT(NN)[LEN(NN)-1][i][DERIV]=-(NN->activations[LEN(NN)-1][i][ZNPRIME]*sqr_prime(expected[i],NN->activations[LEN(NN)-1][i][AN]));
            break;
        }
        BGRD(NN)[LEN(NN)-2][i]+=ACT(NN)[LEN(NN)-1][i][DERIV];
    }
    for (i=LEN(NN)-2;i>-1;i--){ 
        derivActivation(NN,i);
        for (x=0;x<DPTH(NN)[i];x++){
            ACT(NN)[i][x][DERIV]=sum_W_Zn_Deriv(i,x,NN)*NN->activations[i][x][ZNPRIME];
            if (i>0)BGRD(NN)[i-1][x]+=ACT(NN)[i][x][DERIV];
        }
        for (x=0;x<DPTH(NN)[i];x++){
            for (y=0;y<DPTH(NN)[i+1];y++){
                WGRD(NN)[i][x][y]+=ACT(NN)[i][x][AN]*NN->activations[i+1][y][DERIV];               
            }
        }
    }
}

double sum_W_Zn_Deriv(int rank, int ndnum, nNetwork* NN){
    double result = 0;
    for (int i=0;i<DPTH(NN)[rank+1];i++){
                 result+=W(NN)[rank][ndnum][i]*ACT(NN)[rank+1][i][DERIV];   
    }
    return result;
}

double sum_double (double* data, int size_data){
    double result=0;
    for (int i=0;i<size_data;i++){
        result+=data[i];
    }
    return result;
}

double mean_double(double* data,int size_data){
    return sum_double(data,size_data)/size_data;
}

double test(nNetwork* NN, double** test_input,double **test_expected,int size_data){
    double costs[size_data];
    for (int i=0;i<size_data;i++){
        compute (test_input[i], NN);
        costs[i]=multnode_cost(test_expected[i],NN->activations[NN->len-1],NN->depths[NN->len-1],MULTICLASS);
#if DEBUG 
        printf ("Testing\noutput[");
        for (int y=0;y<NN->depths[NN->len-1];y++){
        printf("%f",NN->activations[NN->len-1][y][AN]);
	    if (y<NN->depths[NN->len-1]-1){
                printf (", ");
            }

        }
        printf ("]\nExpected [");
        for (int y=0;y<DPTH(NN)[LEN(NN)-1];y++){
            printf("%.1f ",test_expected[i][y]);
        }
        printf("]\ncosts: ");
    	printf("%f\n",costs[i]);
#endif
    }
    return mean_double(costs,size_data);
}

void batch(double **expected, double **input,int rank, nNetwork* NN, int size_batch, double learning_rate, int function){
	multiply_grd(NN, 0);
	for  (int i=0;i<size_batch;i++){
            compute (input[i+rank], NN);
            compute_grd(expected[i+rank],NN,function);
#if DEBUGCPT
            printf ("\n\nExpected [");
            for (int y=0;y<DPTH(NN)[LEN(NN)-1];y++){
                printf("%.1f ",expected[i+rank][y]);
            }
            printf("]\n");
            printACT(NN);
            printERROR(NN);
            printGrd(NN);
#endif
        }
	multiply_grd(NN, pow(size_batch,-1));
        updateNN(NN,learning_rate);
}

void train(double **expected, double **input,double **test_expected, double **test_input, nNetwork* NN, int size_data, int size_batch,int size_test, double learning_rate, int function, int epochs){
   printf ("training for %d epochs over batch of size %d\n",epochs, size_batch);
    double current_cost;
    for (int i=0;i<epochs;i++){
        for (int x=0;x<size_data;x+=size_batch){
            if (x>size_data-size_batch){size_batch=size_data-x;}
            batch(expected,input,x,NN,size_batch,learning_rate,function);
            for(int z=0;z<=10;z++){
                for (int y=0;y<=10;y++) {
                    if (y*(size_data/10)==x&&z*(epochs/10)==i) {
                        printf("=");
                        fflush(stdout);              
                    }
                }
            }
        }
        for (int y=0;y<=10;y++) {
            if (y*(epochs/10)==i) {
                current_cost=test(NN,test_input,test_expected,size_test);
               printf(">epochs %d cost: %f\n",i,current_cost);
            }
        }
    }
    printf(">Finished\n");
}

void shuffle(double*** data,int len,int depth_in,int depth_out,int rounds){
    for (int a=0;a<rounds;a++){
        for (int i=0;i<len;i++){
            swapTables(data,i,(len-1)*(rand_double()),depth_in,depth_out);
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
