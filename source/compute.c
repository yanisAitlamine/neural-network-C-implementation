#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "compute.h"

double sigmoid(double n){
    return (1/(1+pow(EULER_NUMBER, -n)));
}

double sigmoidprime(double n){
    return (sigmoid(n)*(1-sigmoid(n)));
}

void compute(double *input, double **output, nNetwork *NN){
    size_t maxsize=0;
    printf ("Computing with inputs: [");
    for ( int i=0;i<NN->depths[0];i++){
        printf ("%.1f",input[i]);
        if (i<NN->depths[0]-1){
         printf (", ");
        } else {
         printf ("]");
        }
    }
    printf ("=");
    for (int i=0;i<NN->depths[0];i++){NN->activations[0][i][0]=input[i];}
    for (int i=0;i<NN->len-1;i++){
        printf ("=");
        for (int y=0;y<NN->depths[i+1];y++){
            NN->activations[i+1][y][0]=0;
            NN->activations[i+1][y][1]=0;
            for (int x=0;x<NN->depths[i];x++){
                NN->activations[i+1][y][1]+=(NN->activations[i][x][0]*NN->weights[i][x][y]); 
            }
            NN->activations[i+1][y][0]=sigmoid(NN->activations[i+1][y][1]+NN->bias[i][y]);
        }
    }
    for (int i=0;i<NN->depths[NN->len-1];i++){
        (*output)[i]=NN->activations[NN->len-1][i][0];
    }
    printf (">Finished\n");
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

// compute cost
double regression_cost(double expected, double output){
    return expected-output;
}

double sqr_regression(double expected, double output){
    return pow(expected-output,2)/2;
}

double binary_prime(double expected, double output){
    return (expected/output)-((1-expected)/(1-output));
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

double sum_cost(double *expected, double *output, int len, int function){
    double local_cost=0;
    for (int i=0;i<len;i++){
        local_cost+=cost(expected[i],output[i],function);
    }
    return local_cost;
}

double MSE_cost(double* expected, double* output, int len){
    return sum_cost(expected, output,len,SQR_REG)/len;
}

double MAE_cost(double* expected, double* output, int len){
    return sum_cost(expected, output,len,REGRESSION)/len;
}

double multiclass_cost(double* expected, double* output, int len){
    return sum_cost(expected, output,len,BINARY);
}

double multnode_cost(double *expected, double *output, int len, int function){
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
    printf ("Computing Gradient!\n");
    for (int i=NN->len-2;i>-1;i--){
        printf ("layer %d\t",i+1);
        for (int x=0;x<NN->depths[i+1];x++){
            NN->activations[i+1][x][DERIV]=0;
            NN->activations[i+1][x][ZNPRIME]=sigmoidprime(NN->activations[i+1][x][ZN]);
            if (i==NN->len-2){ 
                switch (function){
                    case MULTICLASS:
                    case BINARY:
                        NN->activations[i+1][x][DERIV]=binary_prime(expected[x],NN->activations[i+1][x][AN]);
                    break;
                    case MSE:
                    case SQR_REG:
                        NN->activations[i+1][x][DERIV]=sqr_prime(expected[x],NN->activations[i+1][x][AN]);
                    break;
                    case MAE:
                    case REGRESSION:
                        NN->activations[i+1][x][DERIV]=-1;
                    break;
                }
            }else{
                NN->activations[i+1][x][DERIV]=sum_W_Zn_Deriv(i+1,x,NN);
            }
            printf ("dC/dA%d: %f, ZnPRIME:%f\t",x,NN->activations[i+1][x][DERIV],NN->activations[i+1][x][ZNPRIME]);
            NN->biasGrd[i][x]+=NN->activations[i+1][x][ZNPRIME]*NN->activations[i+1][x][DERIV];
        }
        printf ("\n");
        for (int x=0;x<NN->depths[i];x++){
            for (int y=0;y<NN->depths[i+1];y++){
                NN->weightsGrd[i][x][y]+=NN->activations[i][x][AN]*NN->activations[i+1][y][ZNPRIME]*NN->activations[i+1][y][DERIV];               
            }
        }
        printf ("\n");
    }
    printf ("\n");
}

double sum_W_Zn_Deriv(int rank, int ndnum, nNetwork* NN){
    double result = 0;
    for (int i=0;i<NN->depths[rank+1];i++){
                 result+=NN->weights[rank][ndnum][i]*NN->activations[rank+1][i][ZNPRIME]*NN->activations[rank+1][i][DERIV];   
    }
    for (int i=rank+1;i<NN->len-1;i++){
        for (int x=0;x<NN->depths[i];x++){
            for (int y=0;y<NN->depths[i+1];y++){
                result+=NN->weights[i][x][y]*NN->activations[i+1][y][ZNPRIME]*NN->activations[i+1][y][DERIV];
                 }
            }
    }
    return result;
}

void batch(double **expected, double **input, double **output, nNetwork* NN, int size_batch, double learning_rate, int function){
	multiply_grd(NN, 0);
	for  (int i=0;i<size_batch;i++){
		compute (input[i], &(output[i]), NN);
		printf ("output: [");
		printf ("[");
		for (int y=0;y<NN->depths[NN->len-1];y++){
			printf("%f",output[i][y]);
			if (y<NN->depths[NN->len-1]-1){
				printf (", ");
			}
		}
		if (i<size_batch-1){
			printf ("],");
		} else {
			printf ("]");
		}
		printf ("]\n");
		printf ("costs: [");
		printf("%f",multnode_cost(expected[i],output[i],NN->depths[NN->len-1],function));
		printf ("]\n");
		compute_grd(expected[i],NN,function);
	}
	multiply_grd(NN, pow(size_batch,-1));
	printNNGrd(NN);
	updateNN(NN,learning_rate);
	printNN(NN);
}
