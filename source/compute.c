#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "compute.h"
#include "utils.h"
#define DEBUGCPT !false

// Softmax activation function
void softmax(nNetwork* NN, int layer) {
    double max_val = max_mtrx(ZN(NN),layer);

    double sum_exp = 0.0;
    add_mtrx(ACT(NN),layer,-max_val); 
    for (int i = 0; i < DPTH(NN)[layer]; i++) {
        sum_exp +=  DATA(ACT(NN),get_index(ACT(NN),layer,i,0));
    }

    exp_mtrx(ACT(NN),layer);
    divide_mtrx(ACT(NN),layer,sum_exp);
}

// Derivative of softmax
void softmaxPrime(nNetwork *NN,int layer) {
    for (int y=0;y<Y(ZNP(NN),layer);y++){
        for (int i = 0; i < DPTH(NN)[layer]; i++) {
            if (i==y) {
                DATA(ZNP(NN),get_index(ZNP(NN),layer,y,0)) += DATA(ACT(NN),get_index(ACT(NN),layer,i,0)) * (1.0 - DATA(ACT(NN),get_index(ACT(NN),layer,i,0)));
            }else{
                DATA(ZNP(NN),get_index(ZNP(NN),layer,y,0)) += DATA(ACT(NN),get_index(ACT(NN),layer,i,0)) * (- DATA(ACT(NN),get_index(ACT(NN),layer,y,0)));
            }
        }
    }
}

void activation(nNetwork *NN, int layer){
    affect_values_vx_vxp(ZN(NN),ACT(NN),layer,layer);
    switch (FUNC(NN)[layer]){
        case SIG:
            sigmoid_mtrx(ACT(NN),layer);
        break;
        case RELU:
            Relu_mtrx(ACT(NN),layer);
        break;
        case SOFT:
            softmax(NN,layer);
        break;
    }
}

void derivActivation(nNetwork *NN,int layer){
    switch (FUNC(NN)[layer]){
        case SIG:
            affect_values_vx_vxp(ACT(NN),ZNP(NN),layer,layer);
            sigmoidP_mtrx(ZNP(NN),layer);
        break;
        case RELU:
            affect_values_vx_vxp(ZN(NN),ZNP(NN),layer,layer);
            ReluP_mtrx(ZNP(NN),layer);
        break;
        case SOFT:
            init_mtrx(ZNP(NN),layer);
            softmaxPrime(NN,layer);
        break;
    }
}

void predict(mtrx *input,int x, nNetwork *NN){
    int i;
#if DEBUGCPT
    printf("inputs\n");
    print_list_m(input,x);
#endif
    affect_values_mx_vxp(input,ACT(NN),x,0);
#if DEBUGCPT
    printf("input copied to first layer\n");
    fflush(stdout);
#endif
    mtrx_vector* buff;
    for (i=0;i<LEN(NN)-1;i++){
        buff=dot(W(NN),ACT(NN),i,i);
        affect_values_vx_vxp(buff,ZN(NN),0,i+1);
        add_mtrx_mtrx(B(NN),ZN(NN),i,i+1);
        activation(NN,i+1);
#if DEBUGCPT
        print_mtrx_v(ACT(NN),i+1);
#endif
    }
    free_vector(buff);
}

double sum_cost(double *expected, double *output, int x, int len, int function){
    double local_cost=0;
    for (int i=0;i<len;i++){
        local_cost+=cost(expected[i],output[i+x],function);
    }
    return local_cost;
}

double MSE_cost(double* expected, double* output, int x, int len){
    return sum_cost(expected, output,x,len,SQR_REG)/len;
}

double MAE_cost(double* expected, double* output, int x, int len){
    return sum_cost(expected, output,x,len,REGRESSION)/len;
}

double multiclass_cost(double* expected, double* output, int x, int len){
    return sum_cost(expected, output,x,len,MULTICLASS);
}

double multnode_cost(double *expected, mtrx_vector *v, int function){
    int len=Y(v,X(v)-1),x=total_size(v)-len;
//outputs are the last len values in output->data

    switch (function){
        case MSE:
            return MSE_cost(expected,v->data,len,x);
            break;
        case MAE:
            return MAE_cost(expected,v->data,len,x);
            break;
        case MULTICLASS:
            return multiclass_cost(expected,v->data,len,x);
            break;
    }
    return ERR_RETURN;
}

void compute_grd(double *expected, nNetwork *NN, int function){
    int i,x,y;
    int len=Y(ERR(NN),X(ERR(NN))-1),start=total_size(ERR(NN))-len;
    derivActivation(NN,LEN(NN)-1);
    for (i=0;i<len;i++){
        switch (function){
            case MULTICLASS:
                DATA(ERR(NN),i+start)=expected[i]-DATA(ACT(NN),i+start);
            break;
            case BINARY:
                DATA(ERR(NN),i+start)=DATA(ZNP(NN),i+start)*binary_prime(expected[i],DATA(ACT(NN),i+start));
            break;
            case MSE:
            case MAE:
            case REGRESSION:
            case SQR_REG:
                DATA(ERR(NN),i+start)=DATA(ZNP(NN),i+start)*binary_prime(expected[i],DATA(ACT(NN),i+start));
            break;
        }
    }
    affect_values_vx_vxp(ERR(NN),BGRD(NN),X(ERR(NN))-1,X(BGRD(NN))-1);
    mtrx_vector *buff,*buff2;
    for (i=LEN(NN)-2;i>-1;i--){ 
        derivActivation(NN,i);
        buff=sum_W_Zn_Deriv(i,NN);
        affect_values_vx_vxp(buff,ERR(NN),0,i);
        free_vector(buff);
        multiply_mtrx_mtrx(ZNP(NN),ERR(NN),i,i);
        if (i>0)affect_values_vx_vxp(ERR(NN),BGRD(NN),i,i-1);
        buff=get_transpose(ERR(NN), i+1);
        buff2=dot(buff, ACT(NN),0,i);
        affect_values_vx_vxp(buff2,W(NN),0,i);
        free_vector(buff);
        free_vector(buff2);
    }
}

mtrx_vector* sum_W_Zn_Deriv(int layer, nNetwork* NN){
    mtrx_vector *buff,*result;
    buff=get_transpose(ERR(NN),layer);
    result =dot(buff, W(NN), 0, layer);
    free_vector(buff);
    return result;
}

double test(nNetwork *NN, mtrx* test_input,mtrx *test_expected,size_t function){
    double costs[X(test_expected)];
    double* expected; 
    for (int i=0;i<X(test_expected);i++){
        predict (test_input,i, NN);
        expected=get_list_from_m(test_expected,i);
        costs[i]=multnode_cost(expected,ACT(NN),function);
#define DEBUGTEST false
#if DEBUGTEST 
        printf ("Testing\noutput:");
        print_mtrx(ACT(NN),X(ACT(NN))-1);
        printf ("\nExpected [");
        for (int y=0;y<DPTH(NN)[LEN(NN)-1];y++){
            printf("%.1f ",expected[y]);
        }
        printf("]\ncosts: ");
    	printf("%f\n",costs[i]);
#endif
        free(expected);
    }
    return mean_double(costs,X(test_expected));
}

#define DEBUGB !true

void batch(mtrx *train_expected, mtrx *train_input, int rank, nNetwork* NN, int size_batch, double learning_rate, int function){
	double* expected;
	for  (int i=0;i<size_batch;i++){
#if DEBUGB
    printf("computing for training\n");
    fflush(stdout);
#endif
            predict (train_input,rank+i, NN);
            expected=get_list_from_m(train_expected,rank+i);
            compute_grd(expected,NN,function);
#if DEBUGB
            if(rank<5*size_batch&&i==0){
                printf ("\n\nExpected [");
                for (int y=0;y<DPTH(NN)[LEN(NN)-1];y++){
                    printf("%.1f ",expected[y]);
                }
                printf("]\n");
                printACT(NN);
                printERR(NN);
                printGrd(NN);
            }
#endif
            free(expected);
        }
	//multiply_grd(NN, pow(size_batch,-1));
        updateNN(NN,learning_rate);
}

void train(mtrx *train_expected, mtrx *train_input,mtrx *test_expected, mtrx *test_input, nNetwork* NN, int size_batch, double learning_rate, int function, int epochs){
   printf ("training for %d epochs over batch of size %d\n",epochs, size_batch);

    double current_cost;
    size_t size_data=X(train_input), size_test=X(test_input);
#if DEBUGCPT
    printf("\t%f\t%d\t%ld\t%ld\n",learning_rate,function,size_data,size_test);
    fflush(stdout);
#endif
    for (int i=0;i<epochs;i++){
        for (int x=0;x<size_data;x+=size_batch){
            if (x>size_data-size_batch){size_batch=size_data-x;}
#if DEBUGCPT
            printf ("batch nb %d\n",x/size_batch);
            fflush(stdout);
#endif
            batch(train_expected,train_input,x,NN,size_batch,learning_rate,function);
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
                current_cost=test(NN,test_input,test_expected,function);
               printf(">epochs %d cost: %f\n",i,current_cost);
            }
        }
    }
    printf(">Finished\n");
}
