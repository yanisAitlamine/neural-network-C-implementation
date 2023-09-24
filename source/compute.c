#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "compute.h"


// Softmax activation function
void softmax(nNetwork* NN, int layer) {
    double max_val = max_mtrx(M(ZN(NN),layer));
    double sum_exp = 0.0;
    add_to_mtrx(M(ACT(NN),layer),-max_val); 
    apply_on_mtrx(M(ACT(NN),layer),exp);
    for (int i = 0; i < DPTH(NN)[layer]; i++) {
        sum_exp +=  M(ACT(NN),layer)->data[i][0];
    }
    divide_mtrx(M(ACT(NN),layer),sum_exp);
}

// Derivative of softmax
void softmaxPrime(nNetwork *NN,int layer) {
    for (int y=0;y<Y(M(ZNP(NN),layer));y++){
        for (int i = 0; i < DPTH(NN)[layer]; i++) {
            if (i==y) {
                DATA(M(ZNP(NN),layer),y,0) += DATA(M(ACT(NN),layer),i,0) * (1.0 - DATA(M(ACT(NN),layer),i,0));
            }else{
                DATA(M(ZNP(NN),layer),y,0) += DATA(M(ACT(NN),layer),i,0) * (- DATA(M(ACT(NN),layer),y,0));
            }
        }
    }
}

void activation(nNetwork *NN, int layer){
    affect_values_mtrx_to_mtrx(M(ZN(NN),layer),M(ACT(NN),layer));
    int func=FUNC(NN)[layer];
    switch (func){
        case SIG:
            apply_on_mtrx(M(ACT(NN),layer),sigmoid);
        break;
        case RELU:
            apply_on_mtrx(M(ACT(NN),layer),Relu);
        break;
        case SOFT:
            softmax(NN,layer);
        break;
    }
}

void derivActivation(nNetwork *NN,int layer){
    switch (FUNC(NN)[layer]){
        case SIG:
            affect_values_mtrx_to_mtrx(M(ACT(NN),layer),M(ZNP(NN),layer));
            apply_on_mtrx(M(ZNP(NN),layer),sigmoidprime);
        break;
        case RELU:
            affect_values_mtrx_to_mtrx(M(ZN(NN),layer),M(ZNP(NN),layer));
            apply_on_mtrx(M(ZNP(NN),layer),Reluprime);
        break;
        case SOFT:
            init_mtrx(M(ZNP(NN),layer));
            softmaxPrime(NN,layer);
        break;
    }
}

void predict(mtrx *input,int y, nNetwork *NN){
    int i;
#if DEBUGCPT
    printf("inputs");
    print_list_m(input,x);
#endif
    //input is a matix of y row of z params, the first layer of activation is size zx1 this avoids a transposition by affecting directly
    for (i=0;i<Y(M(ACT(NN),0));i++)M(ACT(NN),0)->data[i][0]=input->data[y][i];
#if DEBUGCPT
    printf("\ninput copied to first layer\n");
    fflush(stdout);
#endif
    mtrx* buff,*buff2;
    for (i=0;i<LEN(NN)-1;i++){
        buff=get_transpose(M(W(NN),i));
        buff2=dot(buff,M(ACT(NN),i));
        //transpose(buff2,0);
        affect_values_mtrx_to_mtrx(buff2,M(ZN(NN),i+1));
        add_mtrx_to_mtrx(M(B(NN),i),M(ZN(NN),i+1));
        activation(NN,i+1);
#if DEBUGCPT
        printf("Activation rank %d\n",i+1);
        print_mtrx_v(ZN(NN),i);
        print_mtrx_v(ACT(NN),i);
        print_mtrx_v(W(NN),i);
        print_mtrx_v(B(NN),i);
        print_vector(buff2);
        print_mtrx_v(ZN(NN),i+1);
        print_mtrx_v(ACT(NN),i+1);
        fflush(stdout);
#endif
        free_mtrx(buff);
        free_mtrx(buff2);
    }
}

double sum_cost(double *expected, double **output, int len, int function){
    double local_cost=0;
    for (int i=0;i<len;i++){
        local_cost+=cost(expected[i],output[i][0],function);
        if (isnan(local_cost)){
            printf("\n%f+= cost(%f,%f,%d)\n",local_cost,expected[i],output[i][0],function);
            printf("%f=log(%f)\n",log(output[i][0]),output[i][0]);
            exit(1);
        }
    }
    return local_cost;
}

double multnode_cost(double *expected, mtrx *m, int function){
    int len=Y(m);
//outputs are the last len values in output->data
    switch (function){
        case MSE:
            return sum_cost(expected, m->data,len,SQR_REG)/len;
            break;
        case MAE:
            return sum_cost(expected, m->data,len,REGRESSION)/len;
            break;
        case MULTICLASS:
            return sum_cost(expected, m->data,len,MULTICLASS)/len;
            break;
    }
    return ERR_RETURN;
}

#define DEBUGGRD !true

void compute_grd(mtrx *expected, nNetwork *NN, int rank, int function){
    int i,x,y;
    int len=Y(M(ERR(NN),X(ERR(NN))-1));
    derivActivation(NN,LEN(NN)-1);
#if DEBUGGRD
    printf("\nGRD DEBUG\nExpected:\n");
#endif
    switch (function){
        case MULTICLASS:
            for (i=0;i<len;i++){
                DATA(M(ERR(NN),X(ERR(NN))-1),i,0)=DATA(expected,rank,i)-DATA(M(ACT(NN),X(ACT(NN))-1),i,0);
            }
        break;
        case BINARY:
            for (i=0;i<len;i++){
                DATA(M(ERR(NN),X(ERR(NN))-1),i,0)=DATA(M(ZNP(NN),X(ZNP(NN))-1),i,0)*binary_prime(DATA(expected,rank,i),DATA(M(ACT(NN),X(ACT(NN))-1),i,0));
            }
        break;
        case MSE:
        case MAE:
        case REGRESSION:
        case SQR_REG:
            for (i=0;i<len;i++){
                DATA(M(ERR(NN),X(ERR(NN))-1),i,0)=DATA(M(ZNP(NN),X(ZNP(NN))-1),i,0)*(DATA(expected,rank,i)-DATA(M(ACT(NN),X(ACT(NN))-1),i,0));
            }
        break;
    }
    add_mtrx_to_mtrx(M(ERR(NN),X(ERR(NN))-1),M(BGRD(NN),X(BGRD(NN))-1));
    mtrx *buff,*buff2;
    for (i=LEN(NN)-2;i>-1;i--){ 
        derivActivation(NN,i);
        buff=sum_W_Zn_Deriv(i,NN);
        affect_values_mtrx_to_mtrx(buff,M(ERR(NN),i));
        free_mtrx(buff);
        multiply_mtrx_by_mtrx(M(ZNP(NN),i),M(ERR(NN),i));
        if (i>0)add_mtrx_to_mtrx(M(ERR(NN),i),M(BGRD(NN),i-1));
        buff=get_transpose(M(ERR(NN),i+1));
        buff2=dot(M(ACT(NN),i), buff);
        add_mtrx_to_mtrx(buff2,M(WGRD(NN),i));
#if DEBUGGRD
    printf("\ngrd computation layer %d:\n",i);
    print_mtrx_v(ACT(NN),i);
    print_mtrx_v(ZNP(NN),i);
    print_mtrx_v(ERR(NN),i);
    if (i>0)print_mtrx_v(BGRD(NN),i-1);
    print_mtrx_v(WGRD(NN),i);
#endif

        free_mtrx(buff);
        free_mtrx(buff2);
    }
}

mtrx* sum_W_Zn_Deriv(int layer, nNetwork* NN){
    mtrx *result;
    result =dot(M(W(NN),layer), M(ERR(NN), layer+1));
    return result;
}

#define DEBUGTEST true
double test(nNetwork *NN, mtrx* test_input,mtrx *test_expected,size_t function){
    double *accuracy = malloc (sizeof(double)*Y(test_expected));
    double max_value_cost=0;
    for (int i=0;i<Y(test_expected);i++){
        predict (test_input,i, NN);
        //print_mtrx(M(ACT(NN),X(ACT(NN))-1));
        accuracy[i]=multnode_cost(test_expected->data[i],M(ACT(NN),X(ACT(NN))-1),function);
        if (isnan(accuracy[i])){
            printf("\n%d\n",i);
            print_vector(W(NN));
            exit(1);
        }
        if (accuracy[i]>max_value_cost) max_value_cost=accuracy[i];
    }
    double result = Relu(1-mean_double(accuracy,Y(test_expected)));
    free (accuracy);
    return result;
}

#define DEBUGB !true

void batch(mtrx *train_expected, mtrx *train_input, int rank, nNetwork* NN, int size_batch, double learning_rate, int function){
    init_vector(WGRD(NN));
    init_vector(BGRD(NN));
	for  (int i=0;i<size_batch;i++){
#if DEBUGB
    printf("\n BATCH DEBUG\nimage %d computing for training\n",rank+i);
    fflush(stdout);
#endif
            predict (train_input,rank+i, NN);
            compute_grd(train_expected,NN,rank+i,function);
#if DEBUGB
            printf ("\n\nBATCH DEBUG\nExpected [");
            printf("%.1f,%.1f",train_expected->data[rank+i][0],train_expected->data[rank+i][1]);
            printf("]\n");
            printACT(NN);     
            printERR(NN);
#endif
        }
	    divide_v(WGRD(NN),size_batch);
        divide_v(BGRD(NN),size_batch);
        updateNN(NN,learning_rate);
#if DEBUGB
        printGrd(NN);
        printNN(NN);
#endif
}

#define DEBUGCPT !true
void train(mtrx *train_expected, mtrx *train_input,mtrx *test_expected, mtrx *test_input, nNetwork* NN, int size_batch, double learning_rate, int function, int epochs){
   printf ("training for %d epochs over batch of size %d\n",epochs, size_batch);

    double current_accuracy;
    size_t size_data=Y(train_input), size_test=Y(test_input);
#if DEBUGCPT
    printf("\tlr: %f\tfunction: %d\tsize_data: %ld\tsize_test: %ld\n",learning_rate,function,size_data,size_test);
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
        if (epochs>10) {
            for (int y=0;y<=10;y++) {
                if (y*(epochs/10)==i){
                    current_accuracy=test(NN,test_input,test_expected,function);
                    printf(">epochs %d accuracy: %f\n",i,current_accuracy);
                    fflush(stdout);
                }
            }
        }else {
            current_accuracy=test(NN,test_input,test_expected,function);
            printf(">epochs %d accuracy: %f\n",i,current_accuracy);
        }
    }
    printf(">Finished\n");
}
