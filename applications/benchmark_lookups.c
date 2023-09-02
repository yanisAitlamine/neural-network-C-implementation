#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../source/utils.h"
#include "../source/neuralnet.h"
#include "../source/errors.h"
#include "../source/in_outNN.h"
#include "../source/compute.h"
#define SIZE_DATA 1
#define DP_IN 28*28
#define DP_OUT 10
#define LR 0.001
#define EPOCHS 1
#define SIZE_BATCH 1
#define TRAIN true
#define TEST false
#define SIZE_TEST 1
int main(){
    clock_t start_time, end_time,local_start,local_end, elapsed_time,local_elapsed;
    start_time = clock(); 

    local_start = clock(); 
    srand(time(0));
	double*** train_data=init_data_matrix(SIZE_DATA,DP_IN,DP_OUT);	
	if (train_data==NULL){
		ERROR("train_data is null!\n");
		return 1;
	}
	if (readMnistIMG(train_data,SIZE_DATA,TRAIN)){
		ERROR("read train img failed!\n");
		free_data_mtrx(train_data,SIZE_DATA);
		return 1;
	}
	if (readMnistLabels(train_data,SIZE_DATA,TRAIN)){
		ERROR("read train labels failed!\n");
		free_data_mtrx(train_data,SIZE_DATA);
		return 1;
	}
	
    mtrx* input=create_mtrx(SIZE_DATA,DP_IN);
	mtrx* expected=create_mtrx(SIZE_DATA,DP_OUT);
    splitData(SIZE_DATA,DP_IN,DP_OUT,train_data,input,expected);
	normalize(input,255);
    free_data_mtrx(train_data,SIZE_DATA);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to load 1 img %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);


    local_start = clock(); 
    nNetwork* NN=NULL;
    size_t len=3;
    size_t depths[]={DP_IN,256,DP_OUT};
    size_t functions[]={RELU,RELU,SOFT};
    NN = createNN( len, depths,functions);
    if (NN==NULL||NN->failFlag){
        ERROR("NN is NULL!\n");
        freeNN(NN);
        return 1;
    } 
    fillNN(NN);
    printf("Created NN correctly!\n");
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to create a NN %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock(); 
    printf ("Weight at %d;%d,%d=%f!\n",0,150,200,DATA(W(NN),get_index(W(NN),0,150,200)));
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to read 1 weight %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock(); 
    affect_values_mx_vxp(input,ACT(NN),0,0);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to affect inputs to ACT mtrx %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    mtrx* buff,*buff2;

    local_start = clock(); 
    buff=get_transpose_mtrx(W(NN),0);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to transpose the first W %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock(); 
    buff2=dot_m_v(buff,ACT(NN),0);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to dot W and first ACT %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    affect_values_m_vx(buff2,ZN(NN),1);

    local_start = clock();
    add_mtrx_mtrx_v_v(B(NN),ZN(NN),0,1);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to add B and first ACT %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock();
    activation(NN,1);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to act first ACT %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);
    
    local_start = clock();
    free_mtrx(buff);
    free_mtrx(buff2);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to free buffers %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock();
    predict(input,0, NN);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to predict 1 img %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock();
    printACT(NN);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to print act %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock();
    compute_grd(expected,NN,0,MULTICLASS);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to compute grd with 1 img %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);

    local_start = clock();
    updateNN(NN,LR);
    local_end = clock(); 
    local_elapsed=local_end-local_start;
    printf ("Time elapsed to update NN %f\n",((double)local_elapsed)/CLOCKS_PER_SEC);


    free_mtrx(input);
	free_mtrx(expected);
    freeNN(NN);


    end_time = clock(); 
    elapsed_time = end_time-start_time;
    printf ("Time elapsed recorded in the program %f!\n",((double)elapsed_time)/CLOCKS_PER_SEC);
}
