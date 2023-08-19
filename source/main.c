#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mtrx.h"
#include "neuralnet.h"
#include "errors.h"
#include "in_outNN.h"
#include "compute.h"
#define SIZE_DATA 60000 
#define DP_IN 784
#define DP_OUT 10
#define LR 0.01
#define EPOCHS 10
#define BATCH_SIZE 100
#define TRAIN true
#define TEST false
#define SIZE_TEST 10

int main()
{
	srand(time(0));
	size_t len[]={2,2,4};
	size_t dpth[]={4,4,2};
	mtrx_vector *v=create_vector(3,len,dpth);
	printf("Created!\n");
	fflush(stdout);
	init_vector(v);
	DATA(v,get_index(v,1,0,0))=4;
	print_vector(v);
	add_mtrx(v,1,2);
	add_mtrx(v,2,2);
	print_vector(v);
	add_mtrx_mtrx(v,v,1,1);
	print_vector(v);
	transpose(v,1);
	print_vector(v);
	transpose(v,1);
	mtrx_vector *vd=dot(v,v,1,2);
	print_vector(vd);
	free_vector(vd);
	free_vector(v);
	return 0;
}
