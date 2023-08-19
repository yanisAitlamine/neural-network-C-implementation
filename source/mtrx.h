#ifndef MTRX
#define MTRX
#include "errors.h"
#define X(vector) (vector->x)
#define Y(vector,i) (vector->y[i])
#define Z(vector,i) (vector->z[i])
#define DATA(vector,i) (vector->data[i])

typedef struct{
	size_t x;
	size_t *y;
	size_t *z;
	double* data;
}mtrx_vector;

//create vector
mtrx_vector* create_vector(size_t len, size_t* y, size_t* z);
//free vector
void free_vector(mtrx_vector *v);
//return the total size
size_t total_size(mtrx_vector* v){
//get the absolute index of v[x][y][z]
int get_index(mtrx_vector *v,int x,int y,int z);
//print the whole vector
void print_vector(mtrx_vector *v);
//print the matrix at position [x]
void print_mtrx(mtrx_vector *v,int x);
//init data to 0
void init_vector(mtrx_vector *v);
//init mtrx to 0
void init_mtrx(mtrx_vector *v,int x);
//init data to 0
void init_vector_rand(mtrx_vector *v);
//init mtrx to 0
void init_mtrx_rand(mtrx_vector *v,int x);
//add i to matrix 
void add_mtrx(mtrx_vector *v,int x,double r);
//add matrix at position x to the one at xp of two vectors if they have the same dimmension
void add_mtrx_mtrx(mtrx_vector *v, mtrx_vector *vp,int x,int xp);
//multiply the whole vector by r
void multiply_vector(mtrx_vector *v, double r);
void divide_vector(mtrx_vector *v, double r);
//return max of a matrix
double max_mtrx(mtrx_vector *v, int x);
//return max of a vector
double max_vector(mtrx_vector *v);
//multiply mtrx by i
void multiply_mtrx(mtrx_vector *v,int x, double r);
void divide_mtrx(mtrx_vector *v,int x, double r);

void exp_mtrx(mtrx_vector *v,int x);
void sigmoid_mtrx(mtrx_vector* v, int x);
void Relu_mtrx(mtrx_vector* v, int x);
void sigmoidP_mtrx(mtrx_vector* v, int x);
void ReluP_mtrx(mtrx_vector* v, int x);

//switch values xyz and xzy
void transpose_values(mtrx_vector *v,mtrx_vector *vp,int x); 

//affect values of one mtrx to another
void affect_values(mtrx_vector *vp,mtrx_vector *v,int xp,int x); 
//transposes mtrx
void transpose(mtrx_vector *v, int x);
mtrx_vector* get_transpose(mtrx_vector *v, int x);
//does dot operation between 2 matrixes
mtrx_vector* dot(mtrx_vector *v, mtrx_vector *vp,int x,int xp);
#endif
