#ifndef MTRX
#define MTRX
#include "utils.h"
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

typedef struct{
	size_t x;
	size_t y;
	double* data;
}mtrx;


//create vector
mtrx_vector* create_vector(size_t len, size_t* y, size_t* z);
//create vector
mtrx* create_mtrx(size_t len, size_t depth);
//free vector
void free_vector(mtrx_vector *v);
void free_mtrx(mtrx *v);
//return the total size
size_t total_size(mtrx_vector* v);
//get the absolute index of v[x][y][z]
int get_index(mtrx_vector *v,int x,int y,int z);
double *get_mtrx(mtrx_vector *v, int x);
double *get_list_from_m(mtrx *v, int x);
//write a list in the matrix vector regardless of the shape be careful
void write_list_in_vector(double* list,mtrx_vector *v, int x, size_t size_list);
void write_list(double* list,mtrx* v, int x,size_t size_list);
//split data set if entries and expected are a simple list of activations
void splitData(int num_obj,size_t len_in, size_t len_out,double ***data, mtrx* input, mtrx* expected);
//print the whole vector
void print_vector(mtrx_vector *v);
//print the matrix at position [x]
void print_mtrx_v(mtrx_vector *v,int x);
void print_mtrx_m(mtrx *v);
void print_list_m(mtrx *v,int x);
//init data to 0
void init_vector(mtrx_vector *v);
//init mtrx to rand
void init_mtrx(mtrx_vector *v,int x);
//init data to 0
void init_vector_rand(mtrx_vector *v);
//init mtrx to 0
void init_mtrx_rand(mtrx_vector *v,int x);
//normalize values to max
void normalize(mtrx* input, double max);
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
void multiply_mtrx_mtrx(mtrx_vector *v, mtrx_vector *vp,int x,int xp);
void divide_mtrx(mtrx_vector *v,int x, double r);

void exp_mtrx(mtrx_vector *v,int x);
void sigmoid_mtrx(mtrx_vector* v, int x);
void Relu_mtrx(mtrx_vector* v, int x);
void sigmoidP_mtrx(mtrx_vector* v, int x);
void ReluP_mtrx(mtrx_vector* v, int x);

//switch values xyz and xzy
void transpose_values(mtrx_vector *v,mtrx_vector *vp,int x); 

//affect values of one mtrx to another
void affect_values_vx_vxp(mtrx_vector *vp,mtrx_vector *v,int xp,int x); 
void affect_values_m_vx(mtrx *vp,mtrx_vector *v,int x);
void affect_values_mx_vxp(mtrx *v,mtrx_vector *vp,int x,int xp); 
//transposes mtrx
void transpose(mtrx_vector *v, int x);
mtrx_vector* get_transpose(mtrx_vector *v, int x);
//does dot operation between 2 matrixes
mtrx_vector* dot(mtrx_vector *v, mtrx_vector *vp,int x,int xp);
#endif
