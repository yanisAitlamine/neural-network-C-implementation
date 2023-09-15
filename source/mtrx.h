#ifndef MTRX
#define MTRX
//#include <cuda_runtime.h>
#include "utils.h"
#include "errors.h"

typedef struct mtrx{
	size_t y;
	size_t z;
	double** data;
	double** cd_data;
}mtrx;
typedef struct mtrx_vector{
	size_t x;
	mtrx** data;
}mtrx_vector;

#define X(v) (v->x)
#define Y(m) (m->y)
#define Z(m) (m->z)
#define M(v,i) (v->data[i])
#define DATA(m,i,j) (m->data[i][j])
#define CD_DATA(m,i,j) (m->cd_data[i][j])



mtrx_vector* create_vector(size_t len, size_t* y, size_t* z);
mtrx* create_mtrx(size_t len, size_t depth);
void free_vector(mtrx_vector *v);
void free_mtrx(mtrx *v);
void write_list_in_mtrx(double* list, mtrx *m, int rank, size_t size_list);
//split data set if entries and expected are a simple list of activations
void splitData(int num_obj,size_t len_in, size_t len_out,double ***data, mtrx* input, mtrx* expected);
//print the whole vector
void print_vector(mtrx_vector *v);
//print the matrix
void print_mtrx(mtrx *m);
//print the list from mtrx at x
void print_list_m(mtrx *m,int x);
void init_vector(mtrx_vector *v);
void init_mtrx(mtrx *m);
void init_vector_rand(mtrx_vector *v);
void init_mtrx_rand(mtrx *m);
void normalize(mtrx* input, double max);
void add_to_v(mtrx_vector *v,double r);
void add_to_mtrx(mtrx *m,double r);
//add matrix at position x to the one at xp if they have the same dimmension
void add_v_to_v(mtrx_vector *v, mtrx_vector *vp);
//add matrix m to mp if they have the same dimmension
void add_mtrx_to_mtrx(mtrx *m, mtrx *mp);
void multiply_v(mtrx_vector *v,double r);
void multiply_mtrx(mtrx *m,double r);
void divide_v(mtrx_vector *v,double r);
void divide_mtrx(mtrx *m,double r);
void multiply_v_by_v(mtrx_vector *v, mtrx_vector *vp);
void multiply_mtrx_by_mtrx(mtrx *m, mtrx *mp);
void apply_on_mtrx(mtrx *m,double (*func)(double));
void apply_from_mtrx_into(mtrx *m,mtrx *mp,double (*func)(double));
double max_mtrx(mtrx *m);
double max_vector(mtrx_vector *v);
mtrx* get_transpose(mtrx *m); 
void affect_values_mtrx_to_mtrx(mtrx *m,mtrx *mp); 
void affect_values_v_to_v(mtrx_vector *v,mtrx_vector *vp);
//does dot operation between 2 matrixes m into mp
mtrx* dot(mtrx *m, mtrx *mp);
#endif
