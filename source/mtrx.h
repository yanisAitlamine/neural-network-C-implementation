/*
 * mtrx.h
 * Purpose: Implements mtrx-related functionality.
 * Auto-commented by GPT.
 */
#ifndef MTRX
#define MTRX
//#include <cuda_runtime.h>
#include "utils.h" // Include library for required functionality.
#include "errors.h" // Include library for required functionality.

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
void free_vector(mtrx_vector *v); // Function definition.
void free_mtrx(mtrx *v); // Function definition.
void write_list_in_mtrx(double* list, mtrx *m, int rank, size_t size_list); // Function definition.
//split data set if entries and expected are a simple list of activations
void splitData(int num_obj,size_t len_in, size_t len_out,double ***data, mtrx* input, mtrx* expected); // Function definition.
//print the whole vector
void print_vector(mtrx_vector *v); // Function definition.
//print the matrix
void print_mtrx(mtrx *m); // Function definition.
//print the list from mtrx at x
void print_list_m(mtrx *m,int x); // Function definition.
void init_vector(mtrx_vector *v); // Function definition.
void init_mtrx(mtrx *m); // Function definition.
void init_vector_rand(mtrx_vector *v); // Function definition.
void init_mtrx_rand(mtrx *m); // Function definition.
void normalize(mtrx* input, double max); // Function definition.
void add_to_v(mtrx_vector *v,double r); // Function definition.
void add_to_mtrx(mtrx *m,double r); // Function definition.
//add matrix at position x to the one at xp if they have the same dimmension
void add_v_to_v(mtrx_vector *v, mtrx_vector *vp); // Function definition.
//add matrix m to mp if they have the same dimmension
void add_mtrx_to_mtrx(mtrx *m, mtrx *mp); // Function definition.
void multiply_v(mtrx_vector *v,double r); // Function definition.
void multiply_mtrx(mtrx *m,double r); // Function definition.
void divide_v(mtrx_vector *v,double r); // Function definition.
void divide_mtrx(mtrx *m,double r); // Function definition.
void multiply_v_by_v(mtrx_vector *v, mtrx_vector *vp); // Function definition.
void multiply_mtrx_by_mtrx(mtrx *m, mtrx *mp); // Function definition.
void apply_on_mtrx(mtrx *m,double (*func)(double)); // Function definition.
void apply_from_mtrx_into(mtrx *m,mtrx *mp,double (*func)(double)); // Function definition.
double max_mtrx(mtrx *m);
double max_vector(mtrx_vector *v);
mtrx* get_transpose(mtrx *m); 
void affect_values_mtrx_to_mtrx(mtrx *m,mtrx *mp);  // Function definition.
void affect_values_v_to_v(mtrx_vector *v,mtrx_vector *vp); // Function definition.
//does dot operation between 2 matrixes m into mp
mtrx* dot(mtrx *m, mtrx *mp);
#endif