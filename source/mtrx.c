#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "mtrx.h"

#define DEBUGINIT !true
//create vector
mtrx_vector* create_vector(size_t len, size_t* y, size_t* z){
#if DEBUGINIT
    printf("\nCreating vector!\n");
    fflush(stdout);
#endif
    mtrx_vector *v=calloc(1,sizeof(mtrx_vector));
    X(v)=len;
    v->data=calloc(len,sizeof(mtrx*));
    if (check_malloc(v->data,"Mtrx list in vector malloc failed!\n")){
        free_vector(v);
        return NULL;
    }
    for (int i=0;i<X(v);i++){
        v->data[i]=create_mtrx(y[i],z[i]);
        if (check_malloc(v->data,"Mtrx in vector malloc failed!\n")){
            free_vector(v);
            return NULL;
        }
    }
    return v;
}

mtrx* create_mtrx(size_t len, size_t depth){
#if DEBUGINIT
    printf("\nCreating mtrx!\n");
    printf ("v %ld*%ld=%ld\n",len,depth,len*depth);
    fflush(stdout);
#endif
    mtrx* new_mtrx = (mtrx*)malloc(sizeof(mtrx));
    if (!new_mtrx) return NULL;

    new_mtrx->y = len;
    new_mtrx->z = depth;

    new_mtrx->data = (double**)calloc(len , sizeof(double*));
    if (!new_mtrx->data) {
        free(new_mtrx);
        return NULL;
    }

    for (size_t i = 0; i < len; i++) {
        new_mtrx->data[i] = (double*)calloc(depth , sizeof(double));
        if (!new_mtrx->data[i]) {
            // cleanup on failure
            for (size_t j = 0; j < i; j++) {
                free(new_mtrx->data[j]);
                free(new_mtrx->cd_data[j]);
            }
            free(new_mtrx->data);
            free(new_mtrx->cd_data);
            free(new_mtrx);
            return NULL;
        }
    }
    return new_mtrx;
}
//free vector
void free_vector(mtrx_vector *v){
    if(v==NULL)return;
    if (v->data!=NULL){
        for (int i=0;i<X(v);i++)free_mtrx(v->data[i]);
        free(v->data);
    }
    free(v);
}

//free mtrx
void free_mtrx(mtrx *v){
    int i;
    if(v==NULL)return;
    if (v->data!=NULL){
        for (i=0;i<Y(v);i++)free(v->data[i]);
        free(v->data);
    }
    /*if(v->cd_data!=NULL){
        for (i=0;i<Y(v);i++)cudaFree(v->cd_data[i]);
        cudaFree(v->cd_data);
    }*/
    free(v);
}

void write_list_in_mtrx(double* list, mtrx *m, int rank, size_t size_list){
    if (size_list!=Z(m)){
        printf("Can't write list to mtrx %ld!=%ld!\n",Z(m),size_list);
        return;
    }
    for (int i=0;i<size_list;i++){
        DATA(m,rank,i)=list[i];
    }
}

//split data set if entries and expected are a simple list of activations
void splitData(int num_obj,size_t len_in, size_t len_out,double ***data, mtrx* input, mtrx* expected){
    if (num_obj!=Y(input)||num_obj!=Y(expected)){
        printf ("Sizes not correct num_obj=%d, y input=%ld, y expected=%ld",num_obj,Y(input),Y(expected));
        return;
    }
    for (int i=0;i<num_obj;i++){
        write_list_in_mtrx(data[i][0],input,i,len_in);
        write_list_in_mtrx(data[i][1],expected,i,len_out);
    }
}

//print the whole vector
void print_vector(mtrx_vector *v){
    for (int i=0;i<X(v);i++){
        printf("\n");
        print_mtrx(M(v,i));
    }
    printf("\n");
}
//print the matrix
void print_mtrx(mtrx *m){
    for (int i=0;i<Z(m);i++){
        for (int j=0;j<Y(m);j++){
            printf("%f ",DATA(m,j,i));
         }
         printf("\n");
    }

}

//print the list at x
void print_list_m(mtrx *m,int x){
    printf("\n");
    for (int j=0;j<Z(m);j++){
        printf("mtrx[%ld]:%f ",x*Z(m)+j,DATA(m,x,j));
    }
    printf("\n");
}

void init_vector(mtrx_vector *v){
    for (int i=0;i<X(v);i++){
        init_mtrx(M(v,i));
    }
}

void init_mtrx(mtrx *m){
    for (int i=0;i<Y(m);i++){
        for (int j=0;j<Z(m);j++){
            DATA(m,i,j)=0;
         }
    }
}

void init_vector_rand(mtrx_vector *v){
    for (int i=0;i<X(v);i++){
        init_mtrx_rand(M(v,i));
    }
}

void init_mtrx_rand(mtrx *m){
    for (int i=0;i<Y(m);i++){
        for (int j=0;j<Z(m);j++){
            DATA(m,i,j)=rand_decimal();
         }
    }
}

void normalize(mtrx* input, double max){
    int i,j;
    for (i=0;i<Y(input);i++){
        for (j=0;j<Z(input);j++){
         DATA(input,i,j)/=max;
        }
    }
}

void add_to_v(mtrx_vector *v,double r){
    for (int i=0;i<X(v);i++)add_to_mtrx(M(v,i),r);
}

void add_to_mtrx(mtrx *m,double r){
    for (int i=0;i<Y(m);i++){
        for (int j=0;j<Z(m);j++){
            m->data[i][j]+=r;
         }
    }
}

//add matrix at position x to the one at xp if they have the same dimmension
void add_v_to_v(mtrx_vector *v, mtrx_vector *vp){
    if (X(v)!=X(vp)){
        printf ("Adding two vectors with different len!\n");
        return;
    }
    for (int i=0;i<X(v);i++)add_mtrx_to_mtrx(M(v,i),M(vp,i));
}

void add_mtrx_to_mtrx(mtrx *m, mtrx *mp){
    if (Y(m)!=Y(mp)||Z(m)!=Z(mp)){
        ERROR("Matrix sizes incompatible for addition!\n");
        return;
    }
    for (int i=0;i<Y(m);i++){
        for (int j=0;Z(m);j++){
            mp->data[i][j]+=m->data[i][j];
         }
    }
}

void multiply_v(mtrx_vector *v,double r){
    for (int i=0;i<X(v);i++)multiply_mtrx(M(v,i),r);
}

void multiply_mtrx(mtrx *m,double r){
    for (int i=0;i<Y(m);i++){
        for (int j=0;j<Z(m);j++){
            m->data[i][j]*=r;
         }
    }
}

void divide_v(mtrx_vector *v,double r){
    for (int i=0;i<X(v);i++)multiply_mtrx(M(v,i),r);
}

void divide_mtrx(mtrx *m,double r){
    for (int i=0;i<Y(m);i++){
        for (int j=0;j<Z(m);j++){
            m->data[i][j]*=r;
         }
    }
}

void multiply_mtrx_by_mtrx(mtrx *m, mtrx *mp){
    if (Y(m)!=Y(mp)||Z(m)!=Z(mp)){
        ERROR("Matrix sizes incompatible for multiplication!\n");
        return;
    }
    for (int i=0;i<Y(m);i++){
        for (int j=0;Z(m);j++){
            mp->data[i][j]*=m->data[i][j];
         }
    }
}

void apply_on_mtrx(mtrx *m,double (*func)(double)){
    for (int i=0;i<Y(m);i++){
        for (int j=0;Z(m);j++){
            m->data[i][j]=(*func)(m->data[i][j]);
         }
    }
}

void apply_from_mtrx_into(mtrx *m,mtrx *mp,double (*func)(double)){
    if (Y(m)!=Y(mp)||Z(m)!=Z(mp)){
        ERROR("Matrix sizes incompatible for assignment!\n");
        return;
    }
    for (int i=0;i<Y(m);i++){
        for (int j=0;Z(m);j++){
            mp->data[i][j]=(*func)(m->data[i][j]);
         }
    }
}

double max_mtrx(mtrx *m){
    double max_val = 0;
    for (int i=0; i<Y(m); i++) {
        for (int j=0;j<Z(m);j++){
            if (m->data[i][j] > max_val) {
                max_val = m->data[i][j];
            }
        }
    }
    return max_val;
}

double max_vector(mtrx_vector *v){
    double max_val = 0,current_val=0;
    for (int i = 0; i < X(v); i++) {
        current_val=max_mtrx(M(v,i));
        if ( current_val > max_val) {
            max_val = current_val;
        }
    }
    return max_val;
}

mtrx* get_transpose(mtrx *m){ 
    int i,j;
    mtrx* result= create_mtrx(Z(m),Y(m));
    if (check_malloc(result,"Buffer malloc failed in transpose!\n")){
        return NULL;
    }
    for (i=0;i<Y(m);i++){
        for (j=0;j<Z(m);j++){
            result->data[j][i]=m->data[i][j];
        }
    }
    return result;
}

void affect_values_mtrx_to_mtrx(mtrx *m,mtrx *mp){ 
    if (Y(m)!=Y(mp)||Z(m)!=Z(mp)){
        ERROR("Matrix sizes incompatible for multiplication!\n");
        return;
    }
    for (int i=0;i<Y(m);i++){
        for (int j=0;Z(m);j++){
            mp->data[i][j]=m->data[i][j];
         }
    }
}

void affect_values_v_to_v(mtrx_vector *v,mtrx_vector *vp){
    if (X(v)!=X(vp)){
        printf ("Affecting two vectors with different len!\n");
        return;
    }
    for (int i=0;i<X(v);i++)affect_values_mtrx_to_mtrx(M(v,i),M(vp,i));
}

#define DEBUGDOT true
//does dot operation between 2 matrixes m into mp
mtrx* dot(mtrx *m, mtrx *mp){
    if (Y(m)!=Z(mp)||Y(mp)!=Z(m)){
        printf ("Invalid dimensions for dot!\n");
        return NULL;
    }
    mtrx *result=create_mtrx(Y(m),Z(mp));
    if (check_malloc(result,"Buffer malloc failed in transpose!\n")){
        return NULL;
    }
    for (int i=0;i<Y(m);i++){
        for (int j=0;j<Y(mp);j++){
            for (int k;k<Z(mp);k++){
                result->data[i][k]+=m->data[i][j]*mp->data[j][k];
#if DEBUGDOT
    printf ("Dotting result[%d][%d]=%f!\n",i,k,result->data[i][k]);
#endif
            }
        }
    }
    return result;
}
