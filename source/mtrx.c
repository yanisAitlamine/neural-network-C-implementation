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
    mtrx_vector *v=malloc(sizeof(mtrx_vector));
    X(v)=len;
    v->y=malloc(len*sizeof(size_t));
    if (check_malloc(v->y,"Mtrx malloc failed!\n")) {
        free(v);
        return NULL;
    }
    v->z=malloc(len*sizeof(size_t));
    if (check_malloc(v->z,"Mtrx malloc failed!\n")) {
        free_vector(v);
        return NULL;
    }
    size_t size=0;
    for (int i=0;i<len;i++){
        if (y[i]!=0&&z[i]!=0){
            Y(v,i)=y[i];
            Z(v,i)=z[i];
            size+=y[i]*z[i];
        }else{
            ERROR("sizes incorrect");
            free(v->y);
            free(v->z);
            free(v);
            return NULL;
        }
    }
#if DEBUGINIT
            printf("size %ld\n",size);
#endif
    v->data=malloc(size*sizeof(double));
    if (check_malloc(v->data,"Mtrx malloc failed!\n")){
        free_vector(v);
        return NULL;
    }
    return v;
}

mtrx* create_mtrx(size_t len, size_t depth){
#if DEBUGINIT
    printf("\nCreating mtrx!\n");
    printf ("v %ld*%ld=%ld\n",len,depth,len*depth);
    fflush(stdout);
#endif
    mtrx *v=malloc(sizeof(mtrx));
    (*v).x=len;
    (*v).y=depth;
    v->data=malloc(X(v)*(v->y)*sizeof(double));
    if (check_malloc(v->data,"Mtrx malloc failed!\n")){
        free_mtrx(v);
        return NULL;
    }
    return v;
}
//free vector
void free_vector(mtrx_vector *v){
    if(v==NULL)return;
    if (v->data!=NULL)free(v->data);
    if(v->y!=NULL)free(v->y);
    if(v->z!=NULL)free(v->z);
    free(v);
}

//free mtrx
void free_mtrx(mtrx *v){
    if(v==NULL)return;
    if (v->data!=NULL)free(v->data);
    free(v);
}

size_t total_size(mtrx_vector* v){
    size_t size=0;
    for (int i=0;i<X(v);i++){
        size+=Y(v,i)*Z(v,i);
    }
    return size;
}
//get the absolute index of v[x][y][z]
int get_index(mtrx_vector *v,int x,int y,int z){
    int i,j,index=0;
    for (i=0;i<x;i++){
        index+=Y(v,i)*Z(v,i);
    }
    for (j=0;j<y;j++){
        index+=Z(v,i);
    }
    return index+z;
}

double *get_mtrx(mtrx_vector *v, int x){
    double* result=malloc(Y(v,x)*Z(v,x)*sizeof(double));
    for (int i=0;i<Y(v,x);i++){
        for (int j=0;j<Z(v,x);j++){
           result[i*Z(v,x)+j]=DATA(v,get_index(v,x,i,j)); 
        }
    }
    return result;
}

double *get_list_from_m(mtrx *v, int x){
    double* result=malloc(v->y*sizeof(double));
    for (int i=0;i<v->y;i++){
        result[i]=DATA(v,x*v->y+i); 
    }
    return result;
}

void write_list_in_vector(double* list,mtrx_vector *v, int x,size_t size_list){
    int index=get_index(v,x,0,0);
    for (int i=0;i<size_list;i++){
        DATA(v,index+i)=list[i];
    }
}

void write_list(double* list,mtrx *v, int x,size_t size_list){
    for (int i=0;i<size_list;i++){
        DATA(v,(x*v->y)+i)=list[i];
    }
}

//split data set if entries and expected are a simple list of activations
void splitData(int num_obj,size_t len_in, size_t len_out,double ***data, mtrx* input, mtrx* expected){
    for (int i=0;i<num_obj;i++){
        write_list(data[i][0],input,i,len_in);
        write_list(data[i][1],expected,i,len_out);
    }
}

//print the whole vector
void print_vector(mtrx_vector *v){
    for (int i=0;i<X(v);i++){
        printf("\n");
        print_mtrx_v(v,i);
    }
    printf("\n");
}
//print the matrix at position [x]
void print_mtrx_v(mtrx_vector *v,int x){
    for (int i=0;i<Z(v,x);i++){
        for (int j=0;j<Y(v,x);j++){
            printf("%f ",DATA(v,get_index(v,x,j,i)));
         }
         printf("\n");
    }
}
//print the matrix 
void print_mtrx_m(mtrx *v){
    for (int i=0;i<X(v);i++){
        for (int j=0;j<v->y;j++){
            printf("%f ",DATA(v,i*v->y+j));
         }
         printf("\n");
    }
}
//print the list at x
void print_list_m(mtrx *v,int x){
    printf("\n");
    for (int j=0;j<v->y;j++){
        printf("mtrx[%ld]:%f ",x*v->y+j,DATA(v,x*v->y+j));
    }
    printf("\n");
}
void init_vector(mtrx_vector *v){
    for (int i=0;i<X(v);i++){
        init_mtrx(v,i);
    }
}

void init_mtrx(mtrx_vector *v,int x){
    for (int i=0;i<Y(v,x);i++){
        for (int j=0;j<Z(v,x);j++){
            DATA(v,get_index(v,x,i,j))=0;
         }
    }
}

void init_vector_rand(mtrx_vector *v){
    for (int i=0;i<X(v);i++){
        init_mtrx_rand(v,i);
    }
}

void init_mtrx_rand(mtrx_vector *v,int x){
    for (int i=0;i<Y(v,x);i++){
        for (int j=0;j<Z(v,x);j++){
            DATA(v,get_index(v,x,i,j))=rand_decimal();
         }
    }
}

void normalize(mtrx* input, double max){
    int i,j;
    for (i=0;i<X(input);i++){
        for (j=0;j<input->y;j++){
         DATA(input,i*input->y+j)/=max;
        }
    }
}



void add_mtrx(mtrx_vector *v,int x,double r){
    for (int i=0;i<Y(v,x);i++){
        for (int j=0;j<Z(v,x);j++){
            DATA(v,get_index(v,x,i,j))+=r;
         }
    }
}
//add matrix at position x to the one at xp if they have the same dimmension
void add_mtrx_mtrx(mtrx_vector *v, mtrx_vector *vp,int x,int xp){
    if (Y(v,x)!=Y(vp,xp)||Z(v,x)!=Z(vp,xp)){
        ERROR("Matrix sizes incompatible for addition!\n");
        return;
    }
    for (int i=0;i<Y(v,x);i++){
        for (int j=0;j<Z(v,x);j++){
            DATA(vp,get_index(vp,xp,i,j))+=DATA(v,get_index(v,x,i,j));
         }
    }
}

void multiply_vector(mtrx_vector *v, double r){
    for (int x=0;x<X(v);x++){
        multiply_mtrx(v,x,r);
    }
}
//multiply mtrx by i
void multiply_mtrx(mtrx_vector *v,int x, double r){
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            DATA(v,get_index(v,x,y,z))*=r;
         }
    }
}

void multiply_mtrx_mtrx(mtrx_vector *v, mtrx_vector *vp,int x,int xp){
    if (Y(v,x)!=Y(vp,xp)||Z(v,x)!=Z(vp,xp)){
        ERROR("Matrix sizes incompatible for addition!\n");
        return;
    }
    for (int i=0;i<Y(v,x);i++){
        for (int j=0;j<Z(v,x);j++){
            DATA(vp,get_index(vp,xp,i,j))*=DATA(v,get_index(v,x,i,j));
         }
    }
}

void divide_vector(mtrx_vector *v, double r){
    for (int x=0;x<X(v);x++){
        divide_mtrx(v,x,r);
    }
}
//multiply mtrx by i
void divide_mtrx(mtrx_vector *v,int x, double r){
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            DATA(v,get_index(v,x,y,z))/=r;
         }
    }
}

void exp_mtrx(mtrx_vector *v,int x){
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            DATA(v,get_index(v,x,y,z))=exp(DATA(v,get_index(v,x,y,z)));
         }
    }
}

void sigmoid_mtrx(mtrx_vector *v,int x){
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            DATA(v,get_index(v,x,y,z))=sigmoid(DATA(v,get_index(v,x,y,z)));
         }
    }
}

void Relu_mtrx(mtrx_vector *v,int x){
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            DATA(v,get_index(v,x,y,z))=Relu(DATA(v,get_index(v,x,y,z)));
         }
    }
}

void sigmoidP_mtrx(mtrx_vector *v,int x){
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            DATA(v,get_index(v,x,y,z))=sigmoidprime(DATA(v,get_index(v,x,y,z)));
         }
    }
}

void ReluP_mtrx(mtrx_vector *v,int x){
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            DATA(v,get_index(v,x,y,z))=Reluprime(DATA(v,get_index(v,x,y,z)));
         }
    }
}


double max_mtrx(mtrx_vector *v, int x){
    double max_val = DATA(v,get_index(v,x,0,0));
    for (int i = 1; i < Y(v,x); i++) {
        for (int j=0;j<Z(v,x);j++){
            if (DATA(v,get_index(v,x,i,j)) > max_val) {
                max_val = DATA(v,get_index(v,x,i,j));
            }
        }
    }
    return max_val;
}

double max_vector(mtrx_vector *v){
    double max_val = DATA(v,0);
    for (int i = 1; i < total_size(v); i++) {
        if (DATA(v,i) > max_val) {
            max_val = DATA(v,i);
        }
    }
    return max_val;
}

void transpose_values(mtrx_vector *v,mtrx_vector *vp,int x){ 
    int i,j;
    for (i=0;i<Y(v,x);i++){
        for (j=0;j<Z(v,x);j++){
        DATA(vp,get_index(vp,0,j,i))=DATA(v,get_index(v,x,i,j));
        }
    }
}

void affect_values_vx_vxp(mtrx_vector *vp,mtrx_vector *v,int xp,int x){ 
    int i,j;
    for (i=0;i<Y(v,x);i++){
        for (j=0;j<Z(v,x);j++){
        DATA(v,get_index(v,x,i,j))=DATA(vp,get_index(vp,xp,i,j));
        }
    }
}

void affect_values_m_vx(mtrx *vp,mtrx_vector *v,int x){ 
    int i,j;
    for (i=0;i<Y(v,x);i++){
        for (j=0;j<Z(v,x);j++){
        DATA(v,get_index(v,x,i,j))=DATA(vp,i*vp->y+j);
        }
    }
}

void affect_values_mx_vxp(mtrx *v,mtrx_vector *vp,int x,int xp){ 
    int j;
    for (j=0;j<v->y;j++){
        DATA(vp,get_index(vp,xp,0,0)+j)=DATA(v,x*v->y+j);
    }
}

//transposes mtrx
void transpose(mtrx_vector *v, int x){
    size_t new_len[]={Z(v,x)};
    size_t new_dpth[]={Y(v,x)};
    mtrx_vector *vp=create_vector(1,new_len,new_dpth);
    transpose_values(v,vp,x);
    size_t buffer=Y(v,x);
    Y(v,x)=Z(v,x);
    Z(v,x)=buffer;
    affect_values_vx_vxp(vp,v,0,x);
    free_vector(vp);
}
//get transposes mtrx
mtrx_vector* get_transpose(mtrx_vector *v, int x){
    size_t new_len[]={Z(v,x)};
    size_t new_dpth[]={Y(v,x)};
    mtrx_vector *vp=create_vector(1,new_len,new_dpth);
    transpose_values(v,vp,x);
    return vp;
}

#define DEBUGDOT !true
//does dot operation between 2 matrixes
mtrx_vector* dot(mtrx_vector *v, mtrx_vector *vp,int x,int xp){
    double current_result=0;
    size_t new_len[]={Y(v,x)};
    size_t new_dpth[]={Z(vp,xp)}; 
    mtrx_vector *vr=create_vector(1,new_len,new_dpth);
    init_vector(vr);
    for (int y=0;y<Y(v,x);y++){
        for (int z=0;z<Z(v,x);z++){
            for (int i=0;i<Z(vp,xp);i++){
                DATA(vr,get_index(vr,0,y,i))+=DATA(v,get_index(v,x,y,z))*DATA(vp,get_index(vp,xp,z,i));
#if DEBUGDOT
                printf("\ndot compute %d,%d+=%d,%d*%d,%d\n%f+=%f*%f\n",y,i,y,z,z,i,DATA(vr,get_index(vr,0,y,i)),DATA(v,get_index(v,x,y,z)),DATA(vp,get_index(vp,xp,z,i)));
#endif 
             }
        }
    }
    return vr;
}

