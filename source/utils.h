#ifndef UT
#define UT
#include <math.h>

//return random double decimal
double rand_decimal();

double rand_decimal(){
    return (double)rand()/(double)RAND_MAX;
}

double sigmoid(double n){
    return (1/(1+pow(EULER_NUMBER, -n)));
}

double sigmoidprime(double n){
    return (n)*(1-(n));
}

double Relu(double n){
    if(n<0)return 0;
    return n;
}

double Reluprime(double n){
    if (n)return 1;
    return 0;
}


#endif
