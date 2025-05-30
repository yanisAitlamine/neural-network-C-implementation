#include "utils.h"

//compute x to the power of n
double pow_double_int(double x, int n){
    double result = 1.0;
    if (n>0){
        for (int i=0;i<n;i++) result*=x;
        return result;
    }
    if (n<0){
        for (int i=0;i<n;i++) result/=x;
        return result;
    }
    return result;
}

//produce a random decimal
double rand_decimal(){
    return (double)rand()/(double)RAND_MAX;
}

//return sigmoid of n
double sigmoid(double n){
    return (1/(1+pow(EULER_NUMBER, -n)));
}

//sigmoid'(n) need to check this again
double sigmoidprime(double n){
    return (n)*(1-(n));
}

//Relu of n
double Relu(double n){
    if(n<0)return 0;
    return n;
}


//Relu prime of n
double Reluprime(double n){
    if (n>=0){
        return 1;
    }
    return 0;
}

// compute cost
double regression_cost(double expected, double output){
    return expected-output;
}

//cost to the power of 2
double sqr_regression(double expected, double output){
    return pow_double_int(expected-output,2);
}

// case expected = 0 output=1 handled, would divide by 0
double binary_prime(double expected, double output){
    if (expected == output){ return 0;}
    if (output!=1&&output!=0){return (expected/output)-((1-expected)/(1-output));}
    return (expected/output+0.000001)-((1-expected)/(1-output+0.000001));
}

// derivative of sqr_regreqssion
double sqr_prime(double expected, double output){
    return -2*(expected-output);
}

// compute binary cost
double binary_cost(double expected, double output){
    return -((expected*log(output))+((1-expected)*log(1-output)));
}

//compute the cost with an int selector ( selector could be 2 bits)
double cost (double expected, double output, int function){
    switch(function){
        case REGRESSION:
            return regression_cost(expected,output);
        break;
        case SQR_REG:
            return sqr_regression(expected,output);
            break;
        case BINARY:
            return binary_cost(expected,output);
        break;
        case MULTICLASS:
            if(output==0)output=0.0000001;
            return -expected*log(output);
        break;
    }
    return ERR_RETURN;
}

//sums numbers of an array
double sum_double(double* data, int size_data){
    double result=0;
    for (int i=0;i<size_data;i++)result+=data[i];
    return result;
}

//average an array
double mean_double(double* data,int size_data){
    return sum_double(data,size_data)/size_data;
}


void swapTables(double ***data,int base,int target,int depth_in,int depth_out){
    double buffer;
    for (int i=0;i<depth_in;i++){
        buffer=data[base][0][i];
        data[base][0][i]=data[target][0][i];
        data[target][0][i]=buffer;
    }
    for (int i=0;i<depth_out;i++){
        buffer=data[base][1][i];
        data[base][1][i]=data[target][1][i];
        data[target][1][i]=buffer;
    }
}


void shuffle(double*** data,int len,int depth_in,int depth_out,int rounds){
    for (int a=0;a<rounds;a++){
        for (int i=0;i<len;i++){
            swapTables(data,i,(len-1)*(rand_decimal()),depth_in,depth_out);
        }
    }
}

