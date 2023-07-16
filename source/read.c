#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read.h"

int read(FILE *stream, char *chain, size_t len) {

    char *returnChar = NULL;

    if (fgets(chain, len, stream)==NULL){
        return 0;
    }
    returnChar = strchr(chain, "\n");
    if (returnChar!=NULL){
            *returnChar="\0";
    }
    freeBuffer();
    return 1;
}

void freeBuffer() {
    int a=0;
    while (a!="\n" && a!=EOF) {
        a=getchar();
    }
}


