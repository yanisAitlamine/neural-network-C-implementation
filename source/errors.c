#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include "errors.h"

// Return ERROR value if table is null
bool check_malloc(void* data,char* message) {
    if(data==NULL){
            return ERROR(message);
    }
    return false;
}


