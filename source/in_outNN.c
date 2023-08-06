#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "in_outNN.h"

// read from a stream into a chain for a set number of characters
int read(FILE *stream, char *chain, size_t len) {

    char *returnChar = NULL;

    if (fgets(chain, len, stream)==NULL){
        return 1;
    }
    returnChar = strchr(chain, '\n');
    if (returnChar!=NULL){
            *returnChar='\0';
    }
    freeBuffer();
    return 0;
}

// read characters in the buffer until the end of the fil or \n
void freeBuffer() {
    int a=0;
    while (a!='\n' && a!=EOF) {
        a=getchar();
    }
}

void writeNN(FILE* toSave, nNetwork NN){
    printf ("Saving neural net of size %ld!\n",NN.len+1);
    fprintf(toSave, "%ld\n",NN.len);
    for (int i=0;i<NN.len;i++){
	fprintf(toSave, "%ld",(NN.weights[i]).len);
	if (i<NN.len-1){fputc(',', toSave);} else {fputc('\n', toSave);}
    }
    for (int i=0;i<NN.len;i++){
    	writeMtrx(toSave, &(NN.weights[i]));
    	fputc('\n', toSave);
    	writeMtrx(toSave, &(NN.bias[i]));
	fputc('\n', toSave);
    }
}

void writeMtrx(FILE* toSave, matrix* mtrx){
    fputc('[', toSave);
	for (int i=0;i<mtrx->len;i++){
		fputc('[',toSave);
		for (int y=0;y<mtrx->depth;y++){
			fprintf(toSave, "%f",(mtrx->data)[i][y]);
			if (y<mtrx->depth-1){fputc(',', toSave);}
		}
		fputc(']', toSave);
	}
	fputc(']', toSave);

}
