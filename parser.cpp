#include <iostream>
#include <string.h>
#include <vector>
#include "parser.h"

using namespace std;

Parser::Parser(int rows, int cols){
    this->rows = rows; 
    this->cols = cols;
    this->data.resize(rows);
    for(int i = 0; i < cols; i++)
        data[i].resize(cols);
}

Parser::Parser(const char* filename){

    FILE *fp = fopen(filename,"r");

    fseek(fp,0L,SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp,0L,SEEK_SET); 

    char *buffer = new char[file_size+1]; 

    fread(buffer, 1, file_size, fp);
    buffer[file_size] = '\0';
    fclose(fp);

    rows = getRowSize(buffer);
    cols = getColSize(buffer);

    data.resize(rows);
    
    for(int i = 0 ; i < rows ; i++)
        data[i].resize(cols);
    
    char *saveptr1, *saveptr2; 
    char *c1 = strtok_r(buffer,"\n",&saveptr1);
    int i = 0; 
    while(c1 != NULL){
        char *c2 = strtok_r(c1, ",",&saveptr2);
        int k = 0; 
        while(c2 != NULL){
          data[i][k] = atof(c2);  
          c2 = strtok_r(NULL,",",&saveptr2);
          k++;  
        }
        c1 = strtok_r(NULL,"\n",&saveptr1);
        i++;
    }
}

/* private functions */

//gets the number of rows in the csv file
int Parser::getRowSize(const char *str_array){
    
    int count = 0; 
    for(int i = 0 ; str_array[i] != '\0'; i++){
        if(str_array[i] == '\n' )
            count++;
    }

    return count; 
}

int Parser::getColSize(const char *str_array){
    
    int count = 1; 

    for(int i = 0 ; str_array[i] != '\0' && str_array[i] != '\n' ; i++){   
       if(str_array[i] == ',') 
        count++;
    }

    return count;
}

void Parser::print(void){
    for(int i = 0 ; i < rows ; i++){
            printf("%i [ ",i);
            for(int k = 0; k < cols ; k++){
               printf("%f ", data[i][k]);
            }
            printf("]\n");
        }
}
