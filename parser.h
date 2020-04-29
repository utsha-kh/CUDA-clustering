#ifndef __PARSER__H
#define __PARSER__H

#include <vector>

class Parser{
public:
    std::vector<std::vector<float> > data;
    float** rdata;
    int rows; 
    int cols; 

    Parser(const char* filename);
 
    void print(void);
    void toCSV(const char *filename, std::vector<std::vector<float> > data, std::vector<int> labels);
    void toCSV(const char *filename, float** data, int* labels, int n, int d);
    void copyData(float** output);

private:
    int getRowSize(const char* str_array);
    int getColSize(const char* str_array);
}; 

#endif
