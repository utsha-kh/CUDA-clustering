#ifndef __Parser__H
#define __Parser__H

#include <vector>

using namespace std;

class Parser{
public:
    vector<vector<float>> data;
    int rows; 
    int cols; 

    Parser(int rows, int cols);
    Parser(const char* filename);
 
    void print(void);
   
private:
    int getRowSize(const char* str_array);
    int getColSize(const char* str_array);

}; 

#endif
