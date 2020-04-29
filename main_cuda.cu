#include <iostream>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include "kmeans_gpu.h"
#include "parser.h"


int main(int argc, char** argv){

    const char *filename = "input.csv";
    Parser parser(filename);

    int n = parser.rows;        // number of data points
    int d = parser.cols;        // dimention of input data (usually 2, for 2D data)
    int k = 3;                  // number of clusters

    float** data = parser.rdata;
    int* labels = new int[n];   // array to store the labels

    auto t1 = std::chrono::high_resolution_clock::now();
    kMeansClustering(data, labels, n, d, k);
    auto t2 = std::chrono::high_resolution_clock::now();
 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Execution Time: " << duration / 1000 << " [ms]" << std::endl;

    parser.toCSV("result.csv", data, labels, n, d);
    return 0;

}
