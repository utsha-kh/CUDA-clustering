#include <iostream>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include "kmeans_gpu.h"
#include "parser.h"

int main(int argc, char** argv){

    // read data from a csv file
    std::cout << "Loading input file..." << std::endl;
    const char *filename = "input.csv";
    Parser parser(filename);
    std::cout << "--Finished loading!" << std::endl;

    int n = parser.rows;        // number of data points
    int d = parser.cols;        // dimention of input data (usually 2, for 2D data)
    int k = 3;                  // number of clusters

    // an option to input k as an arugument
    if(argc > 1){
        k = std::atoi(argv[1]);
    }

    float** data = parser.rdata;
    int* labels = new int[n];   // array to store the labels

    // measure execution time
    auto t1 = std::chrono::high_resolution_clock::now();

    // actual work done here
    kMeansClustering(data, labels, n, d, k);
    
    // display the measured execution time
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Execution Time: " << duration / 1000 << " [ms]" << std::endl;

    // write result to a new csv file
    std::cout << "Writing the result to \"result.csv\"... " << std::endl;
    parser.toCSV("result_cuda.csv", data, labels, n, d);
    std::cout << "--Finished writing!" << std::endl;
    
    return 0;

}
