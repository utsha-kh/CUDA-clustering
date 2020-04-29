#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <chrono>
#include "kmeans.h"
#include "parser.h"

int main (int argc, char** argv) {

    // read data from an input file
    const char *filename = "input.csv";
    Parser parser(filename);

    int k = 3;
    // an option to input k as an arugument
    if(argc > 1){
        k = std::atoi(argv[1]);
    }

    // initialize the KMeans module with parsed data
    KMeans module(parser.rows, parser.cols, k, parser.data);

    // measure execution time
    auto t1 = std::chrono::high_resolution_clock::now();

    // actual work here
    module.kMeansClustering();

    // display the measured execution time
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Execution Time: " << duration / 1000 << " [ms]" << std::endl;

    // write result to a new csv file
    parser.toCSV("result.csv", module.getData(), module.getLabel());

    return 0;
}