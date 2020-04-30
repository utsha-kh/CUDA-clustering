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
    std::cout << "Loading input file..." << std::endl;
    const char *filename = "input.csv";
    Parser parser(filename);
    std::cout << "--Finished loading!" << std::endl;

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
    std::cout << "Writing the result to \"result.csv\"... " << std::endl;
    parser.toCSV("result.csv", module.getData(), module.getLabel());
    std::cout << "--Finished writing!" << std::endl;

    return 0;
}