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

    const char *filename = "input.csv";
    //char *filename = argv[0];
    Parser parser(filename);

    KMeans module(parser.rows, parser.cols, 3, parser.data);

    auto t1 = std::chrono::high_resolution_clock::now();
    module.kMeansClustering();
    auto t2 = std::chrono::high_resolution_clock::now();

    //parser.print();
    parser.toCSV("result.csv", module.getData(), module.getLabel());

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Execution Time: " << duration / 1000 << " [ms]" << std::endl;
    
    return 0;

}