#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include "kmeans.h"
#include "parser.h"

int main (int argc, char** argv) {

    const char *filename = "input.csv";
    //char *filename = argv[0];
    Parser parser(filename);

    KMeans module(parser.rows, parser.cols, 5, parser.data);

    module.kMeansClustering();
    //parser.print();
    parser.toCSV("result.csv", module.getData(), module.getLabel());
    return 0;

}