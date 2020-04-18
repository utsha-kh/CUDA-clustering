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

    KMeans module(parser.rows, parser.cols, 2, parser.data);

    module.kMeansClustering();
    //parser.print();

    return 0;

}