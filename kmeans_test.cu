#include <iostream>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include "kmeans_gpu.h"
#include "parser.h"

#define THREAD_SIZE 1024

int n;  // number of data points
int d;  // dimention of input data (usually 2, for 2D data)
int k;  // number of clusterss

__device__ void d_getDistance(float* x1, float* x2, float* ret);
__global__ void d_getRMSE(float** dataPoints, int* labels, float** centeroids, float* ret);


float myAbs(float a, float b){
    if(a > b)
        return a - b;
    else
        return b - a;
}

// Checks convergence (d/dt < 0.5%)
bool hasConverged(float prevError, float currentError){
    return myAbs(prevError, currentError) / prevError < 0.005;
}

// Calling this function will do everything for the user
void kMeansClustering(float** dataPoints, int* labels){
    float** centeroids = new float*[k];
    for(int i = 0; i < d; i++){
        centeroids[i] = new float[d];
    }

    initCenters(dataPoints, centeroids);
    int iterations = 0;
    float previousError = FLT_MAX;
    float currentError = 0;
    while(true){
        assignDataPoints(dataPoints, labels, centeroids);
        updateCenters(dataPoints, labels, centeroids);
        getRMSE(dataPoints, labels, centeroids, &currentError);
        if(hasConverged(previousError, currentError)) break;
        previousError = currentError;
        iterations++;
        std::cout << "Total Error Now: " << std::setprecision(6) << currentError << std::endl;
    }
    std::cout << "# of iterations: " << iterations << std::endl;
}

int main(){

    const char *filename = "input.csv";
    Parser parser(filename);

    n = parser.rows; d = parser.cols;

    float** data = parser.rdata;
    int* labels = new int[n];


    kMeansClustering(data, labels);
    
    parser.toCSV("result.csv", data, labels, n, d);
    return 0;


}
