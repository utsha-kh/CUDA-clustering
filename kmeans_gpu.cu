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
__global__ void d_addVector(float* x1, float* x2, float* ret);
__global__ void d_divideVector(float* x, int s, float* ret);
__global__ void d_assignDataPoints(float** dataPoints, int* labels, float** centeroids);
__global__ void d_updateCenters(float** dataPoints, float** centeroids);

// return L2 distance between two points
float getDistance(float* x1, float* x2){
    float dist = 0;
    for(int i = 0; i < d; i++){
        dist += (x2[i] - x1[i]) * (x2[i] - x1[i]);
    }
    return dist; 
}

// return L2 distance between 2 points
__device__ void d_getDistance(float* x1, float* x2, float *ret){
	float dist = 0;
    for(int i = 0; i < 2; i++){
        dist += (x2[i] - x1[i]) * (x2[i] - x1[i]);
    }
    *ret = dist; 
}

// return current Root Mean Squared Error value of all points
float getRMSE(float** dataPoints, int* labels, float** centeroids, float* err){

    float error = 0;
    float **d_dataPoints, **d_centeroids, *d_err; 
    int *d_labels;

    cudaMalloc(&d_dataPoints, sizeof(float) * n * d);
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_centeroids, sizeof(float) * k * d);
    cudaMalloc(&d_err, sizeof(float) * n);

    int block_size = n / THREAD_SIZE + (n % THREAD_SIZE != 1);

    cudaMemcpy(d_dataPoints, dataPoints, sizeof(float) * n * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centeroids, centeroids, sizeof(float) * k * d, cudaMemcpyHostToDevice);
    d_getRMSE<<<block_size, THREAD_SIZE>>>(d_dataPoints, d_labels, d_centeroids, d_err);

    cudaMemcpy(err, d_err, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // could be made faster by parallel reduction
    for(int i = 0; i < n; i++){
        error += err[i];
    }

    return sqrt(error / n);
}

// kernel of above function
__global__ void d_getRMSE(float** dataPoints, int* labels, float** centeroids, float* err){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= 1000) return;
    d_getDistance(dataPoints[id], centeroids[labels[id]], &err[id]);
}

// add two vectors
__global__ void d_addVector(float* x1, float* x2, float* ret){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= 1000) return;
    ret[id] = x1[id] + x2[id];
}

void addVector(float* x1, float* x2, float* ret){
    for(int i = 0 ; i < d; i++){
        ret[i] = x1[i] + x2[i];
    }
}

// divide vector by scaler
__global__ void d_divideVector(float* x, int s, float* ret){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= 1000) return;
    ret[id] = x[id] / (float)s;
}

void divideVector(float* x, int s, float* ret){
    for(int i = 0; i < d; i++){
        ret[i] = x[i] / (float)s;
    }
}

// initialize each center values u_i to a randomly chosen data point
void initCenters(float** dataPoints, float** centeroids){
    // Each center u[i] should be a random data point x[j], but 
    // generating a non-repeated random number isn't straightforward
    // so I'll do it later
    for(int i = 0; i < k; i++){
        centeroids[i] = dataPoints[i];
    } 
}

// Assign each data point to the closest centeroid, and store the result in *labels
void assignDataPoints(float** dataPoints, int* labels, float** centeroids){
    for(int i = 0; i < n; i++){
        int closest = 0;
        float minDistance = FLT_MAX;
        for(int j = 0; j < k; j++){
            if(getDistance(dataPoints[i], centeroids[j]) < minDistance){
                closest = j;
                minDistance = getDistance(dataPoints[i], centeroids[j]);
            }
        }
        labels[i] = closest;
    }
}

// kernal of above function
__global__ void d_assignDataPoints(float** dataPoints, int* labels, float** centeroids){
    return;
}

// Update each center of sets u_i to the average of all data points who belong to that set
void updateCenters(float** dataPoints, int* labels, float** centeroids){
    int count = 0;
    for(int i = 0; i < k; i++){
        float* sum = new float[d];
        for(int j = 0; j < n; j++){
            if(labels[j] == i){
                addVector(sum, dataPoints[j], sum);
                count++;
            }
        }
        divideVector(sum, count, centeroids[i]);
    }
}

// kernel of above function
__global__ void d_updateCenters(float** dataPoints, int* labels, float** centeroids){
    return;
}


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
