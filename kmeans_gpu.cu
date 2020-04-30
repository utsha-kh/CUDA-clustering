#include <iostream>
#include <chrono>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include <random>
#include "kmeans_gpu.h"
#include "parser.h"

int n;  // number of data points
int d;  // dimention of input data (usually 2, for 2D data)
int k;  // number of clusters

__device__ void d_getDistance(float* x1, float* x2, float *ret, int n, int d, int k);
__global__ void d_getMSE(float* dataPoints, int* labels, float* centeroids, float* ret, int n, int d, int k);
__global__ void d_assignDataPoints(float* dataPoints, int* labels, float* centeroids, int n, int d, int k);
__global__ void d_updateCenteroids(float* dataPoints, int* labels, float* centeroids, int* centeroids_sizes, int n, int d, int k);
__global__ void d_getWeights(float* dataPoints, float* centeroids, float* weights, int n, int d, int count);

// return L2 distance squared between 2 points
__device__ void d_getDistance(float* x1, float* x2, float *ret, int n, int d, int k){
	float dist = 0;
    for(int i = 0; i < d; i++){
        dist += (x2[i] - x1[i]) * (x2[i] - x1[i]);
    }
    *ret = dist; 
}

// return current Mean Squared Error value of all points. This is needed to detect convergence.
float getMSE(float** dataPoints, int* labels, float** centeroids){

    float error = 0;
    float* err = new float[n];  // distance between each dataPoints to centeroids
    float *d_dataPoints, *d_centeroids, *d_err; 
    int *d_labels;

    // Allocate memory on GPU
    cudaMalloc(&d_dataPoints, sizeof(float) * n * d);
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_centeroids, sizeof(float) * k * d);
    cudaMalloc(&d_err, sizeof(float) * n);    

    // Flattening both matrix to ease copying to GPU
    float* flattenDataPoints = new float[n * d];
    for(int i = 0; i < n; i++){
        for(int j = 0; j < d; j++){
            flattenDataPoints[i * d + j] = dataPoints[i][j];
        }
    }
    float* flattenCenteroids = new float[n * k];
    for(int i = 0; i < k; i++){
        for(int j = 0; j < d; j++){
            flattenCenteroids[i * d + j] = centeroids[i][j];
        }
    }

    // copy flattened data into GPU
    cudaMemcpy(d_dataPoints, flattenDataPoints, sizeof(float) * n * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centeroids, flattenCenteroids, sizeof(float) * k * d, cudaMemcpyHostToDevice);
    
    // call the kernel function to compute RMSE values in parallel
    int block_size = n / THREAD_PER_BLOCK + (n % THREAD_PER_BLOCK != 0);
    d_getMSE<<<block_size, THREAD_PER_BLOCK>>>(d_dataPoints, d_labels, d_centeroids, d_err, n, d, k);
    cudaDeviceSynchronize();

    // copy back the result from GPU to CPU
    cudaMemcpy(err, d_err, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // Summing up computed errors. could be made faster by parallel reduction
    for(int i = 0; i < n; i++){
        error += err[i];
        //std::cout << "error[" << i << "] = " << err[i] << std::endl;
    }

    // deallocate GPU and CPU memory
    cudaFree(d_dataPoints);
    cudaFree(d_labels);
    cudaFree(d_centeroids);
    cudaFree(d_err);
    delete[] flattenDataPoints;
    delete[] flattenCenteroids;

    // return actual Mean of Squared Errors.
    return error / n;
}

// kernel of above function. NOTE: this is like helper of RMSE. The error values stored in err[] still needs to be summed up.
__global__ void d_getMSE(float* dataPoints, int* labels, float* centeroids, float* err, int n, int d, int k){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;
    d_getDistance(&dataPoints[id * d], &centeroids[labels[id] * d], &err[id], n, d, k); 
}

// Initialize the centeroids, based on k-means++ algorithm.
void initCenters(float** dataPoints, float** centeroids){
    
    float *d_dataPoints, *d_centeroids, *d_weights; 

    int count = 1;
    std::vector<float> weights_vec(n);
    float* weights = new float[n];

    std::cout << "Initializing centeroids basaed on k-means++ Algorighm..." << std::endl;

    std::random_device seedGenerator;
    std::mt19937 randomEngine(seedGenerator());
    std::uniform_int_distribution<> uniformRandom(0, n - 1);

    // 0. pick a random centeroid c1.
    int uniformLottery = uniformRandom(randomEngine);
    for(int i = 0; i < d; i++){
        centeroids[0][i] = dataPoints[uniformLottery][i];
    }
    // Allocate memory on GPU
    cudaMalloc(&d_dataPoints, sizeof(float) * n * d);
    cudaMalloc(&d_centeroids, sizeof(float) * k * d);
    cudaMalloc(&d_weights, sizeof(float) * n);    

    // Flattening both matrix to ease copying to GPU
    float* flattenDataPoints = new float[n * d];
    for(int i = 0; i < n; i++){
        for(int j = 0; j < d; j++){
            flattenDataPoints[i * d + j] = dataPoints[i][j];
        }
    }
    float* flattenCenteroids = new float[n * k];
    for(int i = 0; i < k; i++){
        for(int j = 0; j < d; j++){
            flattenCenteroids[i * d + j] = centeroids[i][j];
        }
    }

    // copy flattened data into GPU
    cudaMemcpy(d_dataPoints, flattenDataPoints, sizeof(float) * n * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centeroids, flattenCenteroids, sizeof(float) * k * d, cudaMemcpyHostToDevice);
    
    int block_size = n / THREAD_PER_BLOCK + (n % THREAD_PER_BLOCK != 0);

    while(count < k){
        // 1. for each data Points x, get Shortest Distance between x and a centeroid D(x)^2. This will be weight of that point. 
        d_getWeights<<<block_size, THREAD_PER_BLOCK>>>(d_dataPoints, d_centeroids, d_weights, n, d, count);
        cudaMemcpy(weights, d_weights, sizeof(float) * n, cudaMemcpyDeviceToHost); 
        // 2. pick a new cluster randomly from data points, with weighted sampling D(x)^2 / total D(x)^2
        for(int i = 0; i < n; i++){
            weights_vec[i] = weights[i];
        }
        std::discrete_distribution<int> weightedRandom(weights_vec.begin(), weights_vec.end());
        int weightedLottery = weightedRandom(randomEngine);
        for(int i = 0; i < d; i++){
            centeroids[count][i] = dataPoints[weightedLottery][i];
        }
        for(int i = 0; i < k; i++){
            for(int j = 0; j < d; j++){
                flattenCenteroids[i * d + j] = centeroids[i][j];
            }
        }   
        cudaMemcpy(d_centeroids, flattenCenteroids, sizeof(float) * k * d, cudaMemcpyHostToDevice);
        count++;
    }

    for(int i = 0; i < k; i++){
        for(int j = 0; j < d; j++){
            centeroids[i][j] = flattenCenteroids[i * d + j];
        }
    } 

    std::cout << "--Finished initialization!!" << std::endl;

    // deallocate GPU memory
    cudaFree(d_dataPoints);
    cudaFree(d_weights);
    cudaFree(d_centeroids);
    delete[] flattenDataPoints;
    delete[] flattenCenteroids;
    delete[] weights; 

}

// kernel of above function
__global__ void d_getWeights(float* dataPoints, float* centeroids, float* weights, int n, int d, int count){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;

    float minDistance = FLT_MAX;
    float dist_i = 0;
    // find the closest centeroid from this dataPoint (&dataPoint[id * d]), and store the distance
    for(int i = 0; i < count; i++){
        d_getDistance(&dataPoints[id * d], &centeroids[i * d], &dist_i, n, d, count);
        if(dist_i < minDistance){
            minDistance = dist_i;
        }
    }
    weights[id] = minDistance;
}

// Assign each data point to the closest centeroid, and store the result in *labels
void assignDataPoints(float** dataPoints, int* labels, float** centeroids){

    float *d_dataPoints, *d_centeroids; 
    int *d_labels;

    // Allocate memory on GPU
    cudaMalloc(&d_dataPoints, sizeof(float) * n * d);
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_centeroids, sizeof(float) * k * d);    

    // Flattening both matrix to ease copying to GPU
    float* flattenDataPoints = new float[n * d];
    for(int i = 0; i < n; i++){
        for(int j = 0; j < d; j++){
            flattenDataPoints[i * d + j] = dataPoints[i][j];
        }
    }
    float* flattenCenteroids = new float[n * k];
    for(int i = 0; i < k; i++){
        for(int j = 0; j < d; j++){
            flattenCenteroids[i * d + j] = centeroids[i][j];
        }
    }

    // copy flattened data into GPU
    cudaMemcpy(d_dataPoints, flattenDataPoints, sizeof(float) * n * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centeroids, flattenCenteroids, sizeof(float) * k * d, cudaMemcpyHostToDevice);
    
    // call the kernel function to compute RMSE values in parallel
    int block_size = n / THREAD_PER_BLOCK + (n % THREAD_PER_BLOCK != 0);
    d_assignDataPoints<<<block_size, THREAD_PER_BLOCK>>>(d_dataPoints, d_labels, d_centeroids, n, d, k);
    cudaDeviceSynchronize();

    // copy back the result from GPU to CPU
    cudaMemcpy(labels, d_labels, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // deallocate GPU memory
    cudaFree(d_dataPoints);
    cudaFree(d_labels);
    cudaFree(d_centeroids);
    delete[] flattenDataPoints;
    delete[] flattenCenteroids;

}

// kernal of above function
__global__ void d_assignDataPoints(float* dataPoints, int* labels, float* centeroids, int n, int d, int k){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;
    int closest = 0;
    float minDistance = FLT_MAX;
    float dist_i = 0;
    // find the closest centeroid (centeroids[closest]) from this dataPoint (&dataPoint[id * d])
    for(int i = 0; i < k; i++){
        d_getDistance(&dataPoints[id * d], &centeroids[i * d], &dist_i, n, d, k);
        if(dist_i < minDistance){
            closest = i;
            minDistance = dist_i;
        }
    }
    labels[id] = closest;
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

// Update each center of sets u_i to the average of all data points who belong to that set
void updateCenteroids(float** dataPoints, int* labels, float** centeroids){

    float *d_dataPoints, *d_centeroids; 
    int *d_labels;
    int *d_centeroids_sizes;
    int *centeroids_sizes = new int[k]; // how many data points are avergaged to count each centeroid

    // Allocate memory on GPU
    cudaMalloc(&d_dataPoints, sizeof(float) * n * d);
    cudaMalloc(&d_labels, sizeof(int) * n);
    cudaMalloc(&d_centeroids, sizeof(float) * k * d);    
    cudaMalloc(&d_centeroids_sizes, sizeof(int) * k);

    // Flattening both matrix to ease copying to GPU
    float* flattenDataPoints = new float[n * d];
    for(int i = 0; i < n; i++){
        for(int j = 0; j < d; j++){
            flattenDataPoints[i * d + j] = dataPoints[i][j];
        }
    }
    float* flattenCenteroids = new float[n * k];
    for(int i = 0; i < k; i++){
        for(int j = 0; j < d; j++){
            flattenCenteroids[i * d + j] = 0; // reset values before passing to GPU. centeroids[i][j];
        }
    }
    for(int i = 0; i < k; i++){
        centeroids_sizes[i] = 0;    // reset values before passing to GPU. 
    }

    // copy flattened data into GPU
    cudaMemcpy(d_dataPoints, flattenDataPoints, sizeof(float) * n * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centeroids, flattenCenteroids, sizeof(float) * k * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centeroids_sizes, centeroids_sizes, sizeof(int) * k, cudaMemcpyHostToDevice);

    // call the kernel function to compute RMSE values in parallel
    // Here, I'm using 2D threads to parallelize double for loop so O(n * k) becomes O(1).
    int block_row_size = THREAD_PER_BLOCK / k;  //e.g. if k == 3, this is 1024 / 3 = 341
    int block_col_size = k;
    int number_of_blocks = n / block_row_size + (n % block_row_size != 0);
        // when k = 3, a block's shape will be 341 x 3 x 1, so it will have 1023 threads in a block, with 2 index per block
    dim3 block_shape(block_row_size, block_col_size, 1);
        // in total, there will be n x k threads. 
    d_updateCenteroids<<< number_of_blocks, block_shape >>>(d_dataPoints, d_labels, d_centeroids, d_centeroids_sizes, n, d, k);
    cudaDeviceSynchronize();

    // copy back the result from GPU to CPU
    cudaMemcpy(flattenCenteroids, d_centeroids, sizeof(float) * k * d, cudaMemcpyDeviceToHost);
    cudaMemcpy(centeroids_sizes, d_centeroids_sizes, sizeof(int) * k, cudaMemcpyDeviceToHost);

    // put the flattened form of centeroids back to matrix form
    for(int i = 0; i < k; i++){
        for(int j = 0; j < d; j++){
            centeroids[i][j] = flattenCenteroids[i * d + j];
        }
    }

    // the centeroids computed by GPU was just sum of all points. Still needs to be divided by counts. 
    for(int i = 0; i < k; i++){
        divideVector(centeroids[i], centeroids_sizes[i], centeroids[i]);
    }

    // deallocate GPU memory
    cudaFree(d_dataPoints);
    cudaFree(d_labels);
    cudaFree(d_centeroids);
    cudaFree(d_centeroids_sizes);
    delete[] flattenDataPoints;
    delete[] flattenCenteroids;
    delete[] centeroids_sizes;

}

// kernel of above function
__global__ void d_updateCenteroids(float* dataPoints, int* labels, float* centeroids, int* centeroids_sizes, int n, int d, int k){
        // Here, each thread has a 2D index (id_x, id_y). Range: id_x = [0, n -1], id_y = [0, k -1]. 
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(id_x >= n || id_y >= k) return;
        // if, the data Point at id_x belongs to the id_y-th centeroid
    if(labels[id_x] == id_y){
        for(int i = 0; i < d; i++){
            atomicAdd(&centeroids[id_y * d + i], dataPoints[id_x * d + i]);
        }
        atomicAdd(&centeroids_sizes[id_y], 1);
    }
}

// need this to check convergence
float myAbs(float a, float b){
    if(a > b)
        return a - b;
    else
        return b - a;
}

// Checks convergence (d/dt < 0.5%)
// the CONVERGENCE_RATE is defined in kmeans_gpu.h
bool hasConverged(float prevError, float currentError){
    return (prevError - currentError) / prevError < CONVERGENCE_RATE || (prevError - currentError) / prevError > -CONVERGENCE_RATE;
}

// Calling this function will do everything for the user
void kMeansClustering(float** dataPoints, int* labels, int n_, int d_, int k_){
    n = n_; d = d_; k = k_; // copy arguments to global variables

    float** centeroids = new float*[k];
    for(int i = 0; i < k; i++){
        centeroids[i] = new float[d];
    }

    initCenters(dataPoints, centeroids);

    int iterations = 0;
    float previousError = FLT_MAX;
    float currentError = 0;
    while(iterations < MAX_ITERATIONS){    
        assignDataPoints(dataPoints, labels, centeroids);
        updateCenteroids(dataPoints, labels, centeroids);
        currentError = getMSE(dataPoints, labels, centeroids);
        std::cout << "(iteration" << iterations << ") Total Error Now: " << std::setprecision(6) << currentError << std::endl;
        if(hasConverged(previousError, currentError)) break;
        previousError = currentError;
        iterations++;
        }
    std::cout << "--Finished. # of iterations: " << iterations << std::endl;

    // free memory
    for(int i = 0; i < k; i++){
        delete[] centeroids[i];
    }
    delete[] centeroids;
}
