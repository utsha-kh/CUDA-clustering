#ifndef __KMEANS_GPU__H
#define __KMEANS_GPU__H

/*
from https://cuda.readthedocs.io/ko/latest/rtx2080Ti/

Device : "GeForce RTX 2080 Ti"
driverVersion : 10010
runtimeVersion : 10000
        CUDA Driver Version / Runtime Version  10.1 / 10.0
        CUDA Capability Major/Minor version number : 7.5
        Total amount of global memory : 10.73 GBytes (11523260416 bytes)
        GPU Clock rate :        1545 MHz(1.54 GHz)
        Memory Clock rate :     7000 Mhz
        Memory Bus Width :      352-bit
        L2 Cache Size:  5767168 bytes
        Total amount of constant memory:        65536 bytes
        Total amount of shared memory per block:        49152 bytes
        Total number of registers available per block:  65536
        Warp Size:      32
        Maximum number of threads per multiprocessor:   1024
        Maximum number of thread per block:     1024
        Maximum sizes of each dimension of a block:     1024 x 1024 x 64
        Maximum sizes of each dimension of a grid:      2147483647 x 65535 x 65535
*/

#define MAX_THREAD_PER_BLOCK 1024
#define MAX_BLOCK_X 1024
#define MAX_BLOCK_Y 1024
#define MAX_BLOCK_Z 64
#define MAX_GRID_X 2147483647
#define MAX_GRID_Y 65535
#define MAX_GRID_Z 65535

#define THREAD_PER_BLOCK 1024
#define MAX_ITERATIONS 100
#define CONVERGENCE_RATE 0.0001

void kMeansClustering(float** dataPoints, int* labels, int n, int d, int k);

float getMSE(float** dataPoints, int* labels, float** centeroids);
void initCenters(float** dataPoints, float** centeroids);
void assignDataPoints(float** dataPoints, int* labels, float** centeroids);
void updateCenteroids(float** dataPoints, int* labels, float** centeroids);

#endif

