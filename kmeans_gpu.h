#ifndef __KMEANS_GPU__H
#define __KMEANS_GPU__H

#define THREAD_SIZE 1024
#define MAX_ITERATIONS 100

void kMeansClustering(float** dataPoints, int* labels, int n, int d, int k);

float getMSE(float** dataPoints, int* labels, float** centeroids);
void initCenters(float** dataPoints, float** centeroids);
void assignDataPoints(float** dataPoints, int* labels, float** centeroids);
void updateCenteroids(float** dataPoints, int* labels, float** centeroids);

#endif