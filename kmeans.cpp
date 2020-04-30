#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <cfloat>
#include <cmath>
#include <random>
#include <list>
#include "kmeans.h"

// Need to establish a way to detect convergence.
// Need to implement random selection to initialize centers.

// Also, need a way to visualize vectors so that results can be easily seen.

// return L2 distance between two points
float KMeans::getDistance(std::vector<float> x1, std::vector<float> x2){
    float dist = 0;
    for(int i = 0; i < x1.size(); i++){
        dist += (x2[i] - x1[i]) * (x2[i] - x1[i]);
    }
    return dist; 
}

// return current Mean Squared Error value
float KMeans::getMSE(void){
    float err = 0;
    for(int i = 0; i < n; i++){
        err += getDistance(dataPoints[i], centeroids[labels[i]]);
    }
    return err / n;
}

// add two vectors
std::vector<float> KMeans::addVector(std::vector<float> x1, std::vector<float> x2){
    std::vector<float> retVector(x1.size());
    for(int i = 0; i < x1.size(); i++){
        retVector[i] = x1[i] + x2[i];
    }
    return retVector;
}

// divide vector by scaler
std::vector<float> KMeans::divideVector(std::vector<float> v, int s){
    for(int i = 0; i < v.size(); i++){
        v[i] /= (float)s;
    }
    return v;
}

// initialize each center values u_i to a randomly chosen data point
void KMeans::initCenters(){
    // // Each center u[i] should be a random data point x[j], but 
    // // generating a non-repeated random number isn't straightforward
    // // so I'll do it later
    // for(int i = 0; i < k; i++){
    //     centeroids[i] = dataPoints[i];
    // } 

    std::cout << "initializing centeroids basaed on k-means++ Algorighm..." << std::endl;

    std::random_device seedGenerator;
    std::mt19937 randomEngine(seedGenerator());
    std::uniform_int_distribution<> uniformRandom(0, n - 1);

    int count = 1;
    std::vector<float> weights(n);
    // 0. pick a random centeroid c1.
    centeroids[0] = dataPoints[uniformRandom(randomEngine)];
    while(count < k){
        // 1. for each data Points x, get Shortest Distance between x and a centeroid D(x)^2. This will be weight of that point. 
        for(int i = 0; i < n; i++){
            float minDistance = FLT_MAX;
            for(int j = 0; j < count; j++){
                if(getDistance(dataPoints[i], centeroids[j]) < minDistance){
                    minDistance = getDistance(dataPoints[i], centeroids[j]);
                }    
            }
            weights[i] = minDistance * minDistance; // save weight
        }
        // 2. pick a new cluster randomly from data points, with weighted sampling D(x)^2 / total D(x)^2
        std::discrete_distribution<int> weightedRandom(weights.begin(), weights.end());
        centeroids[count] = dataPoints[weightedRandom(randomEngine)];
        count++;
    }

    std::cout << "Finished initialization!!" << std::endl;

}

// Assign each data point x_i to the closest center u_j
void KMeans::assignDataPoints(){
    for(int i = 0; i < n; i++){
        //S[whichSet[i]].erase(x[i]);
        int closest = 0;
        float minDistance = FLT_MAX;
        for(int j = 0; j < k; j++){
            if(getDistance(dataPoints[i], centeroids[j]) < minDistance){
                closest = j;
                minDistance = getDistance(dataPoints[i], centeroids[j]);
            }
        }
        labels[i] = closest;
        //S[whichSet[i]].insert(x[i]);
    }
}

// Update each center of sets u_i to the average of all data points who belong to that set
void KMeans::updateCenters(){
    int numVectors = 0;
    for(int i = 0; i < k; i++){
        std::vector<float> sum(d, 0); // a d-dimensional vector
        for(int j = 0; j < n; j++){
            if(labels[j] == i){
                sum = addVector(sum, dataPoints[j]);
                numVectors++;
            }
        }
        centeroids[i] = divideVector(sum, numVectors);
    }
}

KMeans::KMeans(int n, int d, int k, std::vector<std::vector<float> > dataPoints){
    this->n = n;
    this->d = d;
    this->k = k;
    this->dataPoints = dataPoints;
    this->labels.resize(n);
    this->centeroids.resize(k);
    this->converged = false;
}

float myAbs(float a, float b){
    if(a > b)
        return a - b;
    else
        return b - a;
}

// Calling this function will do everything for the user
void KMeans::kMeansClustering(){
    initCenters();
    int iterations = 0;
    previousError = FLT_MAX;
    while(true){
        assignDataPoints();
        updateCenters();
        float currentError = getMSE();
        if(hasConverged(previousError, currentError)) break;
        previousError = currentError;
        iterations++;
        std::cout << "Total Error Now: " << std::setprecision(6) << currentError << std::endl;
    }
    std::cout << "# of iterations: " << iterations << std::endl;
}

// Checks convergence (d/dt < 0.5%)
bool KMeans::hasConverged(float prevError, float currentError){
    return myAbs(prevError, currentError) / prevError < 0.005;
}

std::vector<std::vector<float> > KMeans::getData(){
    return dataPoints;
}

std::vector<int> KMeans::getLabel(){
    return labels;
}



