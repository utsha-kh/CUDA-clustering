#include <iostream>
#include <vector>
#include <set>
#include <cfloat>
#include <cmath>
#include "kmeans.h"

using namespace std;

// Need to establish a way to detect convergence.
// Need to implement random selection to initialize centers.

// Also, need a way to visualize vectors so that results can be easily seen.

// return L2 distance between two points
float KMeans::getDistance(vector<float> x1, vector<float> x2){
    float dist = 0;
    for(int i = 0; i < x1.size(); i++){
        dist += (x2[i] - x1[i]) * (x2[i] - x1[i]);
    }
    return dist; 
}

// return current Root Mean Squared Error value
float KMeans::getRMSE(void){
    float err = 0;
    for(int i = 0; i < n; i++){
        err += sqrt(getDistance(x[i], u[whichSet[i]]));
    }
    return err;
}

// add two vectors
vector<float> KMeans::addVector(vector<float> x1, vector<float> x2){
    vector<float> retVector(x1.size());
    for(int i = 0; i < x1.size(); i++){
        retVector[i] = x1[i] + x2[i];
    }
    return retVector;
}

// initialize each center values u_i to a randomly chosen data point
void KMeans::initCenters(){
    // Each center u[i] should be a random data point x[j], but 
    // generating a non-repeated random number isn't straightforward
    // so I'll do it later
    for(int i = 0; i < k; i++){
        u[i] = x[i];
    } 
}

// Assign each data point x_i to the closest center u_j
void KMeans::assignDataPoints(){
    for(int i = 0; i < n; i++){
        //S[whichSet[i]].erase(x[i]);
        int closest = 0;
        float minDistance = FLT_MAX;
        for(int j = 0; j < k; j++){
            if(getDistance(x[i], u[j]) < minDistance){
                closest = j;
                minDistance = getDistance(x[i], u[j]);
            }
        }
        whichSet[i] = closest;
        //S[whichSet[i]].insert(x[i]);
    }
}

// Update each center of sets u_i to the average of all data points who belong to that set
void KMeans::updateCenters(){
    for(int i = 0; i < k; i++){
        vector<float> sum(d, 0); // a d-dimensional vector
        for(int j = 0; j < n; j++){
            if(whichSet[j] == i){
                sum = addVector(sum, x[j]);
            }
        }
        u[i] = sum;
    }
}

KMeans::KMeans(int n, int d, int k, vector<vector<float>> x){
    this->n = n;
    this->d = d;
    this->k = k;
    this->x = x;
    this->whichSet.resize(n);
    this->u.resize(k);
}

// Calling this function will do everything for the user
void KMeans::kMeansClustering(){
    initCenters();
    while(!converged){
        assignDataPoints();
        updateCenters();
        printf("Total Error Now: %3f\n", getRMSE());
    }
}