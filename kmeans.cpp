#include <iostream>
#include <vector>
#include <set>

using namespace std;

// Need to establish a way to detect convergence.
// Need to implement random selection to initialize centers.

// Also, need a way to visualize vectors so that results can be easily seen.

class kMeans{
private:
    int n;  // number of data points
    int d;  // dimension of data points (vectors)
    int k;  // number of groups to cluster
    vector<vector<int>> x;  // input data points
    vector<int> whichSet;  // stores info that which set a vector belong to
    vector<vector<int>> u; // centers of each of k sets
    //vector<set<vector<int>>> S; // k different sets of vectors
    bool converged = false;

    // return L2 distance between two points
    int getDistance(vector<int> x1, vector<int> x2){
        int dist = 0;
        for(int i = 0; i < x1.size; i++){
            dist += (x2[i] - x1[i]) * (x2[i] - x1[i]);
        }
        return dist; 
    }

    // add two vectors
    vector<int> addVector(vector<int> x1, vector<int> x2){
        vector<int> retVector(x1.size, 0);
        for(int i = 0; i < x1.size; i++){
            retVector[i] = x1[i] + x2[i];
        }
        return retVector;
    }

    // initialize each center values u_i to a randomly chosen data point
    void initCenters(){
        // Each center u[i] should be a random data point x[j], but 
        // generating a non-repeated random number isn't straightforward
        // so I'll do it later
        for(int i = 0; i < k; i++){
            u[i] = x[i];
        } 
    }

    // Assign each data point x_i to the closest center u_j
    void assignDataPoints(){
        for(int i = 0; i < n; i++){
            //S[whichSet[i]].erase(x[i]);
            int closest = 0;
            int minDistance = INTMAX_MAX;
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
    void updateCenters(){
        for(int i = 0; i < k; i++){
            vector<int> sum(d, 0); // a d-dimensional vector
            for(int j = 0; j < n; j++){
                if(whichSet[j] == i){
                    sum = addVector(sum, x[j]);
                }
            }
            u[i] = sum;
        }
    }

public:
    kMeans(int n, int d, int k, vector<vector<int>> x){
        this->n = n;
        this->d = d;
        this->k = k;
        this->x = x;
        this->whichSet.resize(n);
        this->u.resize(k);
    }

    // Calling this function will do everything for the user
    void kMeansClustering(){
        initCenters();
        while(!converged){
            assignDataPoints();
            updateCenters();
        }
    }

};