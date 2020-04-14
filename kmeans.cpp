#include <iostream>
#include <vector>
#include <set>

using namespace std;

class kMeans{
private:
    int n;  // number of data points
    int d;  // dimension of data points (vectors)
    int k;  // number of groups to cluster
    vector<vector<int>> x;  // input data points
    vector<int> whichSet;  // stores info that which set a vector belong to
    vector<int> u; // centers of each of k sets
    set<vector<int>> S; // k different sets of vectors

    int getDistance(vector<int> x1, vector<int> x2){
        int d = 0;
        for(int i = 0; i < x1.size; i++){
            d += (x2[i] - x1[i]) * (x2[i] - x1[i]);
        }
        return d; 
    }



public:
    kMeans(int n, int d, int k, vector<vector<int>> x){
        this->n = n;
        this->d = d;
        this->k = k;
        this->x = x;
        this->whichSet = new vector<int>(n);
        this->u = new vector<int>(k);
        this->S = new set<vector<int>>(k);
    }

    void kMeansClustering(){
        return;
    }

}