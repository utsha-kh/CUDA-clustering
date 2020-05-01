# K-Means clustering with CUDA

In this project, I implemented K-Means clustering in parallel, using CUDA. In my implementaion, I used CUDA as much as possible, and achieved more than 60x performance on large dataset.

My CUDA implementation uses as much as possible CUDA threads. I made the 4 main parts of the K-Means Algorithms to utilize CUDA: 1. centeroids initialization (K-Means++), 2. computing MSE (Mean-Squared-Error) to detect convergence, 3. assigning each data point to closest centeroid, 4. update centeroids as the average of all data who shares the same label.

In addition, I implemented K-Means++. This is a common method to initialize the centeroids of clusters by randomly sampling the input data with a weighted distribution. Without K-Means++ initialization, the program converges to local-optimum very often. I realized K-Means++ strongly helped the K-Means Algorithm to converge to the best possible partitioning. 

## Contents:
    1. A C++ Linear implementation of K-means algorithm.
    2. A CUDA implmentation of K-means algorithm.
    3. A jupyter notebook (.ipynb) to generate cluster input files and visualize results.
    4. Pre-generated, various test files to test my code.
    
## Files (descending order of importance)
    kmeans_gpu.cu -- CUDA implementation of K-Means and K-Means++.
    kmeans_gpu.h -- definition of thread counts etc.
    Visualizations.ipynb -- You can use this to generate .csv files easily. (up to k = 10, any N).
    kmeans.cpp -- Linear implementation of K-means, to make comparison between CUDA version.
    parser.cpp -- Class for import/export the .csv file. I modified the code written by my friend Erick.
    (all other files) -- nothing special.
   
## Usage
Both linear and CUDA program reads from the file "input.csv".

If you want to use a different input, please rename that file to be "input.csv"

Alternatively, you can modify the C++/CUDA main file to specify the desired filename.

To run the linear version, run

    $ make clustering
    $ ./clustering 
To run the CUDA version, run

    $ make cuda_clustering
    $ ./cuda_clustering
You can add an argument to specify K value.

    $ ./cuda_clustering 10
This will perform the clustering with k = 10.

## Important Notes

Note 0 : Even though K-Mean++ initialization stabilizes the result of the program, it still sometimes converges to a local optimum. This happens because of the random-sampling nature during the initialization. If the MSE(Mean-Squared-Error) value seems too large when the program terminates, it probably went to a local optimum. In such case, please run the code few times. When the final MSE is about 0.5, it reached the best solution.

NOTE 1 : PLEASE specify the K value that matches the input file. Otherwise, the result wouldn't be meaningful.

NOTE 2 : The main file does no exception-handling with bad arguments, since that's not the main purpose of this project. Please be nice with input K values and filename. 

NOTE 3 : With too few data points such as N = 10, the algorithm doesn't always converge to the best clusters. Please don't be mean with input size.  
 
## Test .csv files
I have included following .csv files to enable testing easy.

small: N = 1000, mid: N = 100000.

The number in the end means how many clusters. 
    
    small3.csv
    small5.csv
    small10.csv
    mid3.csv
    mid5.csv
    mid10.csv


## Visualizations.ipynb Usage
In this file, you will find two function to generate .csv files and visualize the results.

    generateTestFile("FileName.csv", k = 5, N = 1000)
    
 will create 5 cluster data with 1000 points, and saves as "FileName.csv".
 
 By default, it creates "input.csv" which is also the default filename to be loaded in my C++/CUDA programs.
