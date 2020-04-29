clustering: main.o kmeans.o parser.o
	g++ -Wall -o clustering main.o kmeans.o parser.o

cuda_clustering: kmeans_gpu.o parser.o
	nvcc -o cuda_clustering kmeans_gpu.o parser.o

test: kmeans_test.o parser.o
	nvcc -o test kmeans_test.o parser.o

main.o:	main.cpp
	g++ -c main.cpp

kmeans.o: kmeans.cpp
	g++ -c kmeans.cpp

kmeans_gpu.o: kmeans_gpu.cu
	nvcc -c kmeans_gpu.cu

kmeans_test.o: kmeans_test.cu
	nvcc -c kmeans_test.cu

parser.o: parser.cpp
	g++ -c parser.cpp

clean:
	rm -f *.o


