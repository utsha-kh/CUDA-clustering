clustering: main.o kmeans.o parser.o
	g++ -Wall -o clustering main.o kmeans.o parser.o

cuda_clustering: main_cuda.o kmeans_gpu.o parser.o
	nvcc -o cuda_clustering main_cuda.o kmeans_gpu.o parser.o

main.o:	main.cpp
	g++ -c main.cpp

main_cuda.o: main_cuda.cu
	nvcc -c main_cuda.cu

kmeans.o: kmeans.cpp kmeans.h
	g++ -c kmeans.cpp

kmeans_gpu.o: kmeans_gpu.cu kmeans_gpu.h
	nvcc -c kmeans_gpu.cu

parser.o: parser.cpp parser.h
	g++ -c parser.cpp

clean:
	rm -f *.o


