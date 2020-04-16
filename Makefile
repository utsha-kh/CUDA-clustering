clustering: main.o kmeans.o parser.o
	g++ -Wall -o clustering main.o kmeans.o parser.o

main.o:	main.cpp
	g++ -c main.cpp

kmeans.o: kmeans.cpp
	g++ -c kmeans.cpp

parser.o: parser.cpp
	g++ -c parser.cpp



