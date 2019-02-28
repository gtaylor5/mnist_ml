CC=g++
SRC := $(PWD)/src
CFLAGS := -std=c++11 -DEUCLID
INCLUDE_DIR := $(PWD)/include/

all: main

main : $(SRC)/rbfnn.cc $(MNIST_ML_ROOT)/KMEANS/kmeans.o
	$(CC) $(CFLAGS) $(SRC)/rbfnn.cc -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -ldata $(MNIST_ML_ROOT)/KMEANS/kmeans.o

$(MNIST_ML_ROOT)/KMEANS/kmeans.o :
	cd $(MNIST_ML_ROOT)/KMEANS/
	make
	cd $(MNIST_ML_ROOT)/RBFNN/

clean:
	rm main