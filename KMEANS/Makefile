CC=g++
SRC := $(PWD)/src
CFLAGS := -std=c++11
INCLUDE_DIR := $(PWD)/include/

all: main

main : $(SRC)/kmeans.cc $(SRC)/cluster.cc
	$(CC) $(CFLAGS) $(SRC)/* -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

clean:
	rm main