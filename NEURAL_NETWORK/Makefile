CC=g++
SRC := $(PWD)/src
CFLAGS := -std=c++11 -g -DMNIST -DSIGMOID
INCLUDE_DIR := $(PWD)/include/

all: main

main : obj $(SRC)/network.cpp obj/layer.o obj/input_layer.o obj/neuron.o
	$(CC) $(CFLAGS) $(SRC)/network.cpp -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata ./obj/*.o

obj:
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/input_layer.o : $(SRC)/input_layer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/input_layer.cpp -o obj/intput_layer.o -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include

obj/layer.o : $(SRC)/layer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/layer.cpp -o obj/layer.o -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include

obj/neuron.o : $(SRC)/neuron.cpp
	$(CC) -c $(CFLAGS) $(SRC)/neuron.cpp -o obj/neuron.o -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include


clean:
	rm -r main obj
