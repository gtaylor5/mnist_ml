CC=g++
SRC := $(PWD)/src
CFLAGS := -std=c++11 -g -DRELU -DMNIST
INCLUDE_DIR := $(PWD)/include/

all: main

main : obj $(SRC)/network.cpp obj/hidden_layer.o obj/output_layer.o obj/layer.o obj/input_layer.o obj/neuron.o
	$(CC) $(CFLAGS) $(SRC)/network.cpp -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata ./obj/*.o

obj:
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/hidden_layer.o : $(SRC)/hidden_layer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/hidden_layer.cpp -o obj/hidden_layer.o -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

obj/output_layer.o : $(SRC)/output_layer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/output_layer.cpp -o obj/output_layer.o -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

obj/input_layer.o : $(SRC)/input_layer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/input_layer.cpp -o obj/intput_layer.o -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

obj/layer.o : $(SRC)/layer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/layer.cpp -o obj/layer.o -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

obj/neuron.o : $(SRC)/neuron.cpp
	$(CC) -c $(CFLAGS) $(SRC)/neuron.cpp -o obj/neuron.o -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata


clean:
	rm -r main obj
