CC=g++
INCLUDE_DIR := $(MNIST_ML_ROOT)/include
SRC := $(MNIST_ML_ROOT)/src
CFLAGS := -std=c++11 -g
LIB_DATA := libdata.so

all : $(LIB_DATA)

$(LIB_DATA) : libdir objdir obj/data_handler.o obj/data.o obj/common.o
	$(CC) $(CFLAGS) -shared -o $(MNIST_ML_ROOT)/lib/$(LIB_DATA) obj/data_handler.o obj/data.o obj/common.o

libdir :
	mkdir -p $(MNIST_ML_ROOT)/lib

objdir:
	mkdir -p $(MNIST_ML_ROOT)/obj

obj/data_handler.o: $(SRC)/DataHandler.cc
	$(CC) -fPIC $(CFLAGS) -o obj/data_handler.o -I$(INCLUDE_DIR)/*.h -c $(SRC)/DataHandler.cc

obj/data.o: $(SRC)/Data.cc
	$(CC) -fPIC $(CFLAGS)  -o obj/data.o -I$(INCLUDE_DIR)/*.h -c $(SRC)/Data.cc

obj/common.o: $(SRC)/Common.cc
	$(CC) -fPIC $(CFLAGS)  -o obj/common.o -I$(INCLUDE_DIR)/* -c $(SRC)/Common.cc

clean:
	rm -r $(MNIST_ML_ROOT)/lib
	rm -r $(MNIST_ML_ROOT)/obj
