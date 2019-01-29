#!/bin/bash

cd src/neural_network/
echo "Compiling..."
g++ --std=c++11 -larmadillo main.cc neural_network.cc sigmoid.cc ../data_loader/data_loader.cc -o neural_network
echo "Running..."
./neural_network