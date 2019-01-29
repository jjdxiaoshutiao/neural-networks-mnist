#!/bin/bash

mkdir -p image/test_set image/training_set
cd src/
cmake .
make
./image_construct