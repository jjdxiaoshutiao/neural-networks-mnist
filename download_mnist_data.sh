#!/bin/bash

mkdir data
wget -v --directory-prefix=data http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
  http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
    http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
      http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
DATAS=./data/*
for d in $DATAS
do
  echo "Uncompress " $d "..."
  gzip -d $d
done