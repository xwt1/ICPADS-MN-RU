#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd $ROOT_DIR

mkdir -p data && cd data

# download imageNet dataset
mkdir -p imageNet && cd imageNet
mkdir -p hnsw_prime
wget https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/imagenet.tar.gz --no-check-certificate
tar -xvf imagenet.tar.gz
cd ..

# download gist dataset
mkdir -p gist && cd gist
mkdir -p hnsw_prime
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xvf gist.tar.gz
cd ..

# download sift dataset
mkdir -p sift && cd sift
mkdir -p hnsw_prime
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xvf sift.tar.gz
cd ..

# download sift1B dataset to build sift2M
mkdir -p sift_2M && cd sift_2M
mkdir -p hnsw_prime
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
tar -xvf bigann_base.bvecs.gz
tar -xvf bigann_query.bvecs.gz
cd ..


