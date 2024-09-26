#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd $ROOT_DIR

if [ ! -d "build" ]; then
  mkdir build
fi

cd build

# set path to your own libfaiss.a installation folder

cmake -DFAISS_LIB=/usr/local/lib/libfaiss.a -DCMAKE_BUILD_TYPE=Release ..


make -j$(nproc)

