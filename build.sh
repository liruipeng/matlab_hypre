#!/bin/bash

dir=$(pwd)

if [ -d "hypre_v2290" ]
then
   echo use existing hypre...
else
   echo Download hypre...
   git clone git@github.com:hypre-space/hypre.git hypre_v2290
fi
cd hypre_v2290
git checkout tags/v2.29.0

echo Build hypre...
cd src
./configure --enable-debug --without-MPI --enable-shared
make -j

echo install hypre...
make install

cd $dir
rm -f hypre
ln -s hypre_v2290/src/hypre hypre

echo Compile mex
cd mex
make
