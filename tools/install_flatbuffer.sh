git clone https://github.com/google/flatbuffers.git
cd flatbuffers
mkdir build
cd build
cmake ../
make -j4
cp -r ../include/flatbuffers ../../../src
sudo cp flatc /usr/bin
