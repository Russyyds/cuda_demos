BUILD_DIR=./build
INSTALL_DIR=./install
rm -r ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
cmake --build . --config Release
cmake --install .