BUILD_DIR=./build
INSTALL_DIR=./install
if [ -d ${BUILD_DIR} ]; then
echo "Removing old build directory"
rm -r ${BUILD_DIR}
fi
if [ -d ${INSTALL_DIR} ]; then
echo "Removing old install directory"
rm -r ${INSTALL_DIR}
fi
mkdir ${INSTALL_DIR}
mkdir ${BUILD_DIR}
cmake -B ${BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ${BUILD_DIR} --config Release
cmake --install ${BUILD_DIR} --prefix ${INSTALL_DIR}