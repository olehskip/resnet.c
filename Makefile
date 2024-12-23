NVCC = nvcc
CXX = g++
INFERENCE_BIN = cuda_inference_out
INFERENCE_BIN_DEBUG = cuda_inference_out_debug
TORCH_BIN = pytorch_test_out

LIBTORCH = /home/skip/libtorch
CUDA_PATH = /opt/cuda

CXX_DEFINES = -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE
CXX_INCLUDES = -isystem ${LIBTORCH}/include -isystem ${LIBTORCH}/include/torch/csrc/api/include -isystem ${CUDA_PATH}/include
CXX_FLAGS = -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++20

TORCH_LIBS = -L${LIBTORCH}/lib -L${CUDA_PATH}/lib \
	-Wl,-rpath,${LIBTORCH}/lib:${CUDA_PATH}/lib \
	${LIBTORCH}/lib/libtorch.so \
	${LIBTORCH}/lib/libc10.so \
	${LIBTORCH}/lib/libkineto.a \
	${CUDA_PATH}/lib/libnvrtc.so \
	${LIBTORCH}/lib/libc10_cuda.so \
	-Wl,--no-as-needed,${LIBTORCH}/lib/libtorch_cpu.so -Wl,--as-needed \
	-Wl,--no-as-needed,${LIBTORCH}/lib/libtorch_cuda.so -Wl,--as-needed \
	${LIBTORCH}/lib/libc10_cuda.so \
	${LIBTORCH}/lib/libc10.so \
	${CUDA_PATH}/lib/libcudart.so \
	-Wl,--no-as-needed,${LIBTORCH}/lib/libtorch.so -Wl,--as-needed \
	${CUDA_PATH}/lib/libnvToolsExt.so

${INFERENCE_BIN}: cuda/inference/main.cu cuda/ops.cu
	$(NVCC) -I./cuda $^ -o $@ -std=c++20

${INFERENCE_BIN_DEBUG}: cuda/inference/main.cu cuda/ops.cu
	$(NVCC) -I./cuda -g -O0 $^ -o $@ -std=c++20 -DDEBUG

${TORCH_BIN}: cuda/pytorch_test.cpp
	$(CXX) $^ -o $@ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) $(TORCH_LIBS)

clean:
	rm -f ${INFERENCE_BIN} ${INFERENCE_BIN_DEBUG} ${TORCH_BIN}