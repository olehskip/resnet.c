NVCC = nvcc
INFERENCE_BIN = cuda_inference_out
INFERENCE_BIN_DEBUG = cuda_inference_out_debug

${INFERENCE_BIN}: cuda/inference/main.cu cuda/ops.cu
	$(NVCC) -I./cuda $^ -o $@ -std=c++20

${INFERENCE_BIN_DEBUG}: cuda/inference/main.cu cuda/ops.cu
	$(NVCC) -I./cuda -g -O0 $^ -o $@ -std=c++20 -DDEBUG


clean:
	rm -f ${INFERENCE_BIN} ${INFERENCE_BIN_DEBUG}