NVCC = nvcc
INFERENCE_BIN = cuda_inference_out
INFERENCE_BIN_DEBUG = cuda_inference_out_debug

${INFERENCE_BIN}: cuda/inference/main.cu cuda/ops.cu
	$(NVCC) -I./cuda $^ -o $@

${INFERENCE_BIN_DEBUG}: cuda/inference/main.cu cuda/ops.cu
	$(NVCC) -I./cuda -g -O0 $^ -o $@

clean:
	rm -f ${INFERENCE_BIN} ${INFERENCE_BIN_DEBUG}