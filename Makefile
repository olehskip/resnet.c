NVCC = nvcc
INFERENCE_BIN=cuda_inference_out

${INFERENCE_BIN}: cuda/inference/main.cu cuda/ops.cu
	$(NVCC) -I./cuda $^ -o $@

clean:
	rm -f ${INFERENCE_BIN}