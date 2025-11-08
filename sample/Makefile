NVFLAGS := -std=c++11 -O3 -Wno-deprecated-gpu-targets
TARGET := hw4
TARGET_CPU := hw4_cpu

.PHONY: all
all: $(TARGET)

# Main target: CUDA GPU-accelerated version (hw4.cu is the GPU version now)
$(TARGET): hw4.cu sha256.h
	nvcc $(NVFLAGS) -o hw4 hw4.cu

# Optional: CPU version for comparison
$(TARGET_CPU): sha256.o hw4_cpu.cu
	nvcc $(NVFLAGS) -o hw4_cpu hw4_cpu.cu sha256.o

sha256.o: sha256.cu sha256.h
	nvcc $(NVFLAGS) -c sha256.cu

clean:
	rm -rf hw4 hw4_cpu *.o




