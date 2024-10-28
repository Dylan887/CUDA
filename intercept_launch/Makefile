# Makefile for INTERCEPT_LAUNCH project with PTX generation

# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/cuda-12.5/include
NVCCFLAGS = -arch=sm_86 -I/usr/local/cuda-12.5/iclude --compiler-options '-fPIC'
LDFLAGS = -L/usr/local/cuda-12.5/lib64 -lcuda -lcudart -lcupti


# Executable name
EXEC = intercept_launch

# Source files
CUDA_SOURCES = callbackfunc.cu kernels_launch.cu CUDA_init_destory.cu streams.cu load_model.cu main.cu CUPTI_init_unsubscribe.cu 
HEADERS = interceptor.h globalconfig.h

# PTX file
PTX_FILE = kernels.ptx

# Object files
OBJECTS = $(CUDA_SOURCES:.cu=.o)

# Default target
all: $(EXEC)

# Compile CUDA source files and generate PTX file
$(PTX_FILE): kernels.cu
	$(NVCC) $(NVCCFLAGS) -ptx kernels.cu -o $(PTX_FILE)

# Compile CUDA source files
%.o: %.cu $(HEADERS) $(PTX_FILE)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link the object files to create the executable
$(EXEC): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(EXEC) $(LDFLAGS)

# Clean the build directory
clean:
	rm -f $(OBJECTS) $(EXEC) $(PTX_FILE)

.PHONY: clean all

