# Barnes-Hut N-body simulation
#
#   make                      build ./bh (single precision, seq + omp backends)
#   make PRECISION=double     build ./bh_double
#   make ARCH=native          add -march=native (only if every node has the same CPU)
#   make CUDA=1               also compile src/cuda/*.cu and enable --backend cuda
#   make test                 build, then run the validation tests
#   make clean

CXX       ?= g++
NVCC      ?= nvcc
CUDA_HOME ?= /usr/local/cuda
PRECISION ?= float
ARCH      ?=
CUDA      ?= 0

CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
CXXFLAGS  += -fopenmp
CPPFLAGS  += -Iinclude
LDFLAGS   += -fopenmp
NVCCFLAGS ?= -O3 -std=c++17

ifneq ($(ARCH),)
CXXFLAGS += -march=$(ARCH)
endif

BIN   := bh
BUILD := build/$(PRECISION)
ifeq ($(PRECISION),double)
CPPFLAGS += -DBH_DOUBLE
BIN      := bh_double
endif

SRCS := $(wildcard src/*.cpp) $(wildcard src/cpu/*.cpp)

ifeq ($(CUDA),1)
BUILD    := $(BUILD)-cuda
CPPFLAGS += -DBH_CUDA
CU_SRCS  := $(wildcard src/cuda/*.cu)
LDLIBS   += -L$(CUDA_HOME)/lib64 -lcudart
endif

OBJS := $(SRCS:%.cpp=$(BUILD)/%.o) $(CU_SRCS:%.cu=$(BUILD)/%.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all test clean
all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

test: $(BIN)
	BH=./$(BIN) bash tests/run_tests.sh

clean:
	rm -rf build bh bh_double

-include $(DEPS)
