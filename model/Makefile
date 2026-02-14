# ----- toolchain -----
CXX      := /usr/bin/g++
CXXFLAGS := -std=c++17 -O3 -march=native -DNDEBUG -Wall
LDFLAGS  := -pthread

# ----- robust paths (relative to this Makefile's location) -----
# Absolute directory containing this Makefile
MKFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

SRC := $(MKFILE_DIR)/tussock_model.cpp
HDR := $(MKFILE_DIR)/tussock_model.h

# Put the binary in the project root (one level above model/)
OUT := $(MKFILE_DIR)/tussock_model

.PHONY: all clean
all: $(OUT)

$(OUT): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	$(RM) -f $(OUT)
