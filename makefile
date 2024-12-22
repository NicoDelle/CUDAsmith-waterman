# Compiler and flags
NVCC = nvcc
CXX = g++
CXXFLAGS = -I./src -g
NVCCFLAGS = -I./src -g

# Directories
SRC_DIR = src
LIB_DIR = lib

# Source files
SEQ_SRC = $(SRC_DIR)/smithWatermanSeq.cpp
PAR_SRC = $(SRC_DIR)/smithWatermanPar.cu

# Object files
SEQ_OBJ = $(LIB_DIR)/smithWatermanSeq.o
PAR_OBJ = $(LIB_DIR)/smithWatermanPar.o

# Static libraries
SEQ_LIB = $(LIB_DIR)/libsmithWatermanSeq.a
PAR_LIB = $(LIB_DIR)/libsmithWatermanPar.a

# Default source file to compile
MAIN_SRC ?= main.cpp

# Executable
EXEC = out
DEBUG_EXE = swdebug

# Targets
all: $(EXEC)

$(SEQ_LIB): $(SEQ_OBJ)
	ar rcs $@ $^

$(PAR_LIB): $(PAR_OBJ)
	ar rcs $@ $^

$(SEQ_OBJ): $(SEQ_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(PAR_OBJ): $(PAR_SRC)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXEC): $(MAIN_SRC) $(SEQ_LIB) $(PAR_LIB)
	$(NVCC) $(CXXFLAGS) -L$(LIB_DIR) -lsmithWatermanSeq -lsmithWatermanPar $< -o $@

$(DEBUG_EXEC): $(MAIN_SRC) $(SEQ_LIB) $(PAR_LIB)
	$(NVCC) $(CXXFLAGS) -L$(LIB_DIR) -lsmithWatermanSeq -lsmithWatermanPar $< -o $@ -g

clean:
	rm -f $(SEQ_OBJ) $(PAR_OBJ) $(EXEC)

.PHONY: all clean