# ============================================================
# Makefile for Qwen LLM Inference Engine (Cross-Platform)
# ============================================================
# Usage:
#   make              # Build for current platform (auto-detect GPU arch)
#   make SM=89        # Override GPU compute capability (e.g. sm_89 for RTX 4060 Ti)
#   make clean        # Remove build artifacts
#   make run          # Build and run with default model weights
#
# Platform support:
#   Linux:   nvcc + g++ (auto-detected)
#   Windows: nvcc + MSVC (set MSVC_PATH if not in standard location)
# ============================================================

# --- OS detection ---
ifeq ($(OS),Windows_NT)
    HOST_OS := Windows
    EXE_SUFFIX := .exe
    RM := del /Q /F
    RMDIR := rmdir /S /Q
else
    HOST_OS := Linux
    EXE_SUFFIX :=
    RM := rm -f
    RMDIR := rm -rf
endif

# --- Auto-detect GPU compute capability ---
ifeq ($(SM),)
    ifeq ($(HOST_OS),Windows)
        # Check via nvidia-smi (returns e.g. "8.9", "8.6")
        SM_RAW := $(shell where nvidia-smi >nul 2>&1 && nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>nul || echo 8.9)
    else
        SM_RAW := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1-2)
        ifeq ($(SM_RAW),)
            SM_RAW := 8.9
        endif
    endif
    SM := $(subst .,,$(SM_RAW))
endif

ARCH := -arch=sm_$(SM)

# --- Compiler settings ---
NVCC := nvcc

ifeq ($(HOST_OS),Windows)
    # MSVC host compiler (adjust path as needed)
    MSVC_PATH ?= C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
    CCBIN := -ccbin "$(MSVC_PATH)"
else
    # Linux: g++ is auto-detected by nvcc
    CCBIN :=
endif

NVCC_FLAGS := $(ARCH) $(CCBIN) -O3 -lcublas -std=c++17 --use_fast_math
NVCC_FLAGS += --expt-relaxed-constexpr

TARGET := qwen_infer$(EXE_SUFFIX)
SRC := qwen_infer.cu

# Header dependencies for automatic rebuild
KERNEL_DEPS := $(wildcard ../kernels/*.cuh ../kernels/*.cu)

.PHONY: all clean run info

all: $(TARGET)

$(TARGET): $(SRC) $(KERNEL_DEPS)
	@echo "=== Building Qwen Inference Engine ==="
	@echo "  Platform: $(HOST_OS)"
	@echo "  GPU Arch: sm_$(SM)"
	@echo "  Target:   $(TARGET)"
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	@echo "Cleaning build artifacts..."
	$(RM) qwen_infer$(EXE_SUFFIX) qwen_infer_debug$(EXE_SUFFIX) 2>/dev/null; true
	@echo "Clean complete."

run: $(TARGET)
	@echo "Running inference engine..."
	./$(TARGET) ../model_weights/qwen_1.8b.bin

info:
	@echo "=== Build Environment Info ==="
	@echo "  Host OS:      $(HOST_OS)"
	@echo "  GPU Arch:     sm_$(SM)"
	@echo "  Executable:   $(TARGET)"
	@echo "  NVCC:         $(shell $(NVCC) --version 2>/dev/null | head -1 || echo 'not found')"
	@echo "  Model path:   ../model_weights/qwen_1.8b.bin"
