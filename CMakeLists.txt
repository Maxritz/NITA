cmake_minimum_required(VERSION 3.20)
project(NITA C)  # NITA = NC Inference Tool for All

# ------------------------------------------------------------------
# Target architecture – AMD Zen3 AVX2 only
# ------------------------------------------------------------------
add_compile_options(-march=znver3 -mavx2)

# ------------------------------------------------------------------
# Quantisation feature toggles (CMake -D flags)
# ------------------------------------------------------------------
option(QUANTIZED_GEMV "Enable quantized GEMV base path" ON)
option(TERNARY "Enable ternary {-1,0,1} quantisation" OFF)
option(GPTQ "Enable 4‑bit grouped GPTQ with selective de‑quant" OFF)
option(AWQ "Enable activation‑aware weight quantisation (selective)" OFF)
option(BINARY_NET "Enable BitNet / binary quantisation" OFF)
option(BF16 "Enable brain‑float16 float matmul path" OFF)
option(FP16 "Enable IEEE float16 matmul path" OFF)
option(MXFP8 "Enable MXFP8 (forces compile‑time error – not on AVX2)" OFF)

# ------------------------------------------------------------------
# Derive a single QUANT_TYPE macro from the enabled flag.
# Exactly one should be ON; if none is ON, default to Q4_K_M.
# The logic: check each flag in priority order; the first TRUE wins.
# ------------------------------------------------------------------
if(TERNARY)
    set(QUANT_TYPE -DQUANT_TYPE=TERNARY)
elseif(GPTQ)
    set(QUANT_TYPE -DQUANT_TYPE=GPTQ)
elseif(AWQ)
    set(QUANT_TYPE -DQUANT_TYPE=AWQ)
elseif(BINARY_NET)
    set(QUANT_TYPE -DQUANT_TYPE=BINARY_NET)
elseif(BF16)
    set(QUANT_TYPE -DQUANT_TYPE=BF16)
elseif(FP16)
    set(QUANT_TYPE -DQUANT_TYPE=FP16)
elseif(MXFP8)
    set(QUANT_TYPE -DQUANT_TYPE=MXFP8)
else()
    # Default: Q4_K_M when no other flag is given
    set(QUANT_TYPE -DQUANT_TYPE=Q4_K_M)
endif()

# ------------------------------------------------------------------
# Force the preprocessor macros from CMake options
# ------------------------------------------------------------------
if(QUANTIZED_GEMV)
    add_compile_options(-DQUANTIZED_GEMV)
endif()
if(TERNARY)
    add_compile_options(-DTERNARY)
endif()
if(GPTQ)
    add_compile_options(-DGPTQ)
endif()
if(AWQ)
    add_compile_options(-DAWQ)
endif()
if(BINARY_NET)
    add_compile_options(-DBINARY_NET)
endif()
if(BF16)
    add_compile_options(-DBF16)
endif()
if(FP16)
    add_compile_options(-DFP16)
endif()
if(MXFP8)
    add_compile_options(-DMXFP8)
endif()

# Add the derived QUANT_TYPE macro last (overrides any previous)
add_compile_options(${QUANT_TYPE})

# ------------------------------------------------------------------
# Source files – the core kernel plus a minimal main
# ------------------------------------------------------------------
add_executable(minimal_engine src/q4_matmul.c src/main.c)

message(STATUS "Project: NITA (NC Inference Tool for All)")
message(STATUS "Engine flags:")
message(STATUS "  QUANTIZED_GEMV=${QUANTIZED_GEMV}")
message(STATUS "  TERNARY=${TERNARY}")
message(STATUS "  GPTQ=${GPTQ}")
message(STATUS "  AWQ=${AWQ}")
message(STATUS "  BINARY_NET=${BINARY_NET}")
message(STATUS "  BF16=${BF16}")
message(STATUS "  FP16=${FP16}")
message(STATUS "  MXFP8=${MXFP8}")
message(STATUS "  QUANT_TYPE=${QUANT_TYPE}")