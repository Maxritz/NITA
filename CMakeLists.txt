cmake_minimum_required(VERSION 3.20)
project(NITA C)  # NITA = NC Inference Tool for All

# ------------------------------------------------------------------
# Default to an optimised build. "-march=..." alone leaves GCC at -O0,
# which makes the scalar dequant/matvec kernels an order of magnitude
# slower than they should be.
# ------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

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
option(DEBUG_BREADCRUMBS "Enable breadcrumb tracing to stderr" OFF)
option(SHERLOCK_LITE "Enable sherlock‑lite trace to stdout" OFF)

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
if(DEBUG_BREADCRUMBS)
    add_compile_options(-DDEBUG_BREADCRUMBS)
endif()
if(SHERLOCK_LITE)
    add_compile_options(-DSHERLOCK_LITE)
endif()

# Add the derived QUANT_TYPE macro last (overrides any previous)
add_compile_options(${QUANT_TYPE})

# ------------------------------------------------------------------
# Source files – everything but main(), so the tests can link the same
# objects instead of repeating this list (and its flags) per test.
# ------------------------------------------------------------------
add_library(nita_core STATIC
    src/util.c
    src/gguf.c
    src/json.c
    src/tokenizer.c
    src/chat.c
    src/http.c
    src/dequant.c
    src/quant.c
    src/transformer.c
    src/qwen35.c
    src/hadamard.c
    src/q4_matmul.c
)

target_include_directories(nita_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# libm for expf/cosf/powf/sqrtf.
if(NOT MSVC)
    target_link_libraries(nita_core PUBLIC m)
endif()

# Winsock is required by the HTTP server on Windows.
if(WIN32)
    target_link_libraries(nita_core PUBLIC ws2_32)
endif()

add_executable(nita src/main.c)
target_link_libraries(nita PRIVATE nita_core)

# Keep the historical binary name working for existing scripts.
if(WIN32)
    add_custom_command(TARGET nita POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nita> $<TARGET_FILE_DIR:nita>/minimal_engine.exe)
else()
    add_custom_command(TARGET nita POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nita> $<TARGET_FILE_DIR:nita>/minimal_engine)
endif()

# ------------------------------------------------------------------
# Tests
#
#   cmake --build build && ctest --test-dir build
#
# hadamard and q35_heads need no checkpoint, so they always register. The ones
# that read a model file are built unconditionally and registered only when one
# is supplied with -DNITA_TEST_MODEL=<path>. tests/kernel_test.c and
# tests/model_test.c are left out: neither has a main(), and the first includes
# a header the tree no longer has.
# ------------------------------------------------------------------
include(CTest)
set(NITA_TEST_MODEL "" CACHE FILEPATH
    "Bonsai checkpoint for the q35_probe test; empty to skip it")

add_executable(hadamard_test tests/hadamard_test.c)
target_link_libraries(hadamard_test PRIVATE nita_core)
add_test(NAME hadamard COMMAND hadamard_test)

add_executable(q35_heads_test tests/q35_heads_test.c)
target_link_libraries(q35_heads_test PRIVATE nita_core)
add_test(NAME q35_heads COMMAND q35_heads_test)

add_executable(q35_probe tests/q35_probe.c)
target_link_libraries(q35_probe PRIVATE nita_core)

add_executable(quant_dot_test tests/quant_dot_test.c)
target_link_libraries(quant_dot_test PRIVATE nita_core)

if(NITA_TEST_MODEL)
    add_test(NAME q35_probe  COMMAND q35_probe  ${NITA_TEST_MODEL})
    # --all walks the tensor table and tests every quantised type the file
    # stores, so this entry does not have to name a type id that could drift.
    add_test(NAME quant_dot  COMMAND quant_dot_test --all ${NITA_TEST_MODEL})
endif()

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
