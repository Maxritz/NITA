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
# Optional: the reference logits driver
#
# tools/llama_logits.c prints llama.cpp's own logits, which is what the logits
# in the README were checked against. It links the PrismML fork, which is not
# part of this repository, so it is built only when a checkout is present.
# "Configured" means build-mingw exists: that step is the fork's own advice
# (see README) and involves a local patch, so it stays a manual step.
# ------------------------------------------------------------------
set(NITA_FORK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/prism-llama" CACHE PATH
    "PrismML llama.cpp checkout with a configured build-mingw directory")

set(NITA_FORK_DRIVER "")
set(_fork_libs
    "${NITA_FORK_DIR}/build-mingw/src/libllama.a"
    "${NITA_FORK_DIR}/build-mingw/ggml/src/ggml.a"
    "${NITA_FORK_DIR}/build-mingw/ggml/src/ggml-cpu.a"
    "${NITA_FORK_DIR}/build-mingw/ggml/src/ggml-base.a"
    "${NITA_FORK_DIR}/build-mingw/vendor/hash/libvendor-hash.a")

if(NITA_FORK_DIR AND EXISTS "${NITA_FORK_DIR}/include/llama.h")
    set(_fork_missing "")
    foreach(_lib ${_fork_libs})
        if(NOT EXISTS "${_lib}")
            list(APPEND _fork_missing ${_lib})
        endif()
    endforeach()

    if(_fork_missing)
        message(STATUS "fork logits driver: skipped, missing:")
        foreach(_lib ${_fork_missing})
            message(STATUS "  ${_lib}")
        endforeach()
    else()
        add_executable(fork_logits tools/llama_logits.c)
        target_include_directories(fork_logits PRIVATE
            "${NITA_FORK_DIR}/include"
            "${NITA_FORK_DIR}/ggml/include")
        if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
            # Opt this one target out of the project's -march=znver3/-mavx2.
            # With AVX enabled GCC kept the driver's struct arguments and
            # buffers as 32-byte stack slots addressed with vmovdqa, which
            # demands 32-byte alignment; Windows only guarantees 16 at main's
            # entry, so the first such store segfaulted whenever ASLR placed
            # the stack at 16 mod 32 - measured 7/20 runs with AVX against
            # 0/24 without, and the faulting instruction was
            # vmovdqa %ymm3,0x150(%rsp) in main. The driver only parses argv
            # and prints numbers, so baseline x86-64 costs it nothing, and the
            # fork's own AVX code lives in libraries built separately.
            target_compile_options(fork_logits PRIVATE -mno-avx -mno-avx2)
        endif()
        target_link_libraries(fork_logits PRIVATE ${_fork_libs})
        if(WIN32)
            target_link_libraries(fork_logits PRIVATE
                ws2_32 ole32 uuid advapi32 shlwapi)
        endif()
        if(MINGW)
            # llama.cpp is C++, this driver is C: gcc would not pull the C++
            # runtime in on its own.
            target_link_libraries(fork_logits PRIVATE stdc++)
            # The toolchain's bin is not necessarily on PATH where CTest starts
            # the process - a missing runtime DLL kills it with no output at
            # all - so put the DLLs next to the executable.
            # The DLLs sit in the toolchain's bin, which -print-file-name does
            # not search (it only reports its library directories).
            get_filename_component(_mingw_bin "${CMAKE_C_COMPILER}" DIRECTORY)
            foreach(_dll libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll)
                if(EXISTS "${_mingw_bin}/${_dll}")
                    add_custom_command(TARGET fork_logits POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${_mingw_bin}/${_dll}"
                                "$<TARGET_FILE_DIR:fork_logits>")
                endif()
            endforeach()
        endif()
        set(NITA_FORK_DRIVER $<TARGET_FILE:fork_logits>)
        message(STATUS "fork logits driver: fork_logits, linking ${NITA_FORK_DIR}")
    endif()
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

# Comparing against the reference runtime needs two things this repository does
# not build: the PrismML llama.cpp and the driver linked against it, which is
# tools/llama_logits.c (see README, "The reference harness"). Name the driver
# with -DNITA_FORK_LOGITS and the check joins the suite; the script exits 77
# when a piece is missing, which CTest reports as a skip rather than a pass.
set(NITA_FORK_LOGITS "" CACHE FILEPATH
    "logits driver linked against the PrismML fork; empty to use fork_logits")

find_package(Python3 COMPONENTS Interpreter QUIET)

# A driver named by hand wins over the one built here, so a checkout that lives
# somewhere else can still be used.
set(_fork_test_driver "${NITA_FORK_LOGITS}")
if(NOT _fork_test_driver)
    set(_fork_test_driver "${NITA_FORK_DRIVER}")
endif()

if(NITA_TEST_MODEL AND _fork_test_driver AND Python3_Interpreter_FOUND)
    add_test(NAME fork_match
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_fork_match.py
                     --engine $<TARGET_FILE:nita>
                     --fork ${_fork_test_driver}
                     --model ${NITA_TEST_MODEL})
    set_tests_properties(fork_match PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)
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
