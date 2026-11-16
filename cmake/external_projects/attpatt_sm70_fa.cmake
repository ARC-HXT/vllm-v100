# V100 (sm70) D256 prefill attention: 1Cat-vLLM's exact Split-D kernel as its own extension vllm._attpatt_sm70_fa_C
# (AttPatt plan 2026-11-16 round 5, user "行吧，那就用1cat这个吧").
# Source: github.com/zhinianqin/flash-attention-v100 (BSD-3) at the commit 1Cat-vLLM v1.5.3 pins
# (cmake/external_projects/vllm_flash_attn.cmake:39 of that release), with its csrc/cutlass submodule, plus the five
# patches 1Cat applies in this order (cmake/patches/onecat/, copied from 1CatAI/1Cat-vLLM v1.5.3, identical on main
# ad3b337 2026-11-16). Measured on V100 before adoption: Qwen3.5-9B geometry (16 q / 4 KV heads, D256) M = L = 8192
# 12.01 ms vs 28.56 ms cutlass / 28.80 ms Triton, error = fp16 rounding (AttPatt onecat/pp1/kernel-onecat-d256.json).
# Deviation from the reference: 1Cat builds the whole patched tree as _vllm_fa2_C (replacing the vllm_flash_attn
# package); here only flash_fwd_d256_splitd_sm70.cu is compiled, into a separate extension, and vLLM's FA packages are
# not touched.
cuda_archs_loose_intersection(ATTPATT_SM70_FA_ARCHS "7.0" "${CUDA_ARCHS}")
if(NOT ATTPATT_SM70_FA_ARCHS)
  message(STATUS "[attpatt_sm70_fa] skipped: no sm70 in CUDA_ARCHS (${CUDA_ARCHS})")
  return()
endif()

include(FetchContent)
set(ATTPATT_SM70_FA_COMMIT c2eda5e6115b98c3ba4bfd181570668742eece22)
set(_attpatt_fa_patches pipeline splitkv3 k_pingpong gqa_arch splitkv_n)
set(_attpatt_fa_patch_cmd)
foreach(_p ${_attpatt_fa_patches})
  list(APPEND _attpatt_fa_patch_cmd COMMAND patch --batch --forward -p1 -l -i
       ${CMAKE_CURRENT_LIST_DIR}/../patches/onecat/sm70_flash_attn_d256_${_p}.patch)
endforeach()
find_package(Git REQUIRED)
find_program(ATTPATT_PATCH_EXECUTABLE patch REQUIRED)

set(_attpatt_fa_root "${FETCHCONTENT_BASE_DIR}")
if(NOT _attpatt_fa_root)
  set(_attpatt_fa_root "${CMAKE_BINARY_DIR}/_deps")
endif()
set(attpatt_sm70_fa_SOURCE_DIR "${_attpatt_fa_root}/attpatt_sm70_fa-src")
set(_attpatt_fa_stamp "${attpatt_sm70_fa_SOURCE_DIR}/.attpatt_patched_${ATTPATT_SM70_FA_COMMIT}")
if(NOT EXISTS "${_attpatt_fa_stamp}")
  # A pinned commit plus patches: always start from a clean checkout, so a rebuild never patches twice.
  FetchContent_Populate(
    attpatt_sm70_fa
    SUBBUILD_DIR "${_attpatt_fa_root}/attpatt_sm70_fa-subbuild"
    SOURCE_DIR "${attpatt_sm70_fa_SOURCE_DIR}"
    BINARY_DIR "${_attpatt_fa_root}/attpatt_sm70_fa-build"
    GIT_REPOSITORY https://github.com/zhinianqin/flash-attention-v100.git
    GIT_TAG ${ATTPATT_SM70_FA_COMMIT}
    GIT_SUBMODULES csrc/cutlass
    GIT_PROGRESS TRUE
    PATCH_COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> reset --hard ${ATTPATT_SM70_FA_COMMIT}
      COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> clean -fd
      COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR>/csrc/cutlass reset --hard
      COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR>/csrc/cutlass clean -fd
      ${_attpatt_fa_patch_cmd}
  )
  file(WRITE "${_attpatt_fa_stamp}" "patched\n")
endif()
message(STATUS "[attpatt_sm70_fa] patched flash-attention-v100 at ${attpatt_sm70_fa_SOURCE_DIR}")

set(ATTPATT_SM70_FA_SOURCES
  csrc/attpatt_v100/sm70_fa_bind.cpp
  ${attpatt_sm70_fa_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_d256_splitd_sm70.cu)
set_gencode_flags_for_srcs(
  SRCS "${ATTPATT_SM70_FA_SOURCES}"
  CUDA_ARCHS "${ATTPATT_SM70_FA_ARCHS}")

define_extension_target(
  _attpatt_sm70_fa_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${ATTPATT_SM70_FA_SOURCES}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES
    ${attpatt_sm70_fa_SOURCE_DIR}/csrc/flash_attn
    ${attpatt_sm70_fa_SOURCE_DIR}/csrc/flash_attn/src
    ${attpatt_sm70_fa_SOURCE_DIR}/csrc/common
    ${attpatt_sm70_fa_SOURCE_DIR}/csrc/cutlass/include
  USE_SABI 3
  WITH_SOABI)

# the flags and definitions of the fork's own FA2 target (its CMakeLists.txt at that commit)
target_compile_definitions(_attpatt_sm70_fa_C PRIVATE
  FLASHATTENTION_DISABLE_BACKWARD
  FLASHATTENTION_DISABLE_DROPOUT
  FLASHATTENTION_DISABLE_PYBIND
  CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)
set_property(SOURCE ${ATTPATT_SM70_FA_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -O3>)
