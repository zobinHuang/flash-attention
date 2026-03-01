// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

#ifndef FLASHATTENTION_DISABLE_HDIM32
// branch for headdim 32
#define HDIM_BRANCH_32(HEADDIM, ...) else if (HEADDIM <= 32) { constexpr static int kHeadDim = 32; return __VA_ARGS__(); }
#else
// disabled headdim 32
#define HDIM_BRANCH_32(HEADDIM, ...)
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM64
// branch for headdim 64
#define HDIM_BRANCH_64(HEADDIM, ...) else if (HEADDIM <= 64) { constexpr static int kHeadDim = 64; return __VA_ARGS__(); }
#else
// disabled headdim 64
#define HDIM_BRANCH_64(HEADDIM, ...)
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM96
// branch for headdim 96
#define HDIM_BRANCH_96(HEADDIM, ...) else if (HEADDIM <= 96) { constexpr static int kHeadDim = 96; return __VA_ARGS__(); }
#else
// disabled headdim 96
#define HDIM_BRANCH_96(HEADDIM, ...)
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM128
// branch for headdim 128
#define HDIM_BRANCH_128(HEADDIM, ...) else if (HEADDIM <= 128) { constexpr static int kHeadDim = 128; return __VA_ARGS__(); }
#else
// disabled headdim 128
#define HDIM_BRANCH_128(HEADDIM, ...)
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM192
// branch for headdim 192
#define HDIM_BRANCH_192(HEADDIM, ...) else if (HEADDIM <= 192) { constexpr static int kHeadDim = 192; return __VA_ARGS__(); }
#else
// disabled headdim 192
#define HDIM_BRANCH_192(HEADDIM, ...)
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM256
// branch for headdim 256
#define HDIM_BRANCH_256(HEADDIM, ...) else if (HEADDIM <= 256) { constexpr static int kHeadDim = 256; return __VA_ARGS__(); }
#else
// disabled headdim 256
#define HDIM_BRANCH_256(HEADDIM, ...)
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)   \
[&] {                                    \
    if (false) {}                          \
    HDIM_BRANCH_32(HEADDIM, __VA_ARGS__)   \
    HDIM_BRANCH_64(HEADDIM, __VA_ARGS__)   \
    HDIM_BRANCH_96(HEADDIM, __VA_ARGS__)   \
    HDIM_BRANCH_128(HEADDIM, __VA_ARGS__)  \
    HDIM_BRANCH_192(HEADDIM, __VA_ARGS__)  \
    HDIM_BRANCH_256(HEADDIM, __VA_ARGS__)  \
    else {                                 \
    TORCH_CHECK(false, "unsupported headdim. it might have been disabled during compilation."); \
    }                                      \
}()
