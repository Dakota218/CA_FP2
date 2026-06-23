#pragma once

#ifndef HW_CONFIG_CHOICE
#define HW_CONFIG_CHOICE 2
#endif

#if HW_CONFIG_CHOICE == 0
#include "actual_hw.h"
using ActiveHWConfig = actual_hw::Config;
inline constexpr const char *kActiveHWName = "actual_hw";
#elif HW_CONFIG_CHOICE == 1
#include "baseline_hw.h"
using ActiveHWConfig = baseline_hw::Config;
inline constexpr const char *kActiveHWName = "baseline_hw";
#elif HW_CONFIG_CHOICE == 2
#include "student_hw.h"
using ActiveHWConfig = student_hw::Config;
inline constexpr const char *kActiveHWName = "student_hw";
#else
#error "Invalid HW_CONFIG_CHOICE. Use 0=actual, 1=baseline, 2=student."
#endif

static_assert(ActiveHWConfig::NUM_BUFS == 12,
              "HW config must keep NUM_BUFS = 12.");
