
cmake_minimum_required(VERSION 3.16)
project(SystemC_Tests CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# cmake -S /home/wilson/SystemC/test_tool -B /home/wilson/SystemC/test_tool/build -DFP1_GROUP_NUM=1
# cmake --build /home/wilson/SystemC/test_tool/build -j

# ===== SystemC 路徑（你的 conda 環境）=====
# 可用 -DSC_PREFIX=/path/override 覆寫
set(DEFAULT_SC_PREFIX "/opt/systemc")
if(DEFINED ENV{SC_PREFIX})
  set(DEFAULT_SC_PREFIX "$ENV{SC_PREFIX}")
endif()
set(SC_PREFIX "${DEFAULT_SC_PREFIX}" CACHE PATH "SystemC prefix")
set(SC_INC     "${SC_PREFIX}/include")
set(SC_LIB_DIR "${SC_PREFIX}/lib")
set(SC_LIB     "${SC_LIB_DIR}/libsystemc.so")

# 找 pthread（多數環境 SystemC 需要）
find_package(Threads REQUIRED)

# ===== 專案 include 路徑 =====
# 你的 headers 都在 ../include
include_directories(
  "${SC_INC}"
  "../include"
  "../third_party"   #  for nlohmann/json.hpp (third_party/nlohmann/json.hpp)
)

# ===== 可選：建 test_vector =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_vector.cpp")
add_executable(test_vector
  test_vector.cpp
)
target_compile_definitions(test_vector PRIVATE _FILE_OFFSET_BITS=64)
target_link_libraries(test_vector
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
# 設 RPATH，執行時可直接找到 libsystemc.so
set_target_properties(test_vector PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)
endif()

# # ===== 新增：建 test_dram =====
# add_executable(test_dram
#   test_dram.cpp
# )
# target_compile_definitions(test_dram PRIVATE _FILE_OFFSET_BITS=64)
# target_link_libraries(test_dram
#   PRIVATE
#     "${SC_LIB}"
#     Threads::Threads
#     m
# )
# set_target_properties(test_dram PROPERTIES
#   BUILD_RPATH   "${SC_LIB_DIR}"
#   INSTALL_RPATH "${SC_LIB_DIR}"
# )
# ===== 新增：建 test_array =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_array.cpp")
add_executable(test_array
  test_array.cpp
)
target_compile_definitions(test_array PRIVATE _FILE_OFFSET_BITS=64)
target_link_libraries(test_array
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
set_target_properties(test_array PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)
endif()

# ===== 新增：建 test_dma =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_dma.cpp")
add_executable(test_dma
  test_dma.cpp
)
target_compile_definitions(test_dma PRIVATE _FILE_OFFSET_BITS=64)
target_link_libraries(test_dma
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
set_target_properties(test_dma PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)
endif()

# ===== 新增：建 test_core_controller =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_core_controller.cpp")
add_executable(test_core_controller
  test_core_controller.cpp
)
target_compile_definitions(test_core_controller PRIVATE _FILE_OFFSET_BITS=64)
target_link_libraries(test_core_controller
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
set_target_properties(test_core_controller PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)
endif()

# # ===== 新增：建 test_NoC =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_NoC.cpp")
add_executable(test_NoC
  test_NoC.cpp
)
target_compile_definitions(test_NoC PRIVATE _FILE_OFFSET_BITS=64)
target_link_libraries(test_NoC
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
set_target_properties(test_NoC PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)
endif()

# # ===== 新增：建 test_memhub =====
# add_executable(test_memhub
#   test_memhub.cpp
# )
# target_compile_definitions(test_memhub PRIVATE _FILE_OFFSET_BITS=64)
# target_link_libraries(test_memhub
#   PRIVATE
#     "${SC_LIB}"
#     Threads::Threads
#     m
# )
# set_target_properties(test_memhub PROPERTIES
#   BUILD_RPATH   "${SC_LIB_DIR}"
#   INSTALL_RPATH "${SC_LIB_DIR}"
# )

# ===== 新增：建 test_Tcore =====
# add_executable(test_Tcore
#   test_Tcore.cpp
# )
# target_compile_definitions(test_Tcore PRIVATE _FILE_OFFSET_BITS=64)
# target_link_libraries(test_Tcore
#   PRIVATE
#     "${SC_LIB}"
#     Threads::Threads
#     m
# )
# set_target_properties(test_Tcore PROPERTIES
#   BUILD_RPATH   "${SC_LIB_DIR}"
#   INSTALL_RPATH "${SC_LIB_DIR}"
# )
# ===== 新增：建 test_one_Tcore =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_one_Tcore.cpp")
add_executable(test_one_Tcore
  test_one_Tcore.cpp
)
target_compile_definitions(test_one_Tcore PRIVATE _FILE_OFFSET_BITS=64)
target_link_libraries(test_one_Tcore
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
set_target_properties(test_one_Tcore PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)
endif()

# ===== 新增：建 main =====
add_executable(main
  main.cpp
)

set(FP1_GROUP_NUM 1 CACHE STRING "FP1 group number, e.g. 1 for G01")
if(FP1_GROUP_NUM LESS 1)
  message(FATAL_ERROR "FP1_GROUP_NUM must be >= 1")
endif()
set(HW_CONFIG_CHOICE 2 CACHE STRING "Hardware config choice: 0=actual, 1=baseline, 2=student")
target_compile_definitions(main PRIVATE
  _FILE_OFFSET_BITS=64
  FP1_GROUP_NUM=${FP1_GROUP_NUM}
  HW_CONFIG_CHOICE=${HW_CONFIG_CHOICE}
)
target_link_libraries(main
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
set_target_properties(main PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)

# # ===== 新增：建 test_Router =====
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_Router.cpp")
add_executable(test_Router
  test_Router.cpp
)
target_compile_definitions(test_Router PRIVATE _FILE_OFFSET_BITS=64)
target_link_libraries(test_Router
  PRIVATE
    "${SC_LIB}"
    Threads::Threads
    m
)
set_target_properties(test_Router PROPERTIES
  BUILD_RPATH   "${SC_LIB_DIR}"
  INSTALL_RPATH "${SC_LIB_DIR}"
)
endif()
# ===== 新增：建 test_Router_with_scoreboard =====
# add_executable(test_Router_with_scoreboard
#   test_Router_with_scoreboard.cpp
# )
# target_link_libraries(test_Router_with_scoreboard
#   PRIVATE
#     "${SC_LIB}"
#     Threads::Threads
#     m
# )
# set_target_properties(test_Router_with_scoreboard PROPERTIES
#   BUILD_RPATH   "${SC_LIB_DIR}"
#   INSTALL_RPATH "${SC_LIB_DIR}"
# )

