cmake_minimum_required(VERSION 3.16)
project(oceanbdx LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_MAIN "Build main controller (requires onnxruntime)" ON)
option(BUILD_TESTS "Build hardware debug test programs" ON)
# onnxruntime 安装路径, 例如: cmake -DONNXRUNTIME_ROOT=/opt/onnxruntime ..
set(ONNXRUNTIME_ROOT "" CACHE PATH "onnxruntime install prefix")

find_package(Threads REQUIRED)

# ---------------- yaml-cpp ----------------
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    find_library(YAML_CPP_LIB yaml-cpp)
    if(NOT YAML_CPP_LIB)
        message(FATAL_ERROR "yaml-cpp not found, install with: sudo apt install libyaml-cpp-dev")
    endif()
endif()

# ---------------- 宇树 M8010 串口 SDK (预编译库) ----------------
set(UNITREE_MOTOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/drivers/unitree_motor)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(UNITREE_MOTOR_LIB ${UNITREE_MOTOR_DIR}/lib/libUnitreeMotorSDK_Arm64.so)
else()
    set(UNITREE_MOTOR_LIB ${UNITREE_MOTOR_DIR}/lib/libUnitreeMotorSDK_Linux64.so)
endif()

# ---------------- 飞特舵机驱动 ----------------
file(GLOB FEETECH_SRC drivers/feetech/*.cpp)
add_library(feetech STATIC ${FEETECH_SRC})
target_include_directories(feetech PUBLIC drivers/feetech)

# ---------------- YIS320 IMU 协议解析 ----------------
add_library(yis_imu STATIC drivers/yis_imu/analysis_data.c)
target_include_directories(yis_imu PUBLIC drivers/yis_imu)

# ---------------- oceanbdx 核心库 ----------------
add_library(oceanbdx_core STATIC
    src/config.cpp
    src/leg_driver.cpp
    src/imu_driver.cpp
    src/neck_driver.cpp
    src/gamepad_driver.cpp
    src/battery_driver.cpp
    src/fsm.cpp)
target_include_directories(oceanbdx_core PUBLIC
    include
    ${UNITREE_MOTOR_DIR}/include)
target_link_libraries(oceanbdx_core PUBLIC
    ${UNITREE_MOTOR_LIB}
    feetech
    yis_imu
    Threads::Threads)
if(yaml-cpp_FOUND)
    target_link_libraries(oceanbdx_core PUBLIC yaml-cpp)
else()
    target_link_libraries(oceanbdx_core PUBLIC ${YAML_CPP_LIB})
endif()

# ---------------- 主控制程序 (需要 onnxruntime) ----------------
if(BUILD_MAIN)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime
              /usr/local/include/onnxruntime /usr/include/onnxruntime)
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}/lib /usr/local/lib)

    if(ONNXRUNTIME_INCLUDE AND ONNXRUNTIME_LIB)
        add_library(oceanbdx_policy STATIC src/policy.cpp)
        target_include_directories(oceanbdx_policy PUBLIC include ${ONNXRUNTIME_INCLUDE})
        target_link_libraries(oceanbdx_policy PUBLIC ${ONNXRUNTIME_LIB} oceanbdx_core)

        add_executable(oceanbdx_run src/main.cpp)
        target_link_libraries(oceanbdx_run oceanbdx_core oceanbdx_policy)
    else()
        message(WARNING "onnxruntime not found, skip main controller. "
            "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime (download from "
            "https://github.com/microsoft/onnxruntime/releases, aarch64 build for Jetson)")
    endif()
endif()

# ---------------- 分步调试测试程序 ----------------
if(BUILD_TESTS)
    add_executable(test_leg_motor tests/test_leg_motor.cpp)
    target_include_directories(test_leg_motor PRIVATE ${UNITREE_MOTOR_DIR}/include)
    target_link_libraries(test_leg_motor ${UNITREE_MOTOR_LIB} Threads::Threads)

    add_executable(test_imu tests/test_imu.cpp)
    target_link_libraries(test_imu oceanbdx_core)

    add_executable(test_neck tests/test_neck.cpp)
    target_link_libraries(test_neck oceanbdx_core)

    add_executable(test_calibration tests/test_calibration.cpp)
    target_link_libraries(test_calibration oceanbdx_core)

    add_executable(test_gamepad tests/test_gamepad.cpp)
    target_link_libraries(test_gamepad oceanbdx_core)

    add_executable(test_battery tests/test_battery.cpp)
    target_link_libraries(test_battery oceanbdx_core)
endif()
