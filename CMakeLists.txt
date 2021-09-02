
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

project(celiu_optical_flow
	DESCRIPTION "Python bindings for celiu cpp optical flow code."
	HOMEPAGE_URL "https://people.csail.mit.edu/celiu/OpticalFlow/"
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

find_package(SWIG REQUIRED COMPONENTS python)
find_package( OpenCV REQUIRED )
find_package(PythonLibs REQUIRED )

include(${SWIG_USE_FILE})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cflow/cpp/)

set(CMAKE_SWIG_FLAGS "")
set(SWIG_OUTFILE_DIR ${CMAKE_BINARY_DIR})
set(HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cflow/cpp/)
set(PYLIB_DIR ${CMAKE_BINARY_DIR}/../pylib/)

# 
# -- Gaussian Pyramid --
#

# -- cpp --
add_library(GaussianPyramidLib SHARED ${CMAKE_BINARY_DIR}/../cflow/cpp/GaussianPyramid.cpp)

# -- swig --
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/GaussianPyramid.i
  PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/GaussianPyramid.i
  PROPERTY SWIG_MODULE_NAME gpyramid)
set_property(SOURCE gpyramid PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
swig_add_library(gpyramid LANGUAGE python OUTPUT_DIR ${SWIG_OUTFILE_DIR} SOURCES ${CMAKE_BINARY_DIR}/../cflow/python/GaussianPyramid.i ${HEADER_DIR}/GaussianPyramid.h)
target_include_directories(gpyramid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}../cflow/cpp/)
swig_link_libraries(gpyramid GaussianPyramidLib)
swig_link_libraries(gpyramid ${OpenCV_LIBS})

# 
# -- Optical Flow --
#

# -- cpp --
add_library(OpticalFlowLib SHARED ${CMAKE_BINARY_DIR}/../cflow/cpp/OpticalFlow.cpp)

# -- swig --
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/OpticalFlow.i
  PROPERTY CPLUSPLUS ON)
set_property(SOURCE ${CMAKE_BINARY_DIR}/../cflow/python/OpticalFlow.i
  PROPERTY SWIG_MODULE_NAME oflow)
set_property(SOURCE oflow PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
swig_add_library(oflow LANGUAGE python OUTPUT_DIR ${SWIG_OUTFILE_DIR} SOURCES ${CMAKE_BINARY_DIR}/../cflow/python/OpticalFlow.i ${HEADER_DIR}/OpticalFlow.h)
target_include_directories(oflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}../cflow/cpp/)
swig_link_libraries(oflow OpticalFlowLib)
swig_link_libraries(oflow GaussianPyramidLib)
swig_link_libraries(oflow ${OpenCV_LIBS})


# 
# -- Copy "so" files to install location --
#

macro(copy_to_pylib target file)
  add_custom_command(
    TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_BINARY_DIR}/${file}
    ${PYLIB_DIR}/${file}
  )
endmacro()

copy_to_pylib(gpyramid _gpyramid.so)
copy_to_pylib(gpyramid gpyramid.py)
copy_to_pylib(GaussianPyramidLib libGaussianPyramidLib.so)
copy_to_pylib(OpticalFlowLib libOpticalFlowLib.so)
copy_to_pylib(oflow _oflow.so)
copy_to_pylib(oflow oflow.py)



