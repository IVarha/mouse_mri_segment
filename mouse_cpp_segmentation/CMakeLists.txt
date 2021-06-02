 cmake_minimum_required(VERSION 3.10)
project(mouse_segmentation_cpp)
 set(EXEC_NAME mouse_segmentation_cpp)
# list(APPEND CMAKE_PREFIX_PATH "/home/varga/libs/bay_segm/spdlog/build2/lib/cmake/spdlog")
set(CMAKE_CXX_STANDARD 14)
add_compile_options("-fPIC")

 find_package(ZLIB REQUIRED)
 find_package(Armadillo REQUIRED)


 add_subdirectory(external/pybind11-2.6)
include_directories(pybind11_INCLUDE_DIRS)


# include_directories(utils)
# include_directories(segmentation)
# include_directories(py_wrappers)
# include_directories(external/spdlog-1.8.2/include)
 include_directories(${ARMADILLO_INCLUDE_DIRS})
# add_library(Segmentation
#         segmentation/Surface.cpp segmentation/Surface.h
#         segmentation/NiftiImage.cpp segmentation/NiftiImage.h segmentation/Point.cpp segmentation/Point.h
#         segmentation/triangleintersects.h)


# add_library( Utils
#         utils/CLIParser.cpp utils/CLIParser.h)
# set(SOURCES utils/CLIParser.cpp segmentation/Surface.cpp
#         segmentation/NiftiImage.cpp segmentation/NiftiImage.h segmentation/Point.cpp )

#add_executable(mouse_segmentation_cpp ${SOURCES})
 pybind11_add_module(Mouse_C pybind_main.cpp)



# target_link_libraries(mouse_segmentation_cpp PRIVATE ${VTK_LIBRARIES} ZLIB::ZLIB  ${ARMADILLO_LIBRARIES}  )
# target_link_libraries(Segmentation PUBLIC ${VTK_LIBRARIES} ZLIB::ZLIB  ${ARMADILLO_LIBRARIES} )
target_link_libraries(Mouse_C PUBLIC ${VTK_LIBRARIES} ZLIB::ZLIB  ${ARMADILLO_LIBRARIES} )
 # vtk_module_autoinit is needed
