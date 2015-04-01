cmake_minimum_required(VERSION 2.6)

#project(flandmark)

add_subdirectory(libflandmark)

find_package( OpenCV REQUIRED )

if(CMAKE_COMPILER_IS_GNUCC)
    set_target_properties(flandmark_static PROPERTIES COMPILE_FLAGS -fPIC)
endif(CMAKE_COMPILER_IS_GNUCC)

include_directories("${PROJECT_SOURCE_DIR}/libflandmark")
include_directories(${OPENCV_INCLUDE_DIR})

#set(CV_LIBS_2 opencv_core opencv_objdetect)

set(face_landmarks_detection_srcs face_landmarks_detection.cpp)
add_executable(face_landmarks_detection ${face_landmarks_detection_srcs})
target_link_libraries(face_landmarks_detection flandmark_static opencv_highgui)


