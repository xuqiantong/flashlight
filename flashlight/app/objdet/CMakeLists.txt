cmake_minimum_required(VERSION 3.10)

add_library(
  flashlight-app-objdet
  ""
  )

target_link_libraries(
  flashlight-app-objdet
  PUBLIC
  ${gflags_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-objdet
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

# Dataset
include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)

# Models
include(${CMAKE_CURRENT_LIST_DIR}/models/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# NN
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)

add_executable(
  fl_img_imagenet_resnet50_backbone
  ${CMAKE_CURRENT_LIST_DIR}/examples/ImageNetResnet50Backbone.cpp)

target_link_libraries(
  fl_img_imagenet_resnet50_backbone
  flashlight-app-imgclass
  flashlight-app-objdet)

add_executable(fl_img_coco_detr ${CMAKE_CURRENT_LIST_DIR}/examples/CocoDetr.cpp)
target_link_libraries(fl_img_coco_detr flashlight-app-objdet)

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif()
