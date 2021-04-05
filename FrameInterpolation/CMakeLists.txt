macro(addShaderHeader SHADER_SRC)
  get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
  set(SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.comp.hex.h)

  add_custom_command(
    OUTPUT ${SHADER_COMP_HEADER}
    COMMAND ${CMAKE_COMMAND}
    -DSHADER_SRC=${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC}
    -DSHADER_COMP_HEADER=${SHADER_COMP_HEADER} -P
    "${CMAKE_MODULE_PATH}/generate_shader_comp_header.cmake"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC}
    COMMENT "Preprocessing shader source ${SHADER_SRC_NAME_WE}.comp"
    VERBATIM
  )
  set_source_files_properties(${SHADER_COMP_HEADER} PROPERTIES GENERATED TRUE)

  list(APPEND SHADER_SPV_HEX_FILES ${SHADER_COMP_HEADER})
endmacro()

addshaderheader(shaders/rife_preproc.comp)
addshaderheader(shaders/rife_postproc.comp)
addshaderheader(shaders/rife_preproc_tta.comp)
addshaderheader(shaders/rife_postproc_tta.comp)
addshaderheader(shaders/rife_flow_tta_avg.comp)
addshaderheader(shaders/rife_v2_flow_tta_avg.comp)
addshaderheader(shaders/warp.comp)
addshaderheader(shaders/warp_pack4.comp)
addshaderheader(shaders/warp_pack8.comp)

# include directories
set(NCNN_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/dependencies/install/ncnn/include/ncnn)
include_directories(${NCNN_INCLUDE_DIRS})
include_directories(.)
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}) # generated headers in build/

# add custom target
add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

# library files
file(GLOB includes include/*.h)
file(GLOB sources src/*.cpp)

# create library
set(target rife-ncnn-vulkan)
add_library(
  ${target} SHARED
  ${sources}
  ${includes}
  )
add_dependencies(${target} generate-spirv)
target_link_libraries(${target} ncnn ${Vulkan_LIBRARY})

# add tests
add_executable(FrameInterpolationTest test/test.cpp)
if(X11_FOUND)
  target_link_libraries(FrameInterpolationTest ${target} X11)
endif()
