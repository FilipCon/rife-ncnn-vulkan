# This CMake file implements the super-build procedure to download, configure,
# compile and install all dependencies.

project(dependencies)
# We require a version of CMake that supports Visual Studio 2015.
cmake_minimum_required(VERSION 3.2)

include(ExternalProject)
include(CMakeParseArguments)

# Set the default for CMAKE_INSTALL_PREFIX.
function(SetDefaultCMakeInstallPrefix)
  get_filename_component(BASE_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
  # Move one directory up to the folder
  get_filename_component(BASE_DIR ${BASE_DIR} DIRECTORY)
  # Default install prefix for dependencies. If user changes
  # CMAKE_INSTALL_PREFIX, this directory will be removed.
  set(
    DEFAULT_CMAKE_INSTALL_PREFIX
    ${BASE_DIR}/install
    CACHE
    INTERNAL
    "Default CMAKE_INSTALL_PREFIX for dependencies."
  )

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(
      CMAKE_INSTALL_PREFIX
      ${DEFAULT_CMAKE_INSTALL_PREFIX}
      CACHE
      PATH
      "Directory to install binaries of dependencies."
      FORCE
    )
  endif()
endfunction()

# CMake doesn't clear prefix directories when user changes it.
# Remove it to avoid confusion.
function(RemoveDefaultInstallDirIfEmpty DIR)
  file(GLOB CONTENTS ${DIR}/*)
  if(NOT CONTENTS)
    file(REMOVE_RECURSE ${DIR})
  endif()
endfunction()

# Set the default for CMAKE_BUILD_TYPE.
function(SetDefaultCMakeBuildType)
  # CMAKE_BUILD_TYPE is only applicable for single configuration generators.
  if(NOT CMAKE_CONFIGURATION_TYPES)
    set(DOCSTRING "Build type to use for dependencies. Possible values --")
    set(DOCSTRING "${DOCSTRING} Debug, Release, RelWithDebInfo,")
    set(DOCSTRING "${DOCSTRING} MinSizeRel.")

    set(
      CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE
      STRING
      ${DOCSTRING}
    )
  endif()
endfunction()


# Add a dependency. Arguments:
#   NAME       -- (Required) Name of the project.
#   DEFAULT    -- (Required) Default value for SUPERBUILD_${NAME} variable.
#   URL        -- (Required) URL for a zip or tar.gz file of the source code.
#   GIT_URL    -- (Required) git repository to download the sources from.
#   GIT_TAG    -- (Required) git tag to checkout before commencing build.
#   DEPENDS    -- (Optional) Other projects this project depends on.
#   CMAKE_ARGS -- (Optional) A CMake list of arguments to be passed to CMake
#                 while building the project.
# You must provide either URL or GIT_URL and GIT_TAG, but not all 3.
function(AddDependency)
  set(onevalueargs NAME DEFAULT URL GIT_URL GIT_TAG)
  set(multiValueArgs DEPENDS CMAKE_ARGS)
  cmake_parse_arguments(DEP "" "${onevalueargs}" "${multiValueArgs}" ${ARGN})

  # Check for presence of required arguments.
  if(NOT (DEP_NAME AND ((DEP_GIT_URL AND DEP_GIT_TAG) OR DEP_URL)))
    set(MSG "One or more required arguments are missing. Please check the ")
    set(MSG "${MSG}AddDependency() call.")
    message(FATAL_ERROR ${MSG})
  endif()

  # Add a cache entry providing option for user to use (or not) superbuild.
  set(
    SUPERBUILD_${DEP_NAME} ${DEP_DEFAULT} CACHE BOOL
    "Automatically download, configure, build and install ${DEP_NAME}"
  )

  if(SUPERBUILD_${DEP_NAME})
    set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/${DEP_NAME})
    set(BINARY_DIR ${CMAKE_BINARY_DIR}/${DEP_NAME})
    set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${DEP_NAME})

    set(DEFAULT_INSTALL_DIR ${DEFAULT_CMAKE_INSTALL_PREFIX}/${DEP_NAME})
    removedefaultinstalldirifempty(${DEFAULT_INSTALL_DIR})

    set(CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR})
    if(NOT CMAKE_CONFIGURATION_TYPES)
      list(
        APPEND CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      )
    endif()
    list(APPEND CMAKE_ARGS ${DEP_CMAKE_ARGS})

    # Forward cmake arguments to dependencies.
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_CXX_FLAGS_MINSIZEREL:STRING=${CMAKE_CXX_FLAGS_MINSIZEREL}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_C_FLAGS_MINSIZEREL:STRING=${CMAKE_C_FLAGS_MINSIZEREL}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
    )
    list(
      APPEND CMAKE_ARGS
      -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}
    )
    if(APPLE)
      list(
        APPEND CMAKE_ARGS
        -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
      )
    endif()
    if(SWIG_EXECUTABLE)
      list(APPEND CMAKE_ARGS -DSWIG_EXECUTABLE:FILEPATH=${SWIG_EXECUTABLE})
    endif()

    if(DEP_GIT_URL)
      externalproject_add(
        ${DEP_NAME}
        DEPENDS ${DEP_DEPENDS}
        TMP_DIR ${BINARY_DIR}/tmp
        STAMP_DIR ${BINARY_DIR}/stamp
        GIT_REPOSITORY ${DEP_GIT_URL}
        GIT_TAG ${DEP_GIT_TAG}
        SOURCE_DIR ${SOURCE_DIR}
        CMAKE_CACHE_ARGS ${CMAKE_ARGS}
        BINARY_DIR ${BINARY_DIR}/build
        INSTALL_DIR ${INSTALL_DIR}
      )
    else()
      externalproject_add(
        ${DEP_NAME}
        DEPENDS ${DEP_DEPENDS}
        TMP_DIR ${BINARY_DIR}/tmp
        STAMP_DIR ${BINARY_DIR}/stamp
        URL ${DEP_URL}
        SOURCE_DIR ${SOURCE_DIR}
        CMAKE_CACHE_ARGS ${CMAKE_ARGS}
        BINARY_DIR ${BINARY_DIR}/build
        INSTALL_DIR ${INSTALL_DIR}
      )
    endif()
  else()
    file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/${DEP_NAME})
    file(REMOVE_RECURSE ${CMAKE_INSTALL_PREFIX}/${DEP_NAME})
  endif()
endfunction()


setdefaultcmakeinstallprefix()
setdefaultcmakebuildtype()

####################### Add dependencies below.

adddependency(
  NAME ncnn
  DEFAULT ON
  # DEPENDS Threads
  GIT_URL https://github.com/Tencent/ncnn.git
  GIT_TAG fd634e9
  CMAKE_ARGS
  -DNCNN_INSTALL_SDK:BOOL=ON
  -DNCNN_PIXEL_ROTATE:BOOL=OFF
  -DNCNN_PIXEL_AFFINE:BOOL=OFF
  -DNCNN_VULKAN:BOOL=ON
  -DNCNN_VULKAN_ONLINE_SPIRV:BOOL=ON
  -DNCNN_BUILD_BENCHMARK:BOOL=OFF
  -DNCNN_BUILD_TESTS:BOOL=OFF
  -DNCNN_BUILD_TOOLS:BOOL=OFF
  -DNCNN_BUILD_EXAMPLES:BOOL=OFF
  -DNCNN_DISABLE_RTTI:BOOL=ON
  -DNCNN_DISABLE_EXCEPTION:BOOL=OFF

  -DWITH_LAYER_absval:BOOL=OFF
  -DWITH_LAYER_argmax:BOOL=OFF
  -DWITH_LAYER_batchnorm:BOOL=OFF
  -DWITH_LAYER_bias:BOOL=OFF
  -DWITH_LAYER_bnll:BOOL=OFF
  -DWITH_LAYER_concat:BOOL=ON
  -DWITH_LAYER_convolution:BOOL=ON
  -DWITH_LAYER_crop:BOOL=ON
  -DWITH_LAYER_deconvolution:BOOL=ON
  -DWITH_LAYER_dropout:BOOL=OFF
  -DWITH_LAYER_eltwise:BOOL=OFF
  -DWITH_LAYER_elu:BOOL=OFF
  -DWITH_LAYER_embed:BOOL=OFF
  -DWITH_LAYER_exp:BOOL=OFF
  -DWITH_LAYER_flatten:BOOL=ON
  -DWITH_LAYER_innerproduct:BOOL=ON
  -DWITH_LAYER_input:BOOL=ON
  -DWITH_LAYER_log:BOOL=OFF
  -DWITH_LAYER_lrn:BOOL=OFF
  -DWITH_LAYER_memorydata:BOOL=OFF
  -DWITH_LAYER_mvn:BOOL=OFF
  -DWITH_LAYER_pooling:BOOL=ON
  -DWITH_LAYER_power:BOOL=OFF
  -DWITH_LAYER_prelu:BOOL=ON
  -DWITH_LAYER_proposal:BOOL=OFF
  -DWITH_LAYER_reduction:BOOL=OFF
  -DWITH_LAYER_relu:BOOL=ON
  -DWITH_LAYER_reshape:BOOL=OFF
  -DWITH_LAYER_roipooling:BOOL=OFF
  -DWITH_LAYER_scale:BOOL=OFF
  -DWITH_LAYER_sigmoid:BOOL=ON
  -DWITH_LAYER_slice:BOOL=ON
  -DWITH_LAYER_softmax:BOOL=OFF
  -DWITH_LAYER_split:BOOL=ON
  -DWITH_LAYER_spp:BOOL=OFF
  -DWITH_LAYER_tanh:BOOL=OFF
  -DWITH_LAYER_threshold:BOOL=OFF
  -DWITH_LAYER_tile:BOOL=OFF
  -DWITH_LAYER_rnn:BOOL=OFF
  -DWITH_LAYER_lstm:BOOL=OFF
  -DWITH_LAYER_binaryop:BOOL=ON
  -DWITH_LAYER_unaryop:BOOL=ON
  -DWITH_LAYER_convolutiondepthwise:BOOL=OFF
  -DWITH_LAYER_padding:BOOL=ON
  -DWITH_LAYER_squeeze:BOOL=OFF
  -DWITH_LAYER_expanddims:BOOL=OFF
  -DWITH_LAYER_normalize:BOOL=OFF
  -DWITH_LAYER_permute:BOOL=OFF
  -DWITH_LAYER_priorbox:BOOL=OFF
  -DWITH_LAYER_detectionoutput:BOOL=OFF
  -DWITH_LAYER_interp:BOOL=ON
  -DWITH_LAYER_deconvolutiondepthwise:BOOL=OFF
  -DWITH_LAYER_shufflechannel:BOOL=OFF
  -DWITH_LAYER_instancenorm:BOOL=OFF
  -DWITH_LAYER_clip:BOOL=ON
  -DWITH_LAYER_reorg:BOOL=OFF
  -DWITH_LAYER_yolodetectionoutput:BOOL=OFF
  -DWITH_LAYER_quantize:BOOL=OFF
  -DWITH_LAYER_dequantize:BOOL=OFF
  -DWITH_LAYER_yolov3detectionoutput:BOOL=OFF
  -DWITH_LAYER_psroipooling:BOOL=OFF
  -DWITH_LAYER_roialign:BOOL=OFF
  -DWITH_LAYER_packing:BOOL=ON
  -DWITH_LAYER_requantize:BOOL=OFF
  -DWITH_LAYER_cast:BOOL=ON
  -DWITH_LAYER_hardsigmoid:BOOL=OFF
  -DWITH_LAYER_selu:BOOL=OFF
  -DWITH_LAYER_hardswish:BOOL=OFF
  -DWITH_LAYER_noop:BOOL=OFF
  -DWITH_LAYER_pixelshuffle:BOOL=ON
  -DWITH_LAYER_deepcopy:BOOL=OFF
  -DWITH_LAYER_mish:BOOL=OFF
  -DWITH_LAYER_statisticspooling:BOOL=OFF
  -DWITH_LAYER_swish:BOOL=OFF
  -DWITH_LAYER_gemm:BOOL=OFF
  -DWITH_LAYER_groupnorm:BOOL=OFF
  -DWITH_LAYER_layernorm:BOOL=OFF
  -DWITH_LAYER_softplus:BOOL=OFF
)

#######################

removedefaultinstalldirifempty("${DEFAULT_CMAKE_INSTALL_PREFIX}")
