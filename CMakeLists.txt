cmake_minimum_required(VERSION 3.20)
project(tlc_ovgenai_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenVINOGenAI REQUIRED
  CONFIG
  PATHS "${OpenVINO_DIR}" "${OpenVINO_DIR}/cmake"
  NO_DEFAULT_PATH)
find_package(TBB QUIET
  CONFIG
  PATHS "${OpenVINO_DIR}/../3rdparty/tbb/lib/cmake/TBB"
  NO_DEFAULT_PATH)

# OpenVINO GenAI version this build links against (set by the Taskfile from
# OV_PKG_VERSION). Embedded into the binary as a byte-scan marker so the
# extension can download the matching runtime; "unknown" makes the extension
# fall back to its own pinned version.
set(OV_GENAI_VERSION "unknown" CACHE STRING "OpenVINO GenAI version this build links against")

# App version from git: exact tag on HEAD, otherwise <tag>-<N>-g<sha>+<branch>.<build-time>
# ("detached" without a branch, "unknown" outside a git repo).
execute_process(COMMAND git describe --tags --exact-match --dirty
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE GIT_TAG RESULT_VARIABLE GIT_TAG_RC)
execute_process(COMMAND git describe --tags --long --always --dirty
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE GIT_DESCRIBE)
execute_process(COMMAND git branch --show-current
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE GIT_BRANCH)
string(TIMESTAMP BUILD_TIME "%Y%m%d%H%M%S")
set(APP_VERSION "unknown")
if(GIT_TAG_RC EQUAL 0 AND NOT GIT_TAG STREQUAL "")
  set(APP_VERSION "${GIT_TAG}")
elseif(NOT GIT_DESCRIBE STREQUAL "")
  if(GIT_BRANCH STREQUAL "")
    set(GIT_BRANCH "detached")
  endif()
  set(APP_VERSION "${GIT_DESCRIBE}+${GIT_BRANCH}.${BUILD_TIME}")
endif()

add_executable(tlc-ovgenai-runner src/main.cpp)
target_include_directories(tlc-ovgenai-runner PRIVATE vendor src)
target_compile_definitions(tlc-ovgenai-runner PRIVATE
  TLC_OVGENAI_OV_VERSION="${OV_GENAI_VERSION}"
  TLC_OVGENAI_APP_VERSION="${APP_VERSION}")
target_link_libraries(tlc-ovgenai-runner PRIVATE openvino::runtime openvino::genai)
if(TBB_FOUND)
  target_link_libraries(tlc-ovgenai-runner PRIVATE TBB::tbb)
endif()

enable_testing()
add_executable(runner-tests src/tests.cpp)
target_include_directories(runner-tests PRIVATE vendor src)
add_test(NAME runner-tests COMMAND runner-tests)

if(MSVC)
  target_compile_options(tlc-ovgenai-runner PRIVATE /W4 /EHsc)
  target_compile_options(runner-tests PRIVATE /W4 /EHsc)
else()
  target_compile_options(tlc-ovgenai-runner PRIVATE -Wall -Wextra)
  target_compile_options(runner-tests PRIVATE -Wall -Wextra)
endif()

install(TARGETS tlc-ovgenai-runner RUNTIME DESTINATION .)
