cmake_minimum_required(VERSION 3.20)
project(tlc_ovgenai_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenVINOGenAI REQUIRED
  CONFIG
  PATHS "${OpenVINO_DIR}" "${OpenVINO_DIR}/cmake"
  NO_DEFAULT_PATH)
find_package(TBB QUIET
  CONFIG
  PATHS "${OpenVINO_DIR}/../3rdparty/tbb/lib/cmake/TBB"
  NO_DEFAULT_PATH)

add_executable(tlc-ovgenai-runner src/main.cpp)
target_include_directories(tlc-ovgenai-runner PRIVATE vendor src)
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
