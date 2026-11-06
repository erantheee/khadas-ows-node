cmake_minimum_required(VERSION 3.20)
project(khadas_ows_node LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

add_library(khadas_ows_core src/audio_core.cpp src/conversation.cpp)
target_include_directories(khadas_ows_core PUBLIC include)
target_compile_options(khadas_ows_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(khadas_ows_core PUBLIC Threads::Threads)

add_executable(khadas-ows-node src/main.cpp)
target_link_libraries(khadas-ows-node PRIVATE khadas_ows_core)
target_compile_options(khadas-ows-node PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()

function(khadas_add_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE khadas_ows_core)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES TIMEOUT 60)
endfunction()

khadas_add_test(core_test tests/core_test.cpp)
khadas_add_test(policy_test tests/policy_test.cpp)
khadas_add_test(conversation_golden_test tests/conversation_golden_test.cpp)
khadas_add_test(failure_matrix_test tests/failure_matrix_test.cpp)

# Exercises the real binary end-to-end: env-configured ports (KHADAS_AUDIO_PORT /
# KHADAS_HTTP_PORT), TCP PCM passthrough and SIGTERM shutdown.
khadas_add_test(server_env_test tests/server_env_test.cpp)
target_compile_definitions(server_env_test PRIVATE
    KHADAS_NODE_BINARY="$<TARGET_FILE:khadas-ows-node>")
add_dependencies(server_env_test khadas-ows-node)
