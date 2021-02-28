cmake_minimum_required(VERSION 3.0)

project(test_aesni)

add_executable(test_aesni aesni.cpp main.cpp)
set_property(TARGET test_aesni PROPERTY CXX_STANDARD 11)
target_link_libraries(test_aesni -lcrypto)

install(TARGETS test_aesni RUNTIME DESTINATION bin)
