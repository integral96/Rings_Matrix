cmake_minimum_required(VERSION 3.5)

project(Galerkin  ${PROJECT_NAME})


set(SOURCES
    main.cpp)
set(HEADERS
    include/matrix_rings.hpp
    include/base_operators.hpp
    include/puason_equation.hpp
)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
include_directories(include)

find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS context coroutine date_time timer thread system program_options regex)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME}  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    COMPILE_OPTIONS "-Wpedantic;-Wall;-Wextra"
    coroutine PROPERTIES

)

if(NOT DEFINED OpenGL_GL_PREFERENCE)
  set(OpenGL_GL_PREFERENCE "LEGACY")
endif()

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
