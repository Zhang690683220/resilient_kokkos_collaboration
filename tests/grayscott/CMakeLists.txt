





SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -g" )
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall ")




option(USE_TIMERS "Use profiling timers")
if(USE_TIMERS)
  message(STATUS "Enabling profiling timers")
  add_definitions(-DENABLE_TIMERS)
endif()

# We are not using the C++ API of MPI, this will stop the compiler look for it
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

add_executable(gray-scott simulation/main.cpp
  simulation/gray-scott.cpp
  simulation/settings.cpp
  simulation/writer.cpp
)
target_include_directories(gray-scott PUBLIC ${gray-scott}/simulation)
target_link_libraries(gray-scott resilience MPI::MPI_CXX)

add_executable(pdf_calc
analysis/pdf_calc.cpp
analysis/analysis.cpp
simulation/settings.cpp
analysis/reader.cpp
)
target_include_directories(pdf_calc PUBLIC ${gray-scott}/simulation ${gray-scott}/analysis )
target_link_libraries(pdf_calc resilience MPI::MPI_CXX)


