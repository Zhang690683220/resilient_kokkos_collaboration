





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

add_executable(gray-scott-stdio simulation/main_stdio.cpp
  simulation/gray-scott.cpp
  simulation/settings.cpp
)
target_include_directories(gray-scott-stdio PUBLIC ${gray-scott}/simulation)
target_link_libraries(gray-scott-stdio resilience MPI::MPI_CXX)

add_executable(pdf_calc_stdio
analysis/pdf_calc_stdio.cpp
analysis/analysis.cpp
simulation/settings.cpp
)
target_include_directories(pdf_calc_stdio PUBLIC ${gray-scott}/simulation ${gray-scott}/analysis )
target_link_libraries(pdf_calc_stdio resilience MPI::MPI_CXX)

add_executable(gray-scott-h5 simulation/main_h5.cpp
  simulation/gray-scott.cpp
  simulation/settings.cpp
)
target_include_directories(gray-scott-h5 PUBLIC ${gray-scott}/simulation)
target_link_libraries(gray-scott-h5 resilience MPI::MPI_CXX)

add_executable(pdf_calc_h5
analysis/pdf_calc_h5.cpp
analysis/analysis.cpp
simulation/settings.cpp
)
target_include_directories(pdf_calc_h5 PUBLIC ${gray-scott}/simulation ${gray-scott}/analysis )
target_link_libraries(pdf_calc_h5 resilience MPI::MPI_CXX)

add_executable(gray-scott-ds simulation/main_ds.cpp
  simulation/gray-scott.cpp
  simulation/settings.cpp
)
target_include_directories(gray-scott-ds PUBLIC ${gray-scott}/simulation)
target_link_libraries(gray-scott-ds resilience MPI::MPI_CXX)

add_executable(pdf_calc_ds
analysis/pdf_calc_ds.cpp
analysis/analysis.cpp
simulation/settings.cpp
)
target_include_directories(pdf_calc_ds PUBLIC ${gray-scott}/simulation ${gray-scott}/analysis )
target_link_libraries(pdf_calc_ds resilience MPI::MPI_CXX)
