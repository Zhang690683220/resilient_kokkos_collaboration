add_executable(kokkos_test_writer test_writer.cpp)
target_link_libraries(kokkos_test_writer resilience MPI::MPI_CXX)

add_executable(kokkos_test_reader test_reader.cpp)
target_link_libraries(kokkos_test_reader resilience MPI::MPI_CXX)