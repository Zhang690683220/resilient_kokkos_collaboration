
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>

#if defined(KR_ENABLE_HDF5_PARALLEL) || defined(KR_ENABLE_VELOC) || defined(KR_ENABLE_DATASPACES)
#include <mpi.h>
#endif

int
main( int argc, char **argv )
{
  ::testing::InitGoogleTest( &argc, argv );
#if defined(KR_ENABLE_HDF5_PARALLEL) || defined(KR_ENABLE_VELOC) || defined(KR_ENABLE_DATASPACES)
  MPI_Init( &argc, &argv );
#endif
  
  Kokkos::initialize( argc, argv );
  
  auto ret = RUN_ALL_TESTS();
  
  Kokkos::finalize();

#if defined(KR_ENABLE_HDF5_PARALLEL) || defined(KR_ENABLE_VELOC) || defined(KR_ENABLE_DATASPACES)
  MPI_Finalize();
#endif
  
  return ret;
}