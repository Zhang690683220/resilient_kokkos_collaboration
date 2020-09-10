#ifndef __WRITER_H__
#define __WRITER_H__

#include <mpi.h>
#include <string>
#include <vector>
#define KOKKOS_ENABLE_MANUAL_CHECKPOINT
#include <Kokkos_Macros.hpp>
#include <resilience/Resilience.hpp>

#include "../common/timer.hpp"
#include "gray-scott.h"
#include "settings.h"

class Writer
{
public:
    Writer(MPI_Comm comm ,int proc, int appid) {}

#ifdef ENABLE_TIMERS
    Timer timer_deepcopy;
    double time_singletimestep;
#endif

    template <typename ExecSpace, typename CpFileSpace >
    void kokkos_write(const GrayScott &sim, MPI_Comm comm, int step);
    
};

    template <>
    void Writer::kokkos_write<Kokkos::OpenMP, KokkosResilience::DataspacesSpace>
    (const GrayScott &sim, MPI_Comm comm, int step) 
      
    {

    size_t size_x = sim.size_x;
    size_t size_y = sim.size_y;
    size_t size_z = sim.size_z;

    std::vector<double> u = sim.u_noghost();
    std::vector<double> v = sim.v_noghost();

    typedef Kokkos::OpenMP::memory_space     memory_space;

    typedef Kokkos::View<double***,memory_space> Rank3ViewType;
    Rank3ViewType view_3;
    view_3 = Rank3ViewType("memory_view_3", size_x, size_y, size_z);
    Rank3ViewType::HostMirror h_view_3 = Kokkos::create_mirror(view_3);

    typedef KokkosResilience::DataspacesSpace cp_file_space_type;

    std::string fileNameU = "grayscott_u/t" + std::to_string(step);
    std::string fileNameV = "grayscott_v/t" + std::to_string(step);

    Kokkos::View<double***,cp_file_space_type> cp_view_U(fileNameU, size_x, size_y, size_z);
    Kokkos::View<double***,cp_file_space_type> cp_view_V(fileNameV, size_x, size_y, size_z);

    // filling local view and host view with U first
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                view_3(x,y,z) = u[x+size_x*y+size_x*size_y*z];
            }
        }
    });

    Kokkos::deep_copy(h_view_3, view_3);

    // host_space to ExecSpace
#ifdef ENABLE_TIMERS
    MPI_Barrier(comm);
    timer_deepcopy.start();
#endif

    Kokkos::deep_copy( cp_view_U, h_view_3 );
#ifdef ENABLE_TIMERS
    double putUTime = timer_deepcopy.stop();
#endif
    Kokkos::fence();

    // filling local view and host view with V 
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                view_3(x,y,z) = v[x+size_x*y+size_x*size_y*z];
            }
        }
    });

    Kokkos::deep_copy(h_view_3, view_3);

    // host_space to ExecSpace
#ifdef ENABLE_TIMERS
    MPI_Barrier(comm);
    timer_deepcopy.start();
#endif

    Kokkos::deep_copy( cp_view_V, h_view_3 );
#ifdef ENABLE_TIMERS
    double putVTime = timer_deepcopy.stop();
    time_singletimestep = putUTime + putVTime;
#endif
    Kokkos::fence();



    }

    template <>
    void Writer::kokkos_write<Kokkos::OpenMP, KokkosResilience::HDF5Space>
    (const GrayScott &sim, MPI_Comm comm, int step) 
      
    {
    
    int rank;
    MPI_Comm_rank(comm, &rank);

    size_t size_x = sim.size_x;
    size_t size_y = sim.size_y;
    size_t size_z = sim.size_z;

    std::vector<double> u = sim.u_noghost();
    std::vector<double> v = sim.v_noghost();

    typedef Kokkos::OpenMP::memory_space     memory_space;

    typedef Kokkos::View<double***,memory_space> Rank3ViewType;
    Rank3ViewType view_3;
    view_3 = Rank3ViewType("memory_view_3", size_x, size_y, size_z);
    Rank3ViewType::HostMirror h_view_3 = Kokkos::create_mirror(view_3);

    typedef KokkosResilience::HDF5Space cp_file_space_type;

    std::string fileNameU = "grayscott_u.t" + std::to_string(step) + ".r" + std::to_string(rank) + ".hdf";
    std::string fileNameV = "grayscott_v.t" + std::to_string(step) + ".r" + std::to_string(rank) + ".hdf";

    Kokkos::View<double***,cp_file_space_type> cp_view_U(fileNameU, size_x, size_y, size_z);
    Kokkos::View<double***,cp_file_space_type> cp_view_V(fileNameV, size_x, size_y, size_z);

    // filling local view and host view with U first
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                view_3(x,y,z) = u[x+size_x*y+size_x*size_y*z];
            }
        }
    });

    Kokkos::deep_copy(h_view_3, view_3);

    // host_space to ExecSpace
#ifdef ENABLE_TIMERS
    MPI_Barrier(comm);
    timer_deepcopy.start();
#endif
    Kokkos::deep_copy( cp_view_U, h_view_3 );
#ifdef ENABLE_TIMERS
    double putUTime = timer_deepcopy.stop();
#endif
    Kokkos::fence();

    // filling local view and host view with V 
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                view_3(x,y,z) = v[x+size_x*y+size_x*size_y*z];
            }
        }
    });

    Kokkos::deep_copy(h_view_3, view_3);

    // host_space to ExecSpace
#ifdef ENABLE_TIMERS
    MPI_Barrier(comm);
    timer_deepcopy.start();
#endif
    Kokkos::deep_copy( cp_view_V, h_view_3 );
#ifdef ENABLE_TIMERS
    double putVTime = timer_deepcopy.stop();
    time_singletimestep = putUTime + putVTime;
#endif
    Kokkos::fence();


    }

    template <>
    void Writer::kokkos_write<Kokkos::OpenMP, KokkosResilience::StdFileSpace>
    (const GrayScott &sim, MPI_Comm comm, int step) 
      
    {
    
    int rank;
    MPI_Comm_rank(comm, &rank);

    size_t size_x = sim.size_x;
    size_t size_y = sim.size_y;
    size_t size_z = sim.size_z;

    std::vector<double> u = sim.u_noghost();
    std::vector<double> v = sim.v_noghost();

    typedef Kokkos::OpenMP::memory_space     memory_space;

    typedef Kokkos::View<double***,memory_space> Rank3ViewType;
    Rank3ViewType view_3;
    view_3 = Rank3ViewType("memory_view_3", size_x, size_y, size_z);
    Rank3ViewType::HostMirror h_view_3 = Kokkos::create_mirror(view_3);

    typedef KokkosResilience::StdFileSpace cp_file_space_type;

    std::string fileNameU = "grayscott_u.t" + std::to_string(step) + ".r" + std::to_string(rank) + ".bin";
    std::string fileNameV = "grayscott_v.t" + std::to_string(step) + ".r" + std::to_string(rank) + ".bin";

    Kokkos::View<double***,cp_file_space_type> cp_view_U(fileNameU, size_x, size_y, size_z);
    Kokkos::View<double***,cp_file_space_type> cp_view_V(fileNameV, size_x, size_y, size_z);

    // filling local view and host view with U first
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                view_3(x,y,z) = u[x+size_x*y+size_x*size_y*z];
            }
        }
    });

    Kokkos::deep_copy(h_view_3, view_3);

    // host_space to ExecSpace
#ifdef ENABLE_TIMERS
    MPI_Barrier(comm);
    timer_deepcopy.start();
#endif
    Kokkos::deep_copy( cp_view_U, h_view_3 );
#ifdef ENABLE_TIMERS
    double putUTime = timer_deepcopy.stop();
#endif
    Kokkos::fence();

    // filling local view and host view with V 
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                view_3(x,y,z) = v[x+size_x*y+size_x*size_y*z];
            }
        }
    });

    Kokkos::deep_copy(h_view_3, view_3);

    // host_space to ExecSpace
#ifdef ENABLE_TIMERS
    MPI_Barrier(comm);
    timer_deepcopy.start();
#endif
    Kokkos::deep_copy( cp_view_V, h_view_3 );
#ifdef ENABLE_TIMERS
    double putVTime = timer_deepcopy.stop();
    time_singletimestep = putUTime + putVTime;
#endif
    Kokkos::fence();


    }
    //void close() { dspaces_finalize();}




#endif
