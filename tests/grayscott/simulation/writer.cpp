#include "writer.h"
#include <iostream>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#define KOKKOS_ENABLE_MANUAL_CHECKPOINT
#include <resilience/Resilience.hpp>




template < typename ExecSpace, typename CpFileSpace >
void Writer::kokkos_write(const GrayScott &sim, MPI_Comm comm, int step) {

    size_t size_x = sim.size_x;
    size_t size_y = sim.size_y;
    size_t size_z = sim.size_z;

    std::vector<double> u = sim.u_noghost();
    std::vector<double> v = sim.v_noghost();

    typedef typename ExecSpace::memory_space     memory_space;

    typedef Kokkos::View<double***,memory_space> Rank3ViewType;
    Rank3ViewType view_3;
    view_3 = Rank3ViewType("memory_view_3", size_x, size_y, size_z);
    typename Rank3ViewType::HostMirror h_view_3 = Kokkos::create_mirror(view_3);

    typedef CpFileSpace cp_file_space_type;

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
    Kokkos::deep_copy( cp_view_U, h_view_3 );
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
    Kokkos::deep_copy( cp_view_V, h_view_3 );
    Kokkos::fence();


}

