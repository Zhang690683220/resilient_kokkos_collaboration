#include "reader.h"
#include <iostream>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#define KOKKOS_ENABLE_MANUAL_CHECKPOINT
#include <resilience/Resilience.hpp>





void Reader::kokkos_read(Analysis &analysis, MPI_Comm comm, int step)
{
    size_t size_x = analysis.size_x;
    size_t size_y = analysis.size_y;
    size_t size_z = analysis.size_z;

    double *u_buf = (double*) malloc(size_x*size_y*size_z*sizeof(double));
    double *v_buf = (double*) malloc(size_x*size_y*size_z*sizeof(double));


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

    // ExecSpace to host_space
    Kokkos::deep_copy( h_view_3, cp_view_U );
    Kokkos::fence();

    // filling local array with h_view_3_U first
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                u_buf[x+size_x*y+size_x*size_y*z] = h_view_3(x,y,z);
            }
        }
    });

    Kokkos::deep_copy( h_view_3, cp_view_V );
    Kokkos::fence();

    // filling local array with h_view_3_V first
    Kokkos::parallel_for (Kokkos::RangePolicy<Kokkos::OpenMP>(0, size_z), KOKKOS_LAMBDA (const int z) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                v_buf[x+size_x*y+size_x*size_y*z] = h_view_3(x,y,z);
            }
        }
    });

    analysis.fillU(u_buf);
    analysis.fillV(v_buf);
    
}