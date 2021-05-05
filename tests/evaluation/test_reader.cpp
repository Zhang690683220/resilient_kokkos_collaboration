#include <Kokkos_Core.hpp>
#define KOKKOS_ENABLE_MANUAL_CHECKPOINT
#include <resilience/Resilience.hpp>
#include <iostream>
#include <vector>
#include "CLI11.hpp"
#include "mpi.h"
#include "test_reader.hpp"



void print_usage()
{
    std::cerr<<"Usage: test_writer --dims <dims> --np <np[0] .. np[dims-1]> --sp <sp[0] ... sp[dims-1]> "
               "--ts <timesteps> [-s <elem_size>] [-c <var_count>] [-t <terminate>]"<<std::endl
             <<"--dims                      - number of data dimensions. Must be [1-8]"<<std::endl
             <<"--np                        - the number of processes in the ith dimension. "
               "The product of np[0],...,np[dim-1] must be the number of MPI ranks"<<std::endl
             <<"--sp                        - the per-process data size in the ith dimension"<<std::endl
             <<"--backend                   - backend of staging file. 1 for Cpp Std File,"
               " 2 for HDF5. Defaults to 1"<<std::endl
             <<"--ts                        - the number of timestep iterations written"<<std::endl
             <<"-s, --elem_size (optional)  - the number of bytes in each element. Defaults to 8"<<std::endl
             <<"-c, --var_count (optional)  - the number of variables written in each iteration. "
               "Defaults to 1"<<std::endl
             <<"-t (optional)               - send server termination after writing is complete"<<std::endl;
}

int main(int argc, char** argv)
{
    CLI::App app{"Test Writer for Kokkos Staging Space"};
    int dims;              // number of dimensions
    std::vector<int> np;
    std::vector<uint64_t> sp;
    std::vector<int> src_np;
    std::vector<uint64_t> src_sp;
    std::vector<uint64_t> offset;
    int backend = 1;
    int timestep;
    size_t elem_size = 8;
    int num_vars = 1;
    bool terminate = false;
    app.add_option("--dims", dims, "number of data dimensions. Must be [1-8]")->required();
    app.add_option("--np", np, "the number of processes in the ith dimension. The product of np[0],"
                    "...,np[dim-1] must be the number of MPI ranks")->expected(1, 8);
    app.add_option("--sp", sp, "the per-process data size in the ith dimension")->expected(1, 8);
    app.add_option("--src_np", src_np, "the number of processes in the ith dimension of the src. The product of np[0],"
                    "...,np[dim-1] must be the number of MPI ranks")->expected(1, 8);
    app.add_option("--src_sp", src_sp, "the per-process data size in the ith dimension of the src.")->expected(1, 8);
    app.add_option("--offset", offset, "the bounding box offset in the ith dimension of the src.");
    app.add_option("--backend", backend, "backend of staging file. 1 for Cpp Std File, 2 for"
                    "HDF5. Defaults to 1", true);
    app.add_option("--ts", timestep, "the number of timestep iterations written")->required();
    app.add_option("-s, --elem_size", elem_size, "the number of bytes in each element. Defaults to 8",
                    true);
    app.add_option("-c, --var_count", num_vars, "the number of variables written in each iteration."
                    "Defaults to 1", true);
    app.add_option("-t", terminate, "send server termination after writing is complete", true);

    CLI11_PARSE(app, argc, argv);

    int npapp = 1;             // number of application processes
    for(int i = 0; i < dims; i++) {
        npapp *= np[i];
    }

    int nprocs, rank;
    MPI_Comm gcomm;
    // Using SPMD style programming
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    gcomm = MPI_COMM_WORLD;

    int color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &gcomm);

    if(npapp != nprocs) {
        std::cerr<<"Product of np[i] args must equal number of MPI processes!"<<std::endl;
        fprintf(stderr,
                "Product of np[i] args must equal number of MPI processes!\n");
        print_usage();
        return (-1);
    }

    if(offset.empty()) {
        offset.resize(dims);
        for(int i=0; i<dims; i++) {
            offset[i] = 0;
        }
    }

    if(offset.size() != dims) {
        std::cerr<<"offset args must equal number of dims"<<std::endl;
        print_usage();
        return (-1);
    }

    Kokkos::initialize(argc, argv);
    {
        switch (dims)
        {
        case 2:
            switch (elem_size)
            {
            case 4:
                switch (backend)
                {
                case 1:
                    kokkos_run<float, 2, KokkosResilience::StdFileSpace>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;

                case 2:
                    kokkos_run<float, 2, KokkosResilience::HDF5Space>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;
                
                default:
                    break;
                }
                break;

            case 8:
                switch (backend)
                {
                case 1:
                    kokkos_run<double, 2, KokkosResilience::StdFileSpace>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;

                case 2:
                    kokkos_run<double, 2, KokkosResilience::StdFileSpace>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;
                
                default:
                    break;
                }
                break;
            
            default:
                break;
            }
        case 3:
            switch (elem_size)
            {
            case 4:
                switch (backend)
                {
                case 1:
                    kokkos_run<float, 3, KokkosResilience::StdFileSpace>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;

                case 2:
                    kokkos_run<float, 3, KokkosResilience::HDF5Space>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;
                
                default:
                    break;
                }
                break;

            case 8:
                switch (backend)
                {
                case 1:
                    kokkos_run<double, 3, KokkosResilience::StdFileSpace>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;

                case 2:
                    kokkos_run<double, 3, KokkosResilience::StdFileSpace>::get_run(gcomm, np.data(), sp.data(),
                                                                    src_np.data(), src_sp.data(), offset.data(),
                                                                    timestep, num_vars, terminate);
                    break;
                
                default:
                    break;
                }
                break;
            
            default:
                break;
            }

        default:
            break;
        }
    }
    Kokkos::finalize();

    MPI_Barrier(gcomm);
    MPI_Finalize();

    if(rank == 0) {
        fprintf(stderr, "That's all from test_writer, folks!\n");
    }

    return 0;
err_out:
    fprintf(stderr, "test_writer rank %d has failed.!\n", rank);
    return -1;

}