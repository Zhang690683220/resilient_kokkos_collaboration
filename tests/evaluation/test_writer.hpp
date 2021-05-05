#ifndef TEST_WRITER_HPP
#define TEST_WRITER_HPP

#include <Kokkos_Core.hpp>
#define KOKKOS_ENABLE_MANUAL_CHECKPOINT
#include <resilience/Resilience.hpp>
#include <iostream>
#include "timer.hpp"
#include "mpi.h"

// only support 1 var_num now.
template <class Data_t, unsigned int Dims, class StagingSpace>
struct kokkos_run {
    static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num, bool terminate);
};

template <class Data_t>
struct kokkos_run<Data_t, 2, KokkosResilience::StdFileSpace> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);


    using ViewHost_t    = Kokkos::View<Data_t**, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t**, KokkosResilience::StdFileSpace>;

    uint64_t* off = (uint64_t*) malloc(2*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(2*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(2*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<2; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_P("PutView", sp[0], sp[1]);
    //std::string filename = ""
    //ViewStaging_t v_S("StagingView_3D", sp[0], sp[1]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open("test_writer.log", std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                v_P(i0,i1) = i0*sp[1]+i1 + 0.01*ts;
                std::cout<<v_P(i0,i1)<<"\t";
            }
            std::cout<<std::endl;
        });

        std::string filename = "StagingView_2D_" + std::to_string(lb[0]) + "_"
                                + std::to_string(lb[1]) + "_" + std::to_string(ub[0]) 
                                + "_" + std::to_string(ub[1]) + "_t" + std::to_string(ts) + ".bin";
        
        ViewStaging_t v_S(filename, sp[0], sp[1]);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        Kokkos::fence();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
        }
    }


    return 0;
};
};

template <class Data_t>
struct kokkos_run<Data_t, 2, KokkosResilience::HDF5Space> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);


    using ViewHost_t    = Kokkos::View<Data_t**, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t**, KokkosResilience::HDF5Space>;

    uint64_t* off = (uint64_t*) malloc(2*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(2*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(2*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<2; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_P("PutView", sp[0], sp[1]);
    //std::string filename = ""
    //ViewStaging_t v_S("StagingView_3D", sp[0], sp[1], sp[2]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open("test_writer.log", std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                v_P(i0,i1) = i0*sp[1]+i1 + 0.01*ts;
                std::cout<<v_P(i0,i1)<<"\t";
            }
            std::cout<<std::endl;
        });

        std::string filename = "StagingView_2D_" + std::to_string(lb[0]) + "_"
                                + std::to_string(lb[1]) + "_" + std::to_string(ub[0]) 
                                + "_" + std::to_string(ub[1]) + "_t" + std::to_string(ts) + ".hdf";
        
        ViewStaging_t v_S(filename, sp[0], sp[1]);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        Kokkos::fence();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
        }
    }


    return 0;
};
};

template <class Data_t>
struct kokkos_run<Data_t, 3, KokkosResilience::StdFileSpace> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);


    using ViewHost_t    = Kokkos::View<Data_t***, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t***, KokkosResilience::StdFileSpace>;

    uint64_t* off = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(3*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<3; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2]);
    //std::string filename = ""
    ViewStaging_t v_S("StagingView_3D", sp[0], sp[1], sp[2]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open("test_writer.log", std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    v_P(i0,i1,i2) = (i0*sp[1]+i1)*sp[2]+i2 + 0.01*ts;
                    std::cout<<v_P(i0,i1,i2)<<"\t";
                }
                std::cout<<std::endl;
            }
            std::cout<<"******************"<<std::endl;
        });

        std::string filename = "StagingView_3D_" + std::to_string(lb[0]) + "_"
                                + std::to_string(lb[1]) + "_" + std::to_string(lb[2]) + "_"
                                + std::to_string(ub[0]) + "_" + std::to_string(ub[1]) + "_"
                                + std::to_string(ub[2]) + "_t" + std::to_string(ts) + ".bin";
        
        ViewStaging_t v_S(filename, sp[0], sp[1], sp[2]);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        Kokkos::fence();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
        }
    }


    return 0;
};
};

template <class Data_t>
struct kokkos_run<Data_t, 3, KokkosResilience::HDF5Space> {
static int put_run (MPI_Comm gcomm, int* np, uint64_t* sp, int timesteps, int var_num, bool terminate)
{
    int rank, nprocs;
    MPI_Comm_size(gcomm, &nprocs);
    MPI_Comm_rank(gcomm, &rank);


    using ViewHost_t    = Kokkos::View<Data_t***, Kokkos::HostSpace>;
    using ViewStaging_t = Kokkos::View<Data_t***, KokkosResilience::HDF5Space>;

    uint64_t* off = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* lb = (uint64_t*) malloc(3*sizeof(uint64_t));
    uint64_t* ub = (uint64_t*) malloc(3*sizeof(uint64_t));

    // get the lb & ub for each rank
    for(int i=0; i<3; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    ViewHost_t v_P("PutView", sp[0], sp[1], sp[2]);
    //std::string filename = ""
    ViewStaging_t v_S("StagingView_3D", sp[0], sp[1], sp[2]);

    std::ofstream log;
    double* avg_write = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_write = (double*) malloc(sizeof(double)*timesteps);
        log.open("test_writer.log", std::ofstream::out | std::ofstream::trunc);
        log << "step\twrite_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
            for(int i1=0; i1<sp[1]; i1++) {
                for(int i2=0; i2<sp[2]; i2++) {
                    v_P(i0,i1,i2) = (i0*sp[1]+i1)*sp[2]+i2 + 0.01*ts;
                    std::cout<<v_P(i0,i1,i2)<<"\t";
                }
                std::cout<<std::endl;
            }
            std::cout<<"******************"<<std::endl;
        });

        std::string filename = "StagingView_3D_" + std::to_string(lb[0]) + "_"
                                + std::to_string(lb[1]) + "_" + std::to_string(lb[2]) + "_"
                                + std::to_string(ub[0]) + "_" + std::to_string(ub[1]) + "_"
                                + std::to_string(ub[2]) + "_t" + std::to_string(ts) + ".hdf";
        
        ViewStaging_t v_S(filename, sp[0], sp[1], sp[2]);

        Timer timer_write;

        timer_write.start();

        Kokkos::deep_copy(v_S, v_P);

        double time_write = timer_write.stop();

        Kokkos::fence();

        double *avg_time_write = nullptr;

        if(rank == 0) {
            avg_time_write = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_write, 1, MPI_DOUBLE, avg_time_write, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_write[ts-1] += avg_time_write[i];
            }
            avg_write[ts-1] /= nprocs;
            log << ts << "\t" << avg_write[ts-1] << "\t" << std::endl;
            total_avg += avg_write[ts-1];
            free(avg_time_write);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(avg_write);

    if(rank == 0) {
        total_avg /= timesteps;
        log << "Total" << "\t" << total_avg << "\t" << std::endl;
        log.close();
        if(terminate) {
            std::cout<<"Writer sending kill signal to server."<<std::endl;
        }
    }


    return 0;
};
};

#endif // TEST_WRITER_HPP