#ifndef TEST_READER_HPP
#define TEST_READER_HPP

#include <Kokkos_Core.hpp>
#define KOKKOS_ENABLE_MANUAL_CHECKPOINT
#include <resilience/Resilience.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include "mpi.h"

#define bb_max(a, b) (a) > (b) ? (a) : (b)
#define bb_min(a, b) (a) < (b) ? (a) : (b)

#define BBOX_MAX_NDIM 8

struct coord {
    uint64_t c[BBOX_MAX_NDIM];
};

struct bbox {
    int num_dims;
    struct coord lb, ub;
};


/*
  Test if bounding boxes b0 and b1 intersect along dimension dim.
*/
static int bbox_intersect_ondim(const struct bbox *b0, const struct bbox *b1,
                                int dim)
{
    if((b0->lb.c[dim] <= b1->lb.c[dim] && b1->lb.c[dim] <= b0->ub.c[dim]) ||
       (b1->lb.c[dim] <= b0->lb.c[dim] && b0->lb.c[dim] <= b1->ub.c[dim]))
        return 1;
    else
        return 0;
}

/*
  Test if bounding boxes b0 and b1 intersect (on all dimensions).
*/
int bbox_does_intersect(const struct bbox *b0, const struct bbox *b1)
{
    int i;

    for(i = 0; i < b0->num_dims; i++) {
        if(!bbox_intersect_ondim(b0, b1, i))
            return 0;
    }

    return 1;
}

/*
  Compute the intersection of bounding boxes b0 and b1, and store it on
  b2. Implicit assumption: b0 and b1 intersect.
*/
void bbox_intersect(const struct bbox *b0, const struct bbox *b1,
                    struct bbox *b2)
{
    int i;

    b2->num_dims = b0->num_dims;
    for(i = 0; i < b0->num_dims; i++) {
        b2->lb.c[i] = bb_max(b0->lb.c[i], b1->lb.c[i]);
        b2->ub.c[i] = bb_min(b0->ub.c[i], b1->ub.c[i]);
    }
}


// only support 1 var_num now.
template <class Data_t, unsigned int Dims, class StagingSpace>
struct kokkos_run {
    static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int* src_np,
                        uint64_t* src_sp, uint64_t* offset, int timesteps,
                        int var_num, bool terminate);
};

template <class Data_t>
struct kokkos_run<Data_t, 3, KokkosResilience::StdFileSpace> {
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int* src_np,
                    uint64_t* src_sp, uint64_t* offset, int timesteps,
                    int var_num, bool terminate)
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
        lb[i] = off[i] + offset[i];
        ub[i] = off[i] + sp[i] - 1 + offset[i];
    }

    struct bbox local_bb;

    local_bb.num_dims = 3;
    memcpy(local_bb.lb.c, lb, 3*sizeof(uint64_t));
    memcpy(local_bb.ub.c, ub, 3*sizeof(uint64_t));

    struct bbox* src_bbox_tab = (struct bbox*) malloc(src_np[0]*src_np[1]*src_np[2]*sizeof(struct bbox));
    int iter[3];
    for(iter[0]=0; iter[0]<src_np[0]; iter[0]++) {
        for(iter[1]=0; iter[1]<src_np[1]; iter[1]++) {
            for(iter[2]=0; iter[2]<src_np[2]; iter[2]++){
                src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].num_dims = 3;
                for(int d=0; d<3; d++) {
                    src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].lb.c[d] = iter[d]*src_sp[d];
                    src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].ub.c[d] = (iter[d]+1)*src_sp[d]-1;
                }
                std::cout<<"lb: {"<<src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].lb.c[0]<<", "
                        <<src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].lb.c[1]<<", "
                        <<src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].lb.c[2]<<"}\n"
                        <<"ub: {"<<src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].ub.c[0]<<", "
                        <<src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].ub.c[1]<<", "
                        <<src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].ub.c[2]<<"}"<<std::endl;
            }
        }
    }
    

    
    int open_num = 0;
    for(int i=0; i<src_np[0]*src_np[1]*src_np[2]; i++) {
        if(bbox_does_intersect(&local_bb, &src_bbox_tab[i])) {
            open_num++;
        }
    }

    std::vector<std::string> open_tab;

    open_tab.resize(open_num);
    for(int i=0; i<src_np[0]*src_np[1]*src_np[2]; i++) {
        if(bbox_does_intersect(&local_bb, &src_bbox_tab[i])) {
            std::string tmp = "StagingView_3D_" + std::to_string(src_bbox_tab[i].lb.c[0]) + "_"
                                + std::to_string(src_bbox_tab[i].lb.c[1]) + "_" + std::to_string(src_bbox_tab[i].lb.c[2]) + "_"
                                + std::to_string(src_bbox_tab[i].ub.c[0]) + "_" + std::to_string(src_bbox_tab[i].ub.c[1]) + "_"
                                + std::to_string(src_bbox_tab[i].ub.c[2]);
            open_tab[i] = tmp;
        }
    }




    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2]);
    ViewHost_t v_tmp("TmpView", src_sp[0], src_sp[1], src_sp[2]);


    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open("test_reader.log", std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Timer timer_read;

        timer_read.start();

        for(int i=0; i<open_tab.size(); i++) {
            std::string filename = open_tab[i] + "_t" + std::to_string(ts) + ".bin";

            ViewStaging_t v_S(filename, src_sp[0], src_sp[1], src_sp[2]);

            Kokkos::deep_copy(v_tmp, v_S);

            struct bbox tmp_bbox;

            bbox_intersect(&local_bb, &src_bbox_tab[i], &tmp_bbox);
            

            Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
                for(int i1=0; i1<sp[1]; i1++) {
                    for(int i2=0; i2<sp[2]; i2++) {
                        v_G(i0+tmp_bbox.lb.c[0]-local_bb.lb.c[0],
                            i1+tmp_bbox.lb.c[1]-local_bb.lb.c[1],
                            i2+tmp_bbox.lb.c[2]-local_bb.lb.c[2]) = v_tmp(i0+tmp_bbox.lb.c[0]-src_bbox_tab[i].lb.c[0],
                                                                        i1+tmp_bbox.lb.c[0]-src_bbox_tab[i].lb.c[1],
                                                                        i2+tmp_bbox.lb.c[0]-src_bbox_tab[i].lb.c[2]);
                    }
                }
            });

            std::cout<<filename<<std::endl;
        }

        double time_read = timer_read.stop();

        Kokkos::fence();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(src_bbox_tab);
    free(avg_read);

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
static int get_run (MPI_Comm gcomm, int* np, uint64_t* sp, int* src_np,
                    uint64_t* src_sp, uint64_t* offset, int timesteps,
                    int var_num, bool terminate)
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
        lb[i] = off[i] + offset[i];
        ub[i] = off[i] + sp[i] - 1 + offset[i];
    }

    struct bbox local_bb;

    local_bb.num_dims = 3;
    memcpy(local_bb.lb.c, lb, 3*sizeof(uint64_t));
    memcpy(local_bb.ub.c, ub, 3*sizeof(uint64_t));

    struct bbox* src_bbox_tab = (struct bbox*) malloc(src_np[0]*src_np[1]*src_np[2]*sizeof(struct bbox));
    int iter[3];
    for(iter[0]; iter[0]<src_np[0]; iter[0]++) {
        for(iter[1]; iter[1]<src_np[1]; iter[1]++) {
            for(iter[2]; iter[2]<src_np[2]; iter[2]++){
                src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].num_dims = 3;
                for(int d=0; d<3; d++) {
                    src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].lb.c[d] = iter[d]*src_sp[d];
                    src_bbox_tab[(iter[0]*src_np[1]+iter[1])*src_np[2]+iter[2]].ub.c[d] = (iter[d]+1)*src_sp[d]-1;
                }
            }
        }
    }
    

    
    int open_num = 0;
    for(int i=0; i<src_np[0]*src_np[1]*src_np[2]; i++) {
        if(bbox_does_intersect(&local_bb, &src_bbox_tab[i])) {
            open_num++;
        }
    }

    std::vector<std::string> open_tab;

    open_tab.resize(open_num);
    for(int i=0; i<src_np[0]*src_np[1]*src_np[2]; i++) {
        if(bbox_does_intersect(&local_bb, &src_bbox_tab[i])) {
            std::string tmp = "StagingView_3D_" + std::to_string(src_bbox_tab[i].lb.c[0]) + "_"
                                + std::to_string(src_bbox_tab[i].lb.c[1]) + "_" + std::to_string(src_bbox_tab[i].lb.c[2]) + "_"
                                + std::to_string(src_bbox_tab[i].ub.c[0]) + "_" + std::to_string(src_bbox_tab[i].ub.c[1]) + "_"
                                + std::to_string(src_bbox_tab[i].ub.c[2]);
            open_tab[i] = tmp;
        }
    }




    ViewHost_t v_G("GetView", sp[0], sp[1], sp[2]);
    ViewHost_t v_tmp("TmpView", sp[0], sp[1], sp[2]);


    std::ofstream log;
    double* avg_read = nullptr;
    double total_avg = 0;

    if(rank == 0) {
        avg_read = (double*) malloc(sizeof(double)*timesteps);
        log.open("test_reader.log", std::ofstream::out | std::ofstream::trunc);
        log << "step\tread_gs" << std::endl;
    }

    for(int ts=1; ts<=timesteps; ts++) {

        Timer timer_read;

        timer_read.start();

        for(int i=0; i<open_tab.size(); i++) {
            std::string filename = open_tab[i] + "_t" + std::to_string(ts) + ".hdf";

            ViewStaging_t v_S(filename, src_sp[0], src_sp[1], src_sp[2]);

            Kokkos::deep_copy(v_tmp, v_S);

            struct bbox tmp_bbox;

            bbox_intersect(&local_bb, &src_bbox_tab[i], &tmp_bbox);
            

            Kokkos::parallel_for(sp[0], KOKKOS_LAMBDA(const int i0) {
                for(int i1=0; i1<sp[1]; i1++) {
                    for(int i2=0; i2<sp[2]; i2++) {
                        v_G(i0+tmp_bbox.lb.c[0]-local_bb.lb.c[0],
                            i1+tmp_bbox.lb.c[1]-local_bb.lb.c[1],
                            i2+tmp_bbox.lb.c[2]-local_bb.lb.c[2]) = v_tmp(i0+tmp_bbox.lb.c[0]-src_bbox_tab[i].lb.c[0],
                                                                        i1+tmp_bbox.lb.c[0]-src_bbox_tab[i].lb.c[1],
                                                                        i2+tmp_bbox.lb.c[0]-src_bbox_tab[i].lb.c[2]);
                    }
                }
            });
        }

        double time_read = timer_read.stop();

        Kokkos::fence();

        double *avg_time_read = nullptr;

        if(rank == 0) {
            avg_time_read = (double*) malloc(sizeof(double)*nprocs);
        }

        MPI_Gather(&time_read, 1, MPI_DOUBLE, avg_time_read, 1, MPI_DOUBLE, 0, gcomm);

        if(rank == 0) {
            for(int i=0; i<nprocs; i++) {
                avg_read[ts-1] += avg_time_read[i];
            }
            avg_read[ts-1] /= nprocs;
            log << ts << "\t" << avg_read[ts-1] << "\t" << std::endl;
            total_avg += avg_read[ts-1];
            free(avg_time_read);
        }
    }

    free(off);
    free(lb);
    free(ub);
    free(src_bbox_tab);
    free(avg_read);

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

#endif // TEST_READER_HPP