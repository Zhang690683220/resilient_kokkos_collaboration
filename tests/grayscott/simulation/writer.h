#ifndef __WRITER_H__
#define __WRITER_H__

#include <mpi.h>
#include <string>
#include <vector>
#include <Kokkos_Macros.hpp>

#include "gray-scott.h"
#include "settings.h"

class Writer
{
public:
    Writer(MPI_Comm comm ,int proc, int appid) {}
    

    void kokkos_write(const GrayScott &sim, MPI_Comm comm, int step);
    //void close() { dspaces_finalize();}


};

#endif
