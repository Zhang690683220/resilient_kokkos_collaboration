#ifndef __READER_H__
#define __READER_H__

#include <mpi.h>
#include <string>
#include <vector>
#include <Kokkos_Macros.hpp>

#include "../simulation/settings.h"
#include "analysis.h"


class Reader
{
public:
    Reader(MPI_Comm comm ,int proc, int appid) {}
    
    void kokkos_read(Analysis &analysis, MPI_Comm comm, int step);
    //void close() { dspaces_finalize();}


};

#endif
