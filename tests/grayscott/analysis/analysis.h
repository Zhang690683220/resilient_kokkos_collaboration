#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__

#include <vector>

#include <mpi.h>

#include "../simulation/settings.h"

class Analysis
{
public:
    // Dimension of process grid
    size_t npx, npy, npz;
    // Coordinate of this rank in process grid
    size_t px, py, pz;
    // Dimension of local array
    size_t size_x, size_y, size_z;
    // Offset of local array in the global array
    size_t offset_x, offset_y, offset_z;

    Analysis(const Settings &settings, MPI_Comm comm);
    ~Analysis();

    void init();
    void init_pdf();

    const std::vector<double> &u_noghost() const;
    const std::vector<double> &v_noghost() const;

    void fillU(double *array);
    void fillV(double *array);

protected:
    Settings settings;

    std::vector<double> u, v;

    int rank, procs;
    int west, east, up, down, north, south;
    MPI_Comm comm;
    MPI_Comm cart_comm;

    // MPI datatypes for halo exchange
    MPI_Datatype xy_face_type;
    MPI_Datatype xz_face_type;
    MPI_Datatype yz_face_type;

    // Setup cartesian communicator data types
    void init_mpi();
    // Setup vector
    void init_vector();

    void init_mpi_pdf();


};

#endif