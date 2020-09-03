#include <mpi.h>

#include <vector>

#include "analysis.h"

Analysis::Analysis(const Settings &settings, MPI_Comm comm)
    : settings(settings), comm(comm)
{

}

Analysis::~Analysis() {}

void Analysis::init()
{
    init_mpi();
    init_vector();
}

void Analysis::init_pdf()
{
    init_mpi_pdf();
    init_vector();
}

const std::vector<double> &Analysis::u_noghost() const { return u; }

const std::vector<double> &Analysis::v_noghost() const { return v; }

void Analysis::fillU(double *array)
{
    for (int z=0; z < size_z; z++) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                u[x+size_x*y+size_x*size_y*z] = array[x+size_x*y+size_x*size_y*z];
            }
        }  
    }
}

void Analysis::fillV(double *array)
{
    for (int z=0; z < size_z; z++) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                v[x+size_x*y+size_x*size_y*z] = array[x+size_x*y+size_x*size_y*z];
            }
        }  
    }
}

void Analysis::init_mpi()
{
    int dims[3] = {};
    const int periods[3] = {1, 1, 1};
    int coords[3] = {};

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    MPI_Dims_create(procs, 3, dims);
    npx = dims[0];
    npy = dims[1];
    npz = dims[2];

    MPI_Cart_create(comm, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    px = coords[0];
    py = coords[1];
    pz = coords[2];

    size_x = (settings.L + npx - 1) / npx;
    size_y = (settings.L + npy - 1) / npy;
    size_z = (settings.L + npz - 1) / npz;

    offset_x = size_x * px;
    offset_y = size_y * py;
    offset_z = size_z * pz;

    if (px == npx - 1) {
        size_x -= size_x * npx - settings.L;
    }
    if (py == npy - 1) {
        size_y -= size_y * npy - settings.L;
    }
    if (pz == npz - 1) {
        size_z -= size_z * npz - settings.L;
    }

    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &down, &up);
    MPI_Cart_shift(cart_comm, 2, 1, &south, &north);

    // XY faces: size_x * (size_y + 2)
    MPI_Type_vector(size_y + 2, size_x, size_x + 2, MPI_DOUBLE, &xy_face_type);
    MPI_Type_commit(&xy_face_type);

    // XZ faces: size_x * size_z
    MPI_Type_vector(size_z, size_x, (size_x + 2) * (size_y + 2), MPI_DOUBLE,
                    &xz_face_type);
    MPI_Type_commit(&xz_face_type);

    // YZ faces: (size_y + 2) * (size_z + 2)
    MPI_Type_vector((size_y + 2) * (size_z + 2), 1, size_x + 2, MPI_DOUBLE,
                    &yz_face_type);
    MPI_Type_commit(&yz_face_type);
}

void Analysis::init_mpi_pdf()
{
    int dims[3] = {};
    const int periods[3] = {1, 1, 1};
    int coords[3] = {};

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &procs);

    npx = procs;
    npy = 1;
    npz = 1;

    

    px = rank;
    py = 0;
    pz = 0;

    size_x = settings.L / npx;
    size_y = settings.L;
    size_z = settings.L;

    offset_x = size_x * px;
    offset_y = 0;
    offset_z = 0;


}


void Analysis::init_vector()
{
    const int V = size_x * size_y * size_z ;
    u.resize(V, 0.0);
    v.resize(V, 0.0);
}