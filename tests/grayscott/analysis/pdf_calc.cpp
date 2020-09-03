#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "mpi.h"
#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <resilience/Resilience.hpp>
#include "../common/timer.hpp"
#include "../simulation/settings.h"
#include "analysis.h"
#include "reader.h"

bool epsilon(double d) { return (d < 1.0e-20); }
bool epsilon(float d) { return (d < 1.0e-20); }

template <class T>
void compute_pdf(const std::vector<T> &data,
                 const std::vector<std::size_t> &shape, const size_t start,
                 const size_t count, const size_t nbins, const T min,
                 const T max, std::vector<T> &pdf, std::vector<T> &bins)
{
    if (shape.size() != 3)
        throw std::invalid_argument("ERROR: shape is expected to be 3D\n");

    size_t slice_size = shape[1] * shape[2];
    pdf.resize(count * nbins);
    bins.resize(nbins);

    size_t start_data = 0;
    size_t start_pdf = 0;

    T binWidth = (max - min) / nbins;
    for (auto i = 0; i < nbins; ++i) {
        bins[i] = min + (i * binWidth);
    }

    if (nbins == 1) {
        // special case: only one bin
        for (auto i = 0; i < count; ++i) {
            pdf[i] = slice_size;
        }
        return;
    }

    if (epsilon(max - min) || epsilon(binWidth)) {
        // special case: constant array
        for (auto i = 0; i < count; ++i) {
            pdf[i * nbins + (nbins / 2)] = slice_size;
        }
        return;
    }

    for (auto i = 0; i < count; ++i) {
        // Calculate a PDF for 'nbins' bins for values between 'min' and 'max'
        // from data[ start_data .. start_data+slice_size-1 ]
        // into pdf[ start_pdf .. start_pdf+nbins-1 ]
        for (auto j = 0; j < slice_size; ++j) {
            if (data[start_data + j] > max || data[start_data + j] < min) {
                std::cout << " data[" << start * slice_size + start_data + j
                          << "] = " << data[start_data + j]
                          << " is out of [min,max] = [" << min << "," << max
                          << "]" << std::endl;
            }
            size_t bin = static_cast<size_t>(
                std::floor((data[start_data + j] - min) / binWidth));
            if (bin == nbins) {
                bin = nbins - 1;
            }
            ++pdf[start_pdf + bin];
        }
        start_pdf += nbins;
        start_data += slice_size;
    }
    return;
}

void print_settings(const Settings &s)
{
    std::cout << "grid:             " << s.L << "x" << s.L << "x" << s.L
              << std::endl;
    std::cout << "steps:            " << s.steps << std::endl;
    std::cout << "plotgap:          " << s.plotgap << std::endl;
    std::cout << "output:           " << s.output << std::endl;
}

void print_simulator_settings(const Analysis &a)
{
    std::cout << "process layout:   " << a.npx << "x" << a.npy << "x" << a.npz
              << std::endl;
    std::cout << "local grid size:  " << a.size_x << "x" << a.size_y << "x"
              << a.size_z << std::endl;
}

void printUsage()
{
    std::cout
        << "Usage: pdf_calc settings [N] [output_inputdata]\n"
        << "  settings:   Name of the json file handle for reading data\n"
        << "  N:       Number of bins for the PDF calculation, default = 1000\n"
        << "  output_inputdata: YES will write the original variables besides "
           "the analysis results\n\n";
}

template <class T>
void write_pdf(std::string &fname, const std::vector<T> &data, int rank, int timestep)
{
    std::string filename = "./output/" + fname + ".pdf.Rank." + std::to_string(rank) + ".Time." + std::to_string(timestep) + ".dat";
    std::ofstream outFile(filename.c_str(), std::ios::out | std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(&data), data.size());
    outFile.close();
}

/*
 * MAIN
 */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, comm_size, wrank;

    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    const unsigned int color = 2;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &comm);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    if (argc < 2) {
        std::cout << "Not enough arguments\n";
        if (rank == 0) printUsage();
        MPI_Finalize();
        return 0;
    }

    std::string in_filename;
    size_t nbins = 1000;
    bool write_inputvars = false;
    in_filename = argv[1];


    if (argc >= 3) {
        int value = std::stoi(argv[2]);
        if (value > 0) nbins = static_cast<size_t>(value);
    }

    if (argc >= 4) {
        std::string value = argv[3];
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        if (value == "yes") write_inputvars = true;
    }

    Settings settings = Settings::from_json(argv[1]);
    Analysis anly(settings, comm);
    anly.init_pdf();

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        print_settings(settings);
        print_simulator_settings(anly);
        std::cout << "========================================" << std::endl;
    }

    std::size_t u_global_size, v_global_size;
    std::size_t u_local_size, v_local_size;

    std::vector<std::size_t> shape(3, settings.L);



    // Calculate global and local sizes of U and V
    u_global_size = shape[0] * shape[1] * shape[2];
    u_local_size = u_global_size / comm_size;
    v_global_size = shape[0] * shape[1] * shape[2];
    v_local_size = v_global_size / comm_size;


    

    std::vector<double> u;
    std::vector<double> v;

    std::pair<double, double> minmax_u;
    std::pair<double, double> minmax_v;

    std::vector<double> pdf_u;
    std::vector<double> pdf_v;
    std::vector<double> bins_u;
    std::vector<double> bins_v;

    Kokkos::initialize( argc, argv );

    Reader dsreader(comm, comm_size, 2);

    for (int i = 0; i < settings.steps;) {

        i+=settings.plotgap;

        if (rank == 0) {
            std::cout << " reading input step     " << i / settings.plotgap
                      << "pdf_calc at step " << i / settings.plotgap 
                      << std::endl;
        }

        dsreader.kokkos_read(anly, MPI_COMM_WORLD, i);

        u = anly.u_noghost();
        v = anly.v_noghost();

        auto mmu = std::minmax_element(u.begin(), u.end());
        minmax_u = std::make_pair(*mmu.first, *mmu.second);
        auto mmv = std::minmax_element(v.begin(), v.end());
        minmax_v = std::make_pair(*mmv.first, *mmv.second);

        compute_pdf(u, shape, anly.offset_x, anly.size_x, nbins, minmax_u.first,
                    minmax_u.second, pdf_u, bins_u);

        compute_pdf(v, shape, anly.offset_x, anly.size_x, nbins, minmax_v.first,
                    minmax_v.second, pdf_v, bins_v);

        std::string Ufname ("gray_scott_u");
        std::string Vfname ("gray_scott_v");

        write_pdf(Ufname, pdf_u, rank, i);
        write_pdf(Vfname, pdf_v, rank, i);
    }

    MPI_Barrier(comm);

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

        
