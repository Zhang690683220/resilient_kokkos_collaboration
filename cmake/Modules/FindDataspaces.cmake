find_package(MPI)

find_path(_dataspaces_root
          NAMES include/dataspaces.h
          HINTS $ENV{SEMS_DATASPACES_ROOT} $ENV{DATASPACES_ROOT} ${DATASPACES_ROOT} ${DATASPACES_DIR}
          )

find_library(_dataspaces_lib
             NAMES libdspaces.a
             HINTS ${_dataspaces_root}/lib ${_dataspaces_root}/lib64)

find_path(_dataspaces_include_dir
          NAMES dataspaces.h
          HINTS ${_dataspaces_root}/include)

if ((NOT ${_dataspaces_root})
        OR (NOT ${_dataspaces_lib})
        OR (NOT ${_dataspaces_include_dir}))
  set(_fail_msg "Could NOT find Dataspaces (set DATASPACES_DIR or DATASPACES_ROOT to point to install)")
elseif ((NOT ${MPI_FOUND}) OR (NOT ${MPI_CXX_FOUND}))
  set(_fail_msg "Could NOT find Dataspaces (missing MPI)")
else()
  set(_fail_msg "Could NOT find Dataspaces")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Dataspaces ${_fail_msg}
                                  _dataspaces_root
                                  _dataspaces_lib
                                  _dataspaces_include_dir
                                  MPI_FOUND
                                  MPI_CXX_FOUND
                                  )

add_library(Dataspaces::Dataspaces UNKNOWN IMPORTED)
set_target_properties(Dataspaces::Dataspaces PROPERTIES
                      IMPORTED_LOCATION ${_dataspaces_lib}
                      INTERFACE_INCLUDE_DIRECTORIES ${_dataspaces_include_dir}
                      )

set(DATASPACES_DIR ${_dataspaces_root})

mark_as_advanced(
  _dataspaces_library
  _dataspaces_include_dir
)
