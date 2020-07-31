find_package(MPI)

find_path(_dataspaces_root
          NAMES include/dataspaces.h
          HINTS $ENV{SEMS_DATASPACES_ROOT} $ENV{DATASPACES_ROOT} ${DATASPACES_ROOT} ${DATASPACES_DIR}
          )

find_library(_dataspaces_lib
             NAMES libdspaces.a
             HINTS ${_dataspaces_root}/lib ${_dataspaces_root}/lib64)

find_library(_dart_lib
             NAMES libdart.a
             HINTS ${_dataspaces_root}/lib ${_dataspaces_root}/lib64)

find_library(_dscommon_lib
             NAMES libdscommon.a
             HINTS ${_dataspaces_root}/lib ${_dataspaces_root}/lib64)

find_path(_dataspaces_include_dir
          NAMES dataspaces.h
          HINTS ${_dataspaces_root}/include)

find_library(_ds_m_lib m)

find_library(_ds_rt_lib rt)

find_library(_ds_ibverbs_lib ibverbs)

find_library(_ds_rdmacm_lib rdmacm)

if ((NOT ${_dataspaces_root})
        OR (NOT ${_dataspaces_lib})
        OR (NOT ${_dart_lib})
        OR (NOT ${_dscommon_lib})
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
                                  _dart_lib
                                  _dscommon_lib
                                  _dataspaces_include_dir
                                  MPI_FOUND
                                  MPI_CXX_FOUND
                                  )

add_library(Dataspaces::Common UNKNOWN IMPORTED)
set_target_properties(Dataspaces::Common PROPERTIES
                      IMPORTED_LOCATION ${_dscommon_lib}
                      )

add_library(Dataspaces::Dart UNKNOWN IMPORTED)
set_target_properties(Dataspaces::Dart PROPERTIES
                      IMPORTED_LOCATION ${_dart_lib}
                      )

add_library(Dataspaces::Dataspaces UNKNOWN IMPORTED)
set_target_properties(Dataspaces::Dataspaces PROPERTIES
                      IMPORTED_LOCATION ${_dataspaces_lib} ${_dart_lib} ${_dscommon_lib}
                      INTERFACE_INCLUDE_DIRECTORIES ${_dataspaces_include_dir}
                      INTERFACE_LINK_LIBRARIES "Dataspaces::Dart;Dataspaces::Common"
                      )

target_link_libraries(Dataspaces::Dataspaces 
                      PUBLIC ${_ds_m_lib}
                      PUBLIC ${_ds_rt_lib}
                      PUBLIC ${_ds_ibverbs_lib}
                      PUBLIC ${_ds_rdmacm_lib}
                      )

set(DATASPACES_DIR ${_dataspaces_root})

mark_as_advanced(
  _dataspaces_library
  _dataspaces_include_dir
)

message(STATUS "DS_LIB=${_dataspaces_library}")
