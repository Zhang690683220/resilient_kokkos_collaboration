#ifndef INC_RESILIENCE_DATASPACES_DATASPACESSPACE_HPP
#define INC_RESILIENCE_DATASPACES_DATASPACESSPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>
#include <ctime>
#include <limits>
#include <unistd.h>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <resilience/config/Config.hpp>
#include "resilience/filesystem/ExternalIOInterface.hpp"
#include "mpi.h"
#include "dataspaces.h"

namespace KokkosResilience {

class KokkosDataspacesConfigurationManager {

public:
    class OperationPrimitive {
        public:
            enum { OPP_INVALID = 0,
                    OPP_ADD = 1,
                    OPP_SUB = 2,
                    OPP_DIV = 3,
                    OPP_MUL = 4,
                    OPP_MOD = 5 };
            int type;
            size_t val;
            int operation;
            OperationPrimitive * m_left;
            OperationPrimitive * m_right;

            int which() { return type; }

            size_t get_val() { return val; }
            int get_opp() { return operation; }

            OperationPrimitive(  ) : type(0), val(0), operation(0), m_left(nullptr), m_right(nullptr) {}
            OperationPrimitive( size_t val_ ) : type(1), val(val_), operation(0), m_left(nullptr), m_right(nullptr) {}
            OperationPrimitive( int op_, OperationPrimitive * left_ ) : type(2), val(-1), operation(op_), m_left(left_)  {}

            OperationPrimitive( const OperationPrimitive & rhs ) = default;
            OperationPrimitive( OperationPrimitive && rhs ) = default;
            OperationPrimitive & operator = ( OperationPrimitive && ) = default;
            OperationPrimitive & operator = ( const OperationPrimitive & ) = default;

            OperationPrimitive( OperationPrimitive * ptr_ ) : type(ptr_->type), val(ptr_->val), operation(ptr_->operation),
                                                              m_left(ptr_->m_left), m_right(ptr_->m_right) {}

            ~OperationPrimitive() {
                if (m_left != nullptr) delete m_left;
                if (m_right != nullptr) delete m_right;
            }

            void set_right_opp( OperationPrimitive * rhs) {
                m_right = rhs;
            }

            size_t evaluate () {
                switch ( which() ) {
                    case 0:
                        return 0;
                    case 1:
                        return val;
                    case 2:
                        return per_opp( m_left != nullptr ? m_left->evaluate() : 0 ,
                                        m_right != nullptr ? m_right->evaluate() : 0 );
                    default:
                        return 0;
                }
                return 0;
            }

            size_t per_opp ( size_t left, size_t right ) {
                switch ( get_opp() ) {
                    case 0:
                        return 0;
                    case 1:
                        return left + right;
                    case 2:
                        return left - right;
                    case 3:
                        return left / right;
                    case 4:
                        return left * right;
                    case 5:
                        return left % right;
                    default:
                        return 0;
                }
                return 0;
            }

            static OperationPrimitive * parse_operator( const std::string & val, OperationPrimitive * left ) {
                if ( val == "+" ) {
                    return new OperationPrimitive(1,left);
                } else if ( val == "-" ) {
                    return new OperationPrimitive(2,left);
                } else if ( val == "/" ) {
                    return new OperationPrimitive(3,left);
                } else if ( val == "*" ) {
                    return new OperationPrimitive(4,left);
                } else if ( val == "%" ) {
                    return new OperationPrimitive(5,left);
                }
                    return new OperationPrimitive(0,left);
            }
    };

    enum { LAYOUT_DEFAULT = 0,
           LAYOUT_REGULAR = 1 };

    int m_layout;
    boost::property_tree::ptree m_config;

    OperationPrimitive * resolve_variable( std::string data, std::map<const std::string, size_t> & var_map );
    OperationPrimitive * resolve_arithmetic( std::string data, std::map<const std::string, size_t> & var_map );

    int get_layout() { return m_layout; }
    boost::property_tree::ptree * get_config() { return &m_config; }

    void set_param_list( boost::property_tree::ptree l_config, int data_scope, std::string var_name,
                            uint64_t var [], std::map<const std::string, size_t> & var_map );

    KokkosDataspacesConfigurationManager( const boost::property_tree::ptree & config_ ) : m_config(config_) {
        boost::property_tree::ptree l_config = config_.get_child("Layout_Config");
        std::string sLayout  = l_config.get<std::string>("layout");
        if (sLayout == "REGULAR") {
            m_layout = LAYOUT_REGULAR;
        } else {
            m_layout = LAYOUT_DEFAULT;
        }
    }
};


class KokkosDataspacesAccessor : public KokkosIOAccessor {

public:

    size_t rank;            // rank of the dataset (number of dimensions)
    size_t version;         // version of the dataset
    int appid;              // dataspaces client handle
    int elem_size;          // size of single element size, e.g. sizeof(double)
    uint64_t lb[4];         // coordinates for the lower corner of the local bounding box.
    uint64_t ub[4];         // coordinates for the upper corner of the local bounding box.

    int mpi_size;
    int mpi_rank;
    MPI_Comm gcomm;
    int m_layout;
    bool m_is_initialized;


    KokkosDataspacesAccessor() : KokkosIOAccessor(),
                                 rank(1),
                                 version(0),
                                 appid(0),
                                 elem_size(1),
                                 lb{0,0,0,0},
                                 ub{0,0,0,0},
                                 mpi_size(1),
                                 mpi_rank(0),
                                 gcomm(MPI_COMM_WORLD),
                                 m_layout(KokkosDataspacesConfigurationManager::LAYOUT_DEFAULT),
                                 m_is_initialized(false) {}
    KokkosDataspacesAccessor(const size_t size, const std::string & path ) : KokkosIOAccessor(size, path, true),
                                                                             rank(1),
                                                                             version(0),
                                                                             appid(0),
                                                                             elem_size(1),
                                                                             lb{0,0,0,0},
                                                                             ub{0,0,0,0},
                                                                             mpi_size(1),
                                                                             mpi_rank(0),
                                                                             gcomm(MPI_COMM_WORLD),
                                                                             m_layout(KokkosDataspacesConfigurationManager::LAYOUT_DEFAULT),
                                                                             m_is_initialized(false) {}
    KokkosDataspacesAccessor( const KokkosDataspacesAccessor & rhs ) = default;
    KokkosDataspacesAccessor( KokkosDataspacesAccessor && rhs ) = default;
    KokkosDataspacesAccessor & operator = ( KokkosDataspacesAccessor && ) = default;
    KokkosDataspacesAccessor & operator = ( const KokkosDataspacesAccessor & ) = default;
    KokkosDataspacesAccessor( const KokkosDataspacesAccessor & cp_, const size_t size) {
        data_size = size;
        file_path = cp_.file_path;
        rank = cp_.rank;
        version = cp_.version;
        appid = cp_.appid;
        elem_size = cp_.elem_size;
        mpi_size = cp_.mpi_size;
        mpi_rank = cp_.mpi_rank;
        gcomm = cp_.gcomm;
        m_layout = cp_.m_layout;
        for(int i = 0; i < 4; i++) {
            lb[i] = cp_.lb[i];
            ub[i] = cp_.ub[i];
        }

        // need to re-initialize
        if(data_size != cp_.data_size) {
            if(m_layout == KokkosDataspacesConfigurationManager::LAYOUT_DEFAULT) {
                initialize(size, file_path, version);
            } else {
                initialize(size, file_path, version, KokkosDataspacesConfigurationManager(
                                    KokkosIOConfigurationManager::get_instance()->get_config(file_path) ) );
            }
        }
        m_is_initialized = true;
    }

    //TODO: Construction Function with other size

    
    int initialize( const size_t size_, 
                    const std::string &filepath,
                    const size_t version_);

    int initialize( const size_t size_, 
                    const std::string & filepath,
                    const size_t version_, 
                    KokkosDataspacesConfigurationManager config_);

    int open_file(int read_write);
    void close_file();
    bool is_initialized() { return m_is_initialized; }
    

    virtual size_t ReadFile_impl(void * dest, const size_t dest_size);
   
    virtual size_t WriteFile_impl(const void * src, const size_t src_size);
   
    virtual size_t OpenFile_impl();

    void finalize();
   
    virtual ~KokkosDataspacesAccessor() {
    }

};


/// \class DataspacesSpace
/// \brief Memory management for Dataspaces
///
/// DataspacesSpace is a memory space that governs access to Dataspaces data.
///
class DataspacesSpace {
public:
    //! Tag this class as a kokkos memory space
    typedef KokkosResilience::DataspacesSpace  file_space;   // used to uniquely identify file spaces
    typedef KokkosResilience::DataspacesSpace  memory_space;
    typedef size_t     size_type;

    /// \typedef execution_space
    /// \brief Default execution space for this memory space.
    ///
    /// Every memory space has a default execution space.  This is
    /// useful for things like initializing a View (which happens in
    /// parallel using the View's default execution space).
#if defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP )
    typedef Kokkos::OpenMP    execution_space;
#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS )
    typedef Kokkos::Threads   execution_space;
//#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined( KOKKOS_ENABLE_OPENMP )
    typedef Kokkos::OpenMP    execution_space;
#elif defined( KOKKOS_ENABLE_THREADS )
    typedef Kokkos::Threads   execution_space;
//#elif defined( KOKKOS_ENABLE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined( KOKKOS_ENABLE_SERIAL )
    typedef Kokkos::Serial    execution_space;
#else
#  error "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, Kokkos::Qthreads, or Kokkos::Serial.  You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF CMake option, but did not enable any of the other host execution space devices."
#endif

    //! This memory space preferred device_type
    typedef Kokkos::Device< execution_space, memory_space > device_type;

    /**\brief  Default memory space instance */
    DataspacesSpace();
    DataspacesSpace( DataspacesSpace && rhs ) = default;
    DataspacesSpace( const DataspacesSpace & rhs ) = default;
    DataspacesSpace & operator = ( DataspacesSpace && ) = default;
    DataspacesSpace & operator = ( const DataspacesSpace & ) = default;
    ~DataspacesSpace() = default;

    /**\brief  Allocate untracked memory in the space */
    void * allocate( const size_t arg_alloc_size, const std::string & path ) const;

    /**\brief  Deallocate untracked memory in the space */
    void deallocate( void * const arg_alloc_ptr
                    , const size_t arg_alloc_size ) const;

    /**\brief Return Name of the MemorySpace */
    static constexpr const char* name() { return m_name; }

    static void restore_all_views();
    static void restore_view(const std::string name);
    static void checkpoint_views();
    static void checkpoint_create_view_targets();

    static void set_default_path( const std::string path );
    static std::string s_default_path;

    static std::map<const std::string, KokkosDataspacesAccessor> m_accessor_map;

private:
    static constexpr const char* m_name = "Dataspaces";
    friend class Kokkos::Impl::SharedAllocationRecord< KokkosResilience::DataspacesSpace, void >;
};

} // namespace KokkosResilience

namespace Kokkos {

namespace Impl {

template<>
class SharedAllocationRecord< KokkosResilience::DataspacesSpace, void >
    : public SharedAllocationRecord< void, void >
{
private:
    friend KokkosResilience::DataspacesSpace;

    typedef SharedAllocationRecord< void, void >  RecordBase;

    SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
    SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

    static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this HDF5Space instance */
    static RecordBase s_root_record;
#endif

    const KokkosResilience::DataspacesSpace m_space;

protected:
    ~SharedAllocationRecord();
    SharedAllocationRecord() = default;

    SharedAllocationRecord( const KokkosResilience::DataspacesSpace        & arg_space
                            , const std::string              & arg_label
                            , const size_t                     arg_alloc_size
                            , const RecordBase::function_type  arg_dealloc = & deallocate
                            );

public:

    inline
    std::string get_label() const
    {
        return std::string( RecordBase::head()->m_label );
    }

    KOKKOS_INLINE_FUNCTION static
    SharedAllocationRecord * allocate( const KokkosResilience::DataspacesSpace &  arg_space
                                    , const std::string       &  arg_label
                                    , const size_t               arg_alloc_size
                                    )
    {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
        return new SharedAllocationRecord( arg_space, arg_label, arg_alloc_size );
#else
        return (SharedAllocationRecord *) 0;
#endif
    }

    /**\brief  Allocate tracked memory in the space */
    static
    void * allocate_tracked( const KokkosResilience::DataspacesSpace & arg_space
                            , const std::string & arg_label
                            , const size_t arg_alloc_size );

    /**\brief  Reallocate tracked memory in the space */
    static
    void * reallocate_tracked( void * const arg_alloc_ptr
                            , const size_t arg_alloc_size );

    /**\brief  Deallocate tracked memory in the space */
    static
    void deallocate_tracked( void * const arg_alloc_ptr );

    static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

    static void print_records( std::ostream &, const KokkosResilience::DataspacesSpace &, bool detail = false );
};

template<class ExecutionSpace> struct DeepCopy< KokkosResilience::DataspacesSpace , Kokkos::HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {
    KokkosResilience::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    KokkosResilience::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }
};

template<class ExecutionSpace> struct DeepCopy<  Kokkos::HostSpace , KokkosResilience::DataspacesSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {
    KokkosResilience::KokkosIOAccessor::transfer_to_host( dst, src, n );
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    KokkosResilience::KokkosIOAccessor::transfer_to_host( dst, src, n );
  }
};

} // Impl

} // Kokkos

#endif  // INC_RESILIENCE_DATASPACES_DATASPACESSPACE_HPP