#include "Kokkos_Core.hpp"
#include "DataspacesSpace.hpp"

#include "mpi.h"

namespace KokkosResilience {

    int KokkosDataspacesAccessor::initialize(const std::string & filepath) {
        file_path = filepath;
        MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank);
        dspaces_init(nprocs, 1, &gcomm, NULL); // TODO: How to define appid ?
        return 0;
    }

    size_t KokkosDataspacesAccessor::OpenFile_impl() {
        open_file(KokkosIOAccessor::WRITE_FILE);
        close_file();
        return 0;
    }

    int KokkosDataspacesAccessor::open_file(int read_write) {
        //std::string sFullPath = KokkosIOAccessor::resolve_path( file_path, KokkosResilience::DataspacesSpace::s_default_path );
        // Mismatch: Dataspaces have no concept about file handle
        // in other word, dataspaces file don't need to be open or close as long as server runs
        return 0;
    }

    size_t KokkosDataspacesAccessor::ReadFile_impl(void * dest, const size_t dest_size) {
        size_t dataRead = 0;
        char* ptr = (char*)dest;
        if (open_file(KokkosIOAccessor::READ_FILE)) {
            std::string sFullPath = KokkosIOAccessor::resolve_path( file_path, KokkosResilience::DataspacesSpace::s_default_path );
            size_t lb[1] = {0}, ub[1] = {dest_size-1};
            dspaces_lock_on_read(sFullPath.c_str(), NULL);
            int err = dspaces_get(sFullPath.c_str(), 1, 1, 1, lb, ub, dest);
            dspaces_unlock_on_read(sFullPath.c_str(), NULL);
            if(err == 0) {
                // Actual use with high-dimensional get()
                dataRead = dest_size; 
            } else {
                printf("Error with read: %d \n", err);
            }
        }
        close_file();
        return dataRead;
    }

    size_t KokkosDataspacesAccessor::WriteFile_impl(const void * src, const size_t src_size) {
        size_t m_written = 0;
        char* ptr = (char*)src;
        if (open_file(KokkosIOAccessor::WRITE_FILE)) {
            std::string sFullPath = KokkosIOAccessor::resolve_path( file_path, KokkosResilience::DataspacesSpace::s_default_path );
            size_t lb[1] = {0}, ub[1] = {src_size-1};
            dspaces_lock_on_write(sFullPath.c_str(), NULL);
            int err = dspaces_put(sFullPath.c_str(), 1, 1, 1, lb, ub, src);
            /* enable if counting for exact time to put data to server */
            //dspaces_put_sync();
            dspaces_unlock_on_write(sFullPath.c_str(), NULL);
            if(err == 0) {
                m_written = src_size;
            } else {
                printf("StdFile: write failed \n");
            }
        }
        close_file();
        return m_written;
    }

    void KokkosDataspacesAccessor::close_file() {
        /* Same as open_file() */
        return 0;
    }

    void KokkosDataspacesAccessor::finalize() {
      close_file();
      dspaces_finalize();
    }

    std::string DataspacesSpace::s_default_path = "./";

    DataspacesSpace::DataspacesSpace() {

    }

    std::map<const std::string, KokkosDataspacesAccessor> DataspacesSpace::m_accessor_map;

    // Need Reconsideration

    /**\brief  Allocate untracked memory in the space */
    void * DataspacesSpace::allocate( const size_t arg_alloc_size, const std::string & path ) const {

        KokkosDataspacesAccessor acc = m_accessor_map[path];
        if(!acc.isinitialized() ) {

            // TODO: use boost::ptree to provide a constructor with config manager
            acc.initialize(arg_alloc_size, path);
        }
        m_accessor_map[path] = acc;
        KokkosDataspacesAccessor * pAcc = new KokkosDataspacesAccessor(acc, arg_alloc_size);

        KokkosIOInterface * pInt = new KokkosIOInterface;
        pInt->pAcc = static_cast<KokkosIOAccessor*>(pAcc);
        return reinterpret_cast<void*>(pInt);

    }

    void DataspacesSpace::deallocate(void * const arg_alloc_ptr
                                    , const size_t arg_alloc_size ) const {
        const KokkosIOInterface * pInt = reinterpret_cast<KokkosIOInterface *>(arg_alloc_ptr);
        if (pInt) {
            KokkosDataspacesAccessor * pAcc = static_cast<KokkosDataspacesAccessor*>(pInt->pAcc);

            if(pAcc) {
                pAcc->finalize();
                delete pAcc;
            }
            delete pInt;
        }
    }

    void DataspacesSpace::restore_all_views() {
        typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
        Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
        while (pList != nullptr) {
            Kokkos::Impl::DeepCopy< Kokkos::HostSpace, KokkosResilience::DataspacesSpace, Kokkos::DefaultHostExecutionSpace >
                            (((base_record*)pList->src)->data(), ((base_record*)pList->dst)->data(), ((base_record*)pList->src)->size());
            if(pList->pNext == nullptr) {
                delete pList;
                pList = nullptr;
            } else {
                pList = pList->pNext;
                delete pList->pPrev;
            }
        }
    }

    void DataspacesSpace::restore_view(const std::string lbl) {
        typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
        Kokkos::Impl::MirrorTracker * pRes = base_record::get_filtered_mirror_entry( (std::string)name(), lbl );
        if (pRes != nullptr) {
            Kokkos::Impl::DeepCopy< Kokkos::HostSpace, KokkosResilience::DataspacesSpace, Kokkos::DefaultHostExecutionSpace >
                           (((base_record*)pRes->src)->data(), ((base_record*)pRes->dst)->data(), ((base_record*)pRes->src)->size());
            delete pRes;
        }
    }

    void DataspacesSpace::checkpoint_create_view_targets() {
        // ? int mpi_size = 1;
        // ? int mpi_rank = 0;

        // ? MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
        // ? MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank);
        typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
        Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
        if (pList == nullptr) {
            printf("memspace %s returned empty list of checkpoint views \n", name());
        }
        while (pList != nullptr) {
            KokkosIOAccessor::create_empty_file(((base_record*)pList->dst)->data());
            // delete the records along the way...
            if (pList->pNext == nullptr) {
                delete pList;
                pList = nullptr;
            } else {
                pList = pList->pNext;
                delete pList->pPrev;
            }
        }
    }

    void DataspacesSpace::checkpoint_views() {
        typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
        Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
        if (pList == nullptr) {
         printf("memspace %s returned empty list of checkpoint views \n", name());
        }
        while (pList != nullptr) {
            Kokkos::Impl::DeepCopy< KokkosResilience::DataspacesSpace, Kokkos::HostSpace, Kokkos::DefaultHostExecutionSpace >
                            (((base_record*)pList->dst)->data(), ((base_record*)pList->src)->data(), ((base_record*)pList->src)->size());
            // delete the records along the way...
            if (pList->pNext == nullptr) {
                delete pList;
                pList = nullptr;
            } else {
                pList = pList->pNext;
                delete pList->pPrev;
            }                
        }
    }

    void DataspacesSpace::set_default_path( const std::string path) {
        DataspacesSpace::s_default_path = path;
    }

} //namespace KokkosResilience


namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord< void , void >
SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::s_root_record ;
#endif

void
SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::
deallocate(SharedAllocationRecord< void , void > * arg_rec)
{
    delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::
~SharedAllocationRecord()
{
    #if defined(KOKKOS_ENABLE_PROFILING)
    if(Kokkos::Profiling::profileLibraryLoaded()) {
        Kokkos::Profiling::deallocateData(
            Kokkos::Profiling::SpaceHandle(KokkosResilience::HDF5Space::name()),RecordBase::m_alloc_ptr->m_label,
            data(),size());
    }
    #endif

    m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                      , SharedAllocationRecord< void , void >::m_alloc_size
                      );
}

SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::
SharedAllocationRecord( const KokkosResilience::DataspacesSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord< void , void >
        (
#ifdef KOKKOS_DEBUG
        & SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::s_root_record,
#endif
          reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( arg_alloc_size, arg_label ) )
        , arg_alloc_size
        , arg_dealloc
        )
    , m_space( arg_space )
{
#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
   }
#endif
    // Fill in the Header information
    RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

    strncpy( RecordBase::m_alloc_ptr->m_label
            , arg_label.c_str()
            , SharedAllocationHeader::maximum_label_length
            );
    // Set last element zero, in case c_str is too long
    RecordBase::m_alloc_ptr->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char) 0;
}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::
allocate_tracked( const KokkosResilience::DataspacesSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
    if ( ! arg_alloc_size ) return (void *) 0 ;

    SharedAllocationRecord * const r =
        allocate( arg_space , arg_alloc_label , arg_alloc_size );

    RecordBase::increment( r );

    return r->data();
}

void SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
    if ( arg_alloc_ptr != 0 ) {
        SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

        RecordBase::decrement( r );
    }
}

void * SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

SharedAllocationRecord< KokkosResilience::DataspacesSpace , void > *
SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< KokkosResilience::DataspacesSpace , void >::
print_records( std::ostream & s , const KokkosResilience::DataspacesSpace & , bool detail )
{
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "DataspacesSpace" , & s_root_record , detail );
#else
  throw_runtime_exception("SharedAllocationRecord<DataspacesSpace>::print_records only works with KOKKOS_DEBUG enabled");
#endif
}

} // namespace Impl
} // namespace Kokkos