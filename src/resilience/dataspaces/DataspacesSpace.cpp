#include "Kokkos_Core.hpp"
#include "DataspacesSpace.hpp"
#include "Kokkos_Macros.hpp"
#include "mpi.h"
#include <iostream>

namespace KokkosResilience {

    KokkosDataspacesConfigurationManager::OperationPrimitive * KokkosDataspacesConfigurationManager::resolve_variable(
                      std::string data, std::map<const std::string, size_t> & var_map ) {
   
      size_t val = var_map[data];
      //printf("variable [%s] returned %d\n", data.c_str(), val);
      return new KokkosDataspacesConfigurationManager::OperationPrimitive(val);
    }

    KokkosDataspacesConfigurationManager::OperationPrimitive *
    KokkosDataspacesConfigurationManager::resolve_arithmetic( std::string data,
                                                            std::map<const std::string, size_t> & var_map ) {
      KokkosDataspacesConfigurationManager::OperationPrimitive * lhs_op = nullptr;
      for ( size_t n = 0; n< data.length(); ) {
         //printf(" resolve_arithmetic: %d \n", n);
         char c = data.at(n);
         if ( c >= 0x30 && c <= 0x39 ) {
            std::string cur_num = "";
            while ( c >= 0x30 && c <= 0x39 ) {
               cur_num += data.substr(n,1); n++;
               if (n < data.length()) {
                  c = data.at(n);
               } else {
                  break;
               }
            }
            lhs_op = new KokkosDataspacesConfigurationManager::OperationPrimitive((std::atoi(cur_num.c_str())));
         } else if (c == '(') {
            size_t end_pos = data.find_first_of(')',n);
            if (end_pos >= 0 && end_pos < data.length()) {
              // printf("calling resolving arithmetic: %s \n", data.substr(n+1,end_pos-n-1).c_str());
               lhs_op = resolve_arithmetic ( data.substr(n+1,end_pos-n-1), var_map );
            } else {
               printf("syntax error in arithmetic: %s, at %d \n", data.c_str(), n);
               return nullptr;
            }
            n = end_pos+1;
         } else if (c == '{') {
            size_t end_pos = data.find_first_of('}',n);
            if (end_pos >= 0 && end_pos < data.length()) {
              //  printf("resolving variable: %s \n", data.substr(n+1,end_pos-n-1).c_str());
               lhs_op = resolve_variable( data.substr(n+1,end_pos-n-1), var_map );
            } else {
               printf("syntax error in variable name: %s, at %d \n", data.c_str(), n);
               return nullptr;
            }
            n = end_pos+1;
         } else {
          //  printf("parsing rhs: %s \n", data.substr(n,1).c_str());
            KokkosDataspacesConfigurationManager::OperationPrimitive * op =
                        KokkosDataspacesConfigurationManager::OperationPrimitive::parse_operator(data.substr(n,1),lhs_op);
            n++;
            op->set_right_opp( resolve_arithmetic ( data.substr(n), var_map ) );
            return op;
         }
      
      }
      return lhs_op;
   
    }

    void KokkosDataspacesConfigurationManager::set_param_list( boost::property_tree::ptree l_config, int data_scope,
                       std::string param_name, uint64_t output [], std::map<const std::string, size_t> & var_map ) {
    //  printf("set_param_list: %s \n", param_name.c_str());
      for ( auto & param_list : l_config ) {
          if ( param_list.first == param_name ) {
              int n = 0;
              for (auto & param : param_list.second ) {
                // printf("processing param list: %s\n", param.second.get_value<std::string>().c_str());
                 KokkosDataspacesConfigurationManager::OperationPrimitive * opp = resolve_arithmetic( param.second.get_value<std::string>(), var_map );
                 if (opp != nullptr) {
                     output[n++] = opp->evaluate();
                     delete opp;
                 } else {
                     output[n++] = 0;
                 }
              }
            //  printf("param_list resolved [%s]: %d,%d,%d,%d \n", param_name.c_str(), output[0], output[1], output[2], output[3]);
              break;
          }
      }
    }



    int KokkosDataspacesAccessor::initialize( const size_t size_, const std::string & filepath, const size_t version_) {
        data_size = size_;
        file_path = filepath;
        version = version_;
        ub[0] = data_size-1;
        time_t rawtime;
        time(&rawtime);
        std::string time_str (ctime(&rawtime));
        std::string pid_str = std::to_string((int)getpid());
        std::hash<std::string> str_hash;
        appid = str_hash(time_str+pid_str) % std::numeric_limits<int>::max();
        MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank);
        MPI_Bcast(&appid, 1, MPI_INT, 0, gcomm);
        MPI_Barrier(gcomm);
        dspaces_init(mpi_size, appid, &gcomm, NULL); 
        return 0;
    }

    int KokkosDataspacesAccessor::initialize(const size_t size_, const std::string & filepath, const size_t version_,
                                                KokkosDataspacesConfigurationManager config_) {
        data_size = size_;
        file_path = filepath;
        version = version_;
        for (int i = 0; i < 4; i++) {
            lb[i] = 0;
            ub[i] = 0;
        }

        boost::property_tree::ptree l_config = config_.get_config()->get_child("Layout_Config");
        std::map<const std::string, size_t> var_list;
        time_t rawtime;
        time(&rawtime);
        std::string time_str (ctime(&rawtime));
        std::string pid_str = std::to_string((int)getpid());
        std::hash<std::string> str_hash;
        appid = str_hash(time_str+pid_str) % std::numeric_limits<int>::max();
        MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank);
        MPI_Bcast(&appid, 1, MPI_INT, 0, gcomm);
        MPI_Barrier(gcomm);

        var_list["DATA_SIZE"] = data_size;
        var_list["MPI_SIZE"] = (size_t)mpi_size;
        var_list["MPI_RANK"] = (size_t)mpi_rank;
        rank = l_config.get<int>("rank");
        elem_size = l_config.get<int>("element_size");
        config_.set_param_list( l_config, 0, "lower_bbox", lb, var_list);
        config_.set_param_list( l_config, 0, "upper_bbox", ub, var_list);

        m_layout = config_.get_layout();
        m_is_initialized = true;
        dspaces_init(mpi_size, appid, &gcomm, NULL); 

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
        return 1;
    }

    size_t KokkosDataspacesAccessor::ReadFile_impl(void * dest, const size_t dest_size) {
        size_t dataRead = 0;
        char* ptr = (char*)dest;
        if (open_file(KokkosIOAccessor::READ_FILE)) {
            std::string sFullPath = KokkosIOAccessor::resolve_path( file_path, KokkosResilience::DataspacesSpace::s_default_path );
            //size_t lb[1] = {0}, ub[1] = {dest_size-1};
            dspaces_lock_on_read(sFullPath.c_str(), &gcomm);
            int err = dspaces_get(sFullPath.c_str(), version, elem_size, rank, lb, ub, dest);
            dspaces_unlock_on_read(sFullPath.c_str(), &gcomm);
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
            //size_t lb[1] = {0}, ub[1] = {src_size-1};
            dspaces_lock_on_write(sFullPath.c_str(), &gcomm);
            int err = dspaces_put(sFullPath.c_str(), version, elem_size, rank, lb, ub, src);
            /* enable if counting for exact time to put data to server */
            //dspaces_put_sync();
            dspaces_unlock_on_write(sFullPath.c_str(), &gcomm);
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
        
    }

    void KokkosDataspacesAccessor::finalize() {
      close_file();
      atexit(dspaces_finalize);
    }

    std::string DataspacesSpace::s_default_path = "./";

    DataspacesSpace::DataspacesSpace() {
        
    }

    std::map<const std::string, KokkosDataspacesAccessor> DataspacesSpace::m_accessor_map;

    // Need Reconsideration

    /**\brief  Allocate untracked memory in the space */
    void * DataspacesSpace::allocate( const size_t arg_alloc_size, const std::string & path ) const {

        KokkosDataspacesAccessor acc = m_accessor_map[path];
        if(!acc.is_initialized() ) {
            // TODO: use boost::ptree to provide a constructor with config manager
            size_t timestep = 0;
            std::string path_prefix = KokkosIOAccessor::get_timestep(path, timestep);
            boost::property_tree::ptree pConfig;
            if (path_prefix.compare("") == 0) {
                pConfig = KokkosIOConfigurationManager::get_instance()->get_config(path);
                if (pConfig.size() > 0) {
                    acc.initialize(arg_alloc_size, path, timestep, KokkosDataspacesConfigurationManager (pConfig) );
                }
                else {
                    acc.initialize(arg_alloc_size, path, timestep);
                }
            } else {
                pConfig = KokkosIOConfigurationManager::get_instance()->get_config(path_prefix);
                if (pConfig.size() > 0) {
                    acc.initialize(arg_alloc_size, path_prefix, timestep, KokkosDataspacesConfigurationManager (pConfig) );
                }
                else {
                    acc.initialize(arg_alloc_size, path, timestep);
                }

            }
            

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
            Kokkos::Profiling::SpaceHandle(KokkosResilience::DataspacesSpace::name()),RecordBase::m_alloc_ptr->m_label,
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