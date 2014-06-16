//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_GenericSTKFieldContainer.hpp"
#include "Albany_STKNodeFieldContainer.hpp"

#include "Albany_Utils.hpp"
#include "Albany_StateInfoStruct.hpp"
#include <stk_mesh/base/GetBuckets.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

template<bool Interleaved>
Albany::GenericSTKFieldContainer<Interleaved>::GenericSTKFieldContainer(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  stk::mesh::MetaData* metaData_,
  stk::mesh::BulkData* bulkData_,
  const int neq_,
  const int numDim_)
  : metaData(metaData_),
    bulkData(bulkData_),
    params(params_),
    neq(neq_),
    numDim(numDim_) {
}

template<bool Interleaved>
Albany::GenericSTKFieldContainer<Interleaved>::~GenericSTKFieldContainer() {
}


template<bool Interleaved>
void
Albany::GenericSTKFieldContainer<Interleaved>::buildStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis){

  using namespace Albany;

  // QuadPoint fields
  // dim[0] = nCells, dim[1] = nQP, dim[2] = nVec dim[3] = nVec
  typedef typename AbstractSTKFieldContainer::QPScalarFieldType QPSFT;
  typedef typename AbstractSTKFieldContainer::QPVectorFieldType QPVFT;
  typedef typename AbstractSTKFieldContainer::QPTensorFieldType QPTFT;

  // Code to parse the vector of StateStructs and create STK fields
  for(std::size_t i = 0; i < sis->size(); i++) {
    StateStruct& st = *((*sis)[i]);
    StateStruct::FieldDims& dim = st.dim;

    if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode){

        if(dim.size() == 2){ // Scalar at QPs
          qpscalar_states.push_back(& metaData->declare_field< QPSFT >(stk::topology::ELEMENT_RANK, st.name));
          stk::mesh::put_field(*qpscalar_states.back() ,
                               metaData->universal_part(), dim[1]);
        //Debug
        //      cout << "Allocating qps field name " << qpscalar_states.back()->name() <<
        //            " size: (" << dim[0] << ", " << dim[1] << ")" <<endl;
#ifdef ALBANY_SEACAS

          if(st.output) stk::io::set_field_role(*qpscalar_states.back(), Ioss::Field::TRANSIENT);

#endif
        }
        else if(dim.size() == 3){ // Vector at QPs
          qpvector_states.push_back(& metaData->declare_field< QPVFT >(stk::topology::ELEMENT_RANK, st.name));
          // Multi-dim order is Fortran Ordering, so reversed here
          stk::mesh::put_field(*qpvector_states.back() ,
                           metaData->universal_part(), dim[2], dim[1]);
          //Debug
          //      cout << "Allocating qpv field name " << qpvector_states.back()->name() <<
          //            " size: (" << dim[0] << ", " << dim[1] << ", " << dim[2] << ")" <<endl;
#ifdef ALBANY_SEACAS

          if(st.output) stk::io::set_field_role(*qpvector_states.back(), Ioss::Field::TRANSIENT);

#endif
        }
        else if(dim.size() == 4){ // Tensor at QPs
          qptensor_states.push_back(& metaData->declare_field< QPTFT >(stk::topology::ELEMENT_RANK, st.name));
          // Multi-dim order is Fortran Ordering, so reversed here
          stk::mesh::put_field(*qptensor_states.back() ,
                           metaData->universal_part(), dim[3], dim[2], dim[1]);
          //Debug
          //      cout << "Allocating qpt field name " << qptensor_states.back()->name() <<
          //            " size: (" << dim[0] << ", " << dim[1] << ", " << dim[2] << ", " << dim[3] << ")" <<endl;
#ifdef ALBANY_SEACAS

          if(st.output) stk::io::set_field_role(*qptensor_states.back(), Ioss::Field::TRANSIENT);

#endif
        }
        // Something other than a scalar, vector, or tensor at the QPs is an error
        else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
            "Error: GenericSTKFieldContainer - cannot match QPData");
    } // end QuadPoint
    // Single scalar at center of the workset
    else if(dim.size() == 1 && st.entity == StateStruct::WorksetValue) { // A single value that applies over the entire workset (time)
      scalarValue_states.push_back(st.name);
    } // End scalar at center of element
    else if(st.entity == StateStruct::NodalData) { // Data at the node points

        const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer 
               = sis->getNodalDataBlock()->getNodeContainer();

        (*nodeContainer)[st.name] = Albany::buildSTKNodeField(st.name, dim, metaData, bulkData, st.output);
 
    } // end Node class - anything else is an error
    else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
            "Error: GenericSTKFieldContainer - cannot match unknown entity : " << st.entity << std::endl);

  }
}


template<bool Interleaved>
template<class T>
typename boost::disable_if< boost::is_same<T, Albany::AbstractSTKFieldContainer::ScalarFieldType>, void >::type
Albany::GenericSTKFieldContainer<Interleaved>::fillVectorHelper(Epetra_Vector& soln,
    T* solution_field,
    const Teuchos::RCP<Epetra_Map>& node_map,
    const stk::mesh::Bucket& bucket, int offset) {

  // Fill the result vector
  // Create a multidimensional array view of the
  // solution field data for this bucket of nodes.
  // The array is two dimensional ( Cartesian X NumberNodes )
  // and indexed by ( 0..2 , 0..NumberNodes-1 )

  double* raw_data = stk::mesh::field_data(*solution_field, bucket);

  const size_t num_vec_components  = stk::mesh::field_scalars_per_entity(*solution_field, bucket);
  const size_t num_nodes_in_bucket = bucket.size();

  for(std::size_t i = 0; i < num_nodes_in_bucket; i++)  {

    //      const unsigned node_gid = bucket[i].identifier();
    const int node_gid = bulkData->identifier(bucket[i]) - 1;
    int node_lid = node_map->LID(node_gid);

    for(std::size_t j = 0; j < num_vec_components; j++)

      soln[getDOF(node_lid, offset + j)] = raw_data[i*num_vec_components + j];

  }
}

// Specialization for ScalarFieldType

template<bool Interleaved>
void Albany::GenericSTKFieldContainer<Interleaved>::fillVectorHelper(Epetra_Vector& soln,
    ScalarFieldType* solution_field,
    const Teuchos::RCP<Epetra_Map>& node_map,
    const stk::mesh::Bucket& bucket, int offset) {

  // Fill the result vector
  // Create a multidimensional array view of the
  // solution field data for this bucket of nodes.
  // The array is two dimensional ( Cartesian X NumberNodes )
  // and indexed by ( 0..2 , 0..NumberNodes-1 )

  double* raw_data = stk::mesh::field_data(*solution_field, bucket);

  const size_t num_nodes_in_bucket = bucket.size();

  for(std::size_t i = 0; i < num_nodes_in_bucket; i++)  {

    //      const unsigned node_gid = bucket[i].identifier();
    const int node_gid = bulkData->identifier(bucket[i]) - 1;
    int node_lid = node_map->LID(node_gid);

    soln[getDOF(node_lid, offset)] = raw_data[i];

  }
}

template<bool Interleaved>
template<class T>
typename boost::disable_if< boost::is_same<T, Albany::AbstractSTKFieldContainer::ScalarFieldType>, void >::type
Albany::GenericSTKFieldContainer<Interleaved>::saveVectorHelper(const Epetra_Vector& soln,
    T* solution_field,
    const Teuchos::RCP<Epetra_Map>& node_map,
    const stk::mesh::Bucket& bucket, int offset) {

  // Fill the result vector
  // Create a multidimensional array view of the
  // solution field data for this bucket of nodes.
  // The array is two dimensional ( Cartesian X NumberNodes )
  // and indexed by ( 0..2 , 0..NumberNodes-1 )

  double* raw_data = stk::mesh::field_data(*solution_field, bucket);

  const size_t num_vec_components  = stk::mesh::field_scalars_per_entity(*solution_field, bucket);
  const size_t num_nodes_in_bucket = bucket.size();

  for(std::size_t i = 0; i < num_nodes_in_bucket; i++)  {

    const int node_gid = bulkData->identifier(bucket[i]) - 1;
    int node_lid = node_map->LID(node_gid);

    for(std::size_t j = 0; j < num_vec_components; j++)

      raw_data[i*num_vec_components + j] = soln[getDOF(node_lid, offset + j)];

  }
}

// Specialization for ScalarFieldType
template<bool Interleaved>
void Albany::GenericSTKFieldContainer<Interleaved>::saveVectorHelper(const Epetra_Vector& soln,
    ScalarFieldType* solution_field,
    const Teuchos::RCP<Epetra_Map>& node_map,
    const stk::mesh::Bucket& bucket, int offset) {

  // Fill the result vector
  // Create a multidimensional array view of the
  // solution field data for this bucket of nodes.
  // The array is two dimensional ( Cartesian X NumberNodes )
  // and indexed by ( 0..2 , 0..NumberNodes-1 )


  double* raw_data = stk::mesh::field_data(*solution_field, bucket);

  const size_t num_nodes_in_bucket = bucket.size();

  for(std::size_t i = 0; i < num_nodes_in_bucket; i++)  {

    //      const unsigned node_gid = bucket[i].identifier();
    const int node_gid = bulkData->identifier(bucket[i]) - 1;
    int node_lid = node_map->LID(node_gid);

    raw_data[i] = soln[getDOF(node_lid, offset)];

  }
}

template<bool Interleaved>
template<class T>
typename boost::disable_if< boost::is_same<T, Albany::AbstractSTKFieldContainer::ScalarFieldType>, void >::type
Albany::GenericSTKFieldContainer<Interleaved>::copySTKField(const T* source, T* target) {

  const stk::mesh::BucketVector& bv = this->bulkData->buckets(stk::topology::NODE_RANK);

  for(stk::mesh::BucketVector::const_iterator it = bv.begin() ; it != bv.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    double* raw_source = stk::mesh::field_data(*source, bucket);
    double* raw_target = stk::mesh::field_data(*target, bucket);

    const size_t num_source_components = stk::mesh::field_scalars_per_entity(*source, bucket);
    const size_t num_target_components = stk::mesh::field_scalars_per_entity(*target, bucket);
    const size_t num_nodes_in_bucket = bucket.size();

    const int uneven_downsampling = num_source_components % num_target_components;

    TEUCHOS_TEST_FOR_EXCEPTION(uneven_downsampling,
                               std::logic_error,
                               "Error in stk fields: specification of coordinate vector vs. solution layout is incorrect." 
                               << std::endl);

    for(std::size_t i = 0; i < num_nodes_in_bucket; i++) {

// In source, j varies over neq (num phys vectors * numDim)
// We want target to only vary over the first numDim components
// Not sure how to do this generally...

      for(std::size_t j = 0; j < num_target_components; j++) {

        raw_target[i*num_target_components + j] = raw_source[i*num_source_components + j];
      }
   }

  }
}

// Specialization for ScalarFieldType

template<bool Interleaved>
void Albany::GenericSTKFieldContainer<Interleaved>::copySTKField(const ScalarFieldType* source, ScalarFieldType* target) {

  const stk::mesh::BucketVector& bv = this->bulkData->buckets(stk::topology::NODE_RANK);

  for(stk::mesh::BucketVector::const_iterator it = bv.begin() ; it != bv.end() ; ++it) {

    const stk::mesh::Bucket& bucket = **it;

    double* raw_source = stk::mesh::field_data(*source, bucket);
    double* raw_target = stk::mesh::field_data(*target, bucket);

    const size_t num_nodes_in_bucket = bucket.size();

    for(std::size_t i = 0; i < num_nodes_in_bucket; i++)

      raw_target[i] = raw_source[i];

  }
}

