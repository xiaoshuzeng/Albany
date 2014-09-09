//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

///
/// Fracture criteria classes are required to have a method
/// called check that takes as argument an entity and returns a bool.
///

#if !defined(LCM_Topology_FractureCriterion_h)
#define LCM_Topology_FractureCriterion_h

#include <cassert>

#include <stk_mesh/base/FieldBase.hpp>

#include "Teuchos_ScalarTraits.hpp"
#include "Topology.h"
#include "Topology_Types.h"
#include "Topology_Utils.h"

namespace LCM{

///
/// Base class for fracture criteria
///
class AbstractFractureCriterion {

public:

  AbstractFractureCriterion(
      Topology & topology,
      std::string const & bulk_block_name,
      std::string const & interface_block_name) :
  topology_(topology),
  bulk_block_name_(bulk_block_name),
  interface_block_name_(interface_block_name),
  stk_discretization_(*(topology.getSTKDiscretization())),
  stk_mesh_struct_(*(stk_discretization_.getSTKMeshStruct())),
  bulk_data_(*(stk_mesh_struct_.bulkData)),
  meta_data_(*(stk_mesh_struct_.metaData)),
  dimension_(stk_mesh_struct_.numDim),
  bulk_part_(*(meta_data_.get_part(bulk_block_name))),
  interface_part_(*(meta_data_.get_part(interface_block_name)))
  {}


  virtual
  bool
  check(stk::mesh::BulkData & mesh, Entity interface) = 0;

  virtual
  ~AbstractFractureCriterion() {}

  Topology &
  getTopology() {return topology_;}

  std::string const &
  getBulkBlockName() {return bulk_block_name_;}

  std::string const &
  getInterfaceBlockName() {return interface_block_name_;}

  Albany::STKDiscretization &
  getSTKDiscretization() {return stk_discretization_;}

  Albany::AbstractSTKMeshStruct const &
  getAbstractSTKMeshStruct() {return stk_mesh_struct_;}

  stk::mesh::BulkData const &
  getBulkData() {return bulk_data_;}

  MetaData const &
  getMetaData() {return meta_data_;}

  Intrepid::Index
  getDimension() {return dimension_;}

  Part &
  getBulkPart() {return bulk_part_;}

  Part &
  getInterfacePart() {return interface_part_;}

protected:

  Topology &
  topology_;

  std::string
  bulk_block_name_;

  std::string
  interface_block_name_;

  Albany::STKDiscretization &
  stk_discretization_;

  Albany::AbstractSTKMeshStruct const &
  stk_mesh_struct_;

  stk::mesh::BulkData const &
  bulk_data_;

  MetaData const &
  meta_data_;

  Intrepid::Index
  dimension_;

  Part &
  bulk_part_;

  Part &
  interface_part_;

private:

  AbstractFractureCriterion();
  AbstractFractureCriterion(const AbstractFractureCriterion &);
  AbstractFractureCriterion &operator=(const AbstractFractureCriterion &);

};

///
/// Random fracture criterion given a probability of failure
///
class FractureCriterionRandom : public AbstractFractureCriterion {

public:

  FractureCriterionRandom(
      Topology & topology,
      std::string const & bulk_block_name,
      std::string const & interface_block_name,
      double const probability) :
  AbstractFractureCriterion(topology, bulk_block_name, interface_block_name),
  probability_(probability) {}

  bool
  check(stk::mesh::BulkData & bulk_data, Entity interface)
  {
    stk::mesh::EntityRank const rank = bulk_data.entity_rank(interface);

    assert(bulk_data.num_connectivity(interface, (stk::mesh::EntityRank)(rank+1)) == 2);

    double const
    random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    return random < probability_;
  }

private:

  FractureCriterionRandom();
  FractureCriterionRandom(FractureCriterionRandom const &);
  FractureCriterionRandom & operator=(FractureCriterionRandom const &);

private:

  double
  probability_;
};

///
/// Fracture criterion that open only once (for debugging)
///
class FractureCriterionOnce : public AbstractFractureCriterion {

public:

  FractureCriterionOnce(
      Topology & topology,
      std::string const & bulk_block_name,
      std::string const & interface_block_name,
      double const probability) :
  AbstractFractureCriterion(topology, bulk_block_name, interface_block_name),
  probability_(probability),
  open_(true) {}

  bool
  check(stk::mesh::BulkData& mesh, Entity interface)
  {
    stk::mesh::EntityRank const rank = mesh.entity_rank(interface);

    assert(mesh.num_connectivity(interface, (stk::mesh::EntityRank)(rank+1)) == 2);

    double const
    random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    bool const
    is_open = random < probability_ && open_;

    if (is_open == true) open_ = false;

    return is_open;
  }

private:

  FractureCriterionOnce();
  FractureCriterionOnce(FractureCriterionOnce const &);
  FractureCriterionOnce & operator=(FractureCriterionOnce const &);

private:

  double
  probability_;

  bool
  open_;
};

///
/// Traction fracture criterion
///
class FractureCriterionTraction : public AbstractFractureCriterion {

public:

  FractureCriterionTraction(
      Topology & topology,
      std::string const & bulk_block_name,
      std::string const & interface_block_name,
      std::string const & stress_name,
      double const critical_traction,
      double const beta);

  bool
  check(stk::mesh::BulkData & bulk_data, Entity interface);

private:

  FractureCriterionTraction();
  FractureCriterionTraction(FractureCriterionTraction const &);
  FractureCriterionTraction & operator=(FractureCriterionTraction const &);

  void
  computeNormals();

private:

  TensorFieldType const &
  stress_field_;

  double
  critical_traction_;

  double
  beta_;

  std::vector<Intrepid::Vector<double> >
  normals_;
};

} // namespace LCM

#endif // LCM_Topology_FractureCriterion_h
