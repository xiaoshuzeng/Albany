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

#include <stk_mesh/base/FieldData.hpp>

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
  check(Entity const & interface) = 0;

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

  stk_classic::mesh::BulkData const &
  getBulkData() {return bulk_data_;}

  stk_classic::mesh::fem::FEMMetaData const &
  getMetaData() {return meta_data_;}

  Intrepid::Index
  getDimension() {return dimension_;}

  stk_classic::mesh::Part &
  getBulkPart() {return bulk_part_;}

  stk_classic::mesh::Part &
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

  stk_classic::mesh::BulkData const &
  bulk_data_;

  stk_classic::mesh::fem::FEMMetaData const &
  meta_data_;

  Intrepid::Index
  dimension_;

  stk_classic::mesh::Part &
  bulk_part_;

  stk_classic::mesh::Part &
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
  check(Entity const & interface)
  {
    EntityRank const
    rank = interface.entity_rank();

    stk_classic::mesh::PairIterRelation const
    relations = interface.relations(rank + 1);

    assert(relations.size() == 2);

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
  check(Entity const & interface)
  {
    EntityRank const
    rank = interface.entity_rank();

    stk_classic::mesh::PairIterRelation const
    relations = interface.relations(rank + 1);

    assert(relations.size() == 2);

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
  check(Entity const & interface);

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
