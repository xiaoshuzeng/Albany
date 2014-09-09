//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Topology.h"
#include "Topology_FractureCriterion.h"

namespace LCM{

FractureCriterionTraction::FractureCriterionTraction(
    Topology & topology,
    std::string const & bulk_block_name,
    std::string const & interface_block_name,
    std::string const & stress_name,
    double const critical_traction,
    double const beta) :
AbstractFractureCriterion(topology, bulk_block_name, interface_block_name),
stress_field_(*(getMetaData().get_field<TensorFieldType>(stk::topology::NODE_RANK, stress_name))),
critical_traction_(critical_traction),
beta_(beta)
{
  if (&stress_field_ == 0) {
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "Cannot find field for traction criterion: ";
    std::cerr << stress_name;
    std::cerr << '\n';
    exit(1);
  }
  computeNormals();
}


bool
FractureCriterionTraction::check(
    stk::mesh::BulkData & bulk_data,
    stk::mesh::Entity interface)
{
  // Check the adjacent bulk elements. Proceed only
  // if both elements belong to the bulk part.
  stk::mesh::Entity const *
  relations_up = bulk_data.begin(
      interface,
      (stk::mesh::EntityRank)(bulk_data.entity_rank(interface) + 1)
  );

  assert(bulk_data.num_connectivity(
          interface,
          (stk::mesh::EntityRank)(bulk_data.entity_rank(interface) + 1)
         ) == 2);

  stk::mesh::Entity
  element_0 = relations_up[0];

  stk::mesh::Entity
  element_1 = relations_up[1];

  stk::mesh::Bucket const &
  bucket_0 = bulk_data.bucket(element_0);

  stk::mesh::Bucket const &
  bucket_1 = bulk_data.bucket(element_1);

  bool const
  is_embedded =
      bucket_0.member(getBulkPart()) && bucket_1.member(getBulkPart());

  if (is_embedded == false) return false;

  // Now traction check
  stk::mesh::EntityVector
  nodes = getTopology().getBoundaryEntityNodes(interface);

  EntityVectorIndex const
  number_nodes = nodes.size();

  Intrepid::Tensor<double>
  stress(getDimension(), Intrepid::ZEROS);

  Intrepid::Tensor<double>
  nodal_stresses(number_nodes);

  nodal_stresses.set_dimension(getDimension());

  // The traction is evaluated at centroid of face, so a simple
  // average yields the value.
  for (EntityVectorIndex i = 0; i < number_nodes; ++i) {

    stk::mesh::Entity
    node = nodes[i];

    double * const
    pstress = stk::mesh::field_data(stress_field_, node);

    nodal_stresses.fill(pstress);

    stress += nodal_stresses;
  }

  stress = stress / static_cast<double>(number_nodes);

  Intrepid::Index const
  face_index = bulk_data.identifier(interface) - 1;

  Intrepid::Vector<double> const &
  normal = normals_[face_index];

  Intrepid::Vector<double> const
  traction = stress * normal;

  double
  t_n = Intrepid::dot(traction, normal);

  Intrepid::Vector<double> const
  traction_normal = t_n * normal;

  Intrepid::Vector<double> const
  traction_shear = traction - traction_normal;

  double const
  t_s = Intrepid::norm(traction_shear);

  // Ignore compression
  t_n = std::max(t_n, 0.0);

  double const
  effective_traction = std::sqrt(t_s * t_s / beta_ / beta_ + t_n * t_n);

  return effective_traction >= critical_traction_;
}

void
FractureCriterionTraction::computeNormals()
{
  stk::mesh::Selector
  local_selector = getMetaData().locally_owned_part();

  std::vector<stk::mesh::Bucket*> const &
  node_buckets = getBulkData().buckets(stk::topology::NODE_RANK);

  stk::mesh::EntityVector
  nodes;

  stk::mesh::get_selected_entities(local_selector, node_buckets, nodes);

  EntityVectorIndex const
  number_nodes = nodes.size();

  std::vector<Intrepid::Vector<double> >
  coordinates(number_nodes);

  Teuchos::ArrayRCP<double> &
  node_coordinates = getSTKDiscretization().getCoordinates();

  for (EntityVectorIndex i = 0; i < number_nodes; ++i) {

    double const * const
    pointer_coordinates = &(node_coordinates[getDimension() * i]);

    coordinates[i].set_dimension(getDimension());
    coordinates[i].fill(pointer_coordinates);

  }

  std::vector<stk::mesh::Bucket*> const &
    face_buckets = bulk_data_.buckets(getMetaData().side_rank());

  stk::mesh::EntityVector
  faces;

  stk::mesh::get_selected_entities(local_selector, face_buckets, faces);

  EntityVectorIndex const
  number_normals = faces.size();

  normals_.resize(number_normals);

  for (EntityVectorIndex i = 0; i < number_normals; ++i) {

    stk::mesh::Entity
    face = faces[i];

    stk::mesh::EntityVector
    nodes = getTopology().getBoundaryEntityNodes(face);

    Intrepid::Vector<double> &
    normal = normals_[i];

    normal.set_dimension(getDimension());

    // Depending on the dimension is how the normal is computed.
    // TODO: generalize this for all topologies.
    switch (getDimension()) {

    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__ << '\n';
      std::cerr << "Wrong dimension: " << getDimension() << '\n';
      exit(1);
      break;

    case 2:
      {
        int const
        gid0 = bulk_data_.identifier(nodes[0]) - 1;

        Intrepid::Index const
        lid0 = getSTKDiscretization().getNodeMap()->LID(gid0);

        assert(lid0 < number_nodes);

        int const
        gid1 = bulk_data_.identifier(nodes[1]) - 1;

        Intrepid::Index const
        lid1 = getSTKDiscretization().getNodeMap()->LID(gid1);

        assert(lid1 < number_nodes);

        Intrepid::Vector<double>
        v = coordinates[lid1] - coordinates[lid0];

        normal(0) = -v(1);
        normal(1) = v(0);

        normal = Intrepid::unit(normal);
      }
      break;

    case 3:
      {
        int const
        gid0 = bulk_data_.identifier(nodes[0]) - 1;

        Intrepid::Index const
        lid0 = getSTKDiscretization().getNodeMap()->LID(gid0);

        assert(lid0 < number_nodes);

        int const
        gid1 = bulk_data_.identifier(nodes[1]) - 1;

        Intrepid::Index const
        lid1 = getSTKDiscretization().getNodeMap()->LID(gid1);

        assert(lid1 < number_nodes);

        int const
        gid2 = bulk_data_.identifier(nodes[2]) - 1;

        Intrepid::Index const
        lid2 = getSTKDiscretization().getNodeMap()->LID(gid2);

        assert(lid2 < number_nodes);

        Intrepid::Vector<double>
        v1 = coordinates[lid1] - coordinates[lid0];

        Intrepid::Vector<double>
        v2 = coordinates[lid2] - coordinates[lid0];

        normal = Intrepid::cross(v1, v2);

        normal = Intrepid::unit(normal);
      }
      break;

    }

  }

}

} // namespace LCM

