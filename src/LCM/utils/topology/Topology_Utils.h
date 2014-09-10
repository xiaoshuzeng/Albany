//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Utils_h)
#define LCM_Topology_Utils_h

#include "Topology_Types.h"

#define DEBUG_LCM_TOPOLOGY

namespace LCM {

///
/// \brief Output the mesh connectivity
///
/// Outputs the nodal connectivity of the elements as stored by
/// bulkData. Assumes that relationships between the elements and
/// nodes exist.
///
inline
void
display_connectivity(
    stk::mesh::BulkData & bulk_data,
    stk::mesh::EntityRank cell_rank)
{
  // Create a list of element entities
  stk::mesh::EntityVector
  elements;

  stk::mesh::get_entities(bulk_data, cell_rank, elements);

  typedef stk::mesh::EntityVector::size_type size_type;

  // Loop over the elements
  size_type const
  number_of_elements = elements.size();

  for (size_type i = 0; i < number_of_elements; ++i) {

    stk::mesh::Entity const* relations = bulk_data.begin_nodes(elements[i]);

    stk::mesh::EntityId const
    element_id = bulk_data.identifier(elements[i]);

    std::cout << std::setw(16) << element_id << ":";

    size_t const
    nodes_per_element = bulk_data.num_nodes(elements[i]);

    for (size_t j = 0; j < nodes_per_element; ++j) {

      stk::mesh::Entity
      node = relations[j];

      stk::mesh::EntityId const
      node_id = bulk_data.identifier(node);

      std::cout << std::setw(16) << node_id;
    }

    std::cout << '\n';
  }

  return;
}

///
/// \brief Output relations associated with entity
///        The entity may be of any rank
///
/// \param[in] entity
///
inline
void
display_relation(stk::mesh::BulkData& bulk_data, stk::mesh::Entity entity)
{
  std::cout << "Relations for entity (identifier,rank): ";
  std::cout << bulk_data.identifier(entity) << ",";
  std::cout << bulk_data.entity_rank(entity);
  std::cout << '\n';

  for (stk::topology::rank_t rank = stk::topology::NODE_RANK;
      rank <= stk::topology::ELEMENT_RANK; ++rank) {

    stk::mesh::Entity const *
    relations = bulk_data.begin(entity, rank);

    stk::mesh::ConnectivityOrdinal const *
    ords = bulk_data.begin_ordinals(entity, rank);

    size_t num_rels = bulk_data.num_connectivity(entity, rank);
    for (size_t i = 0; i < num_rels; ++i) {
      std::cout << "entity:\t";
      std::cout << bulk_data.identifier(relations[i]) << ",";
      std::cout << bulk_data.entity_rank(relations[i]);
      std::cout << "\tlocal id: ";
      std::cout << ords[i];
      std::cout << '\n';
    }
  }
  return;
}

///
/// \brief Output relations of a given rank associated with entity
///
/// \param[in] entity
/// \param[in] the rank of the entity
///
inline
void
display_relation(
    stk::mesh::BulkData & bulk_data,
    stk::mesh::Entity entity,
    stk::mesh::EntityRank const rank)
{
  std::cout << "Relations of rank ";
  std::cout << rank;
  std::cout << " for entity (identifier,rank): ";
  std::cout << bulk_data.identifier(entity) << ",";
  std::cout << bulk_data.entity_rank(entity);
  std::cout << '\n';

  stk::mesh::Entity const*
  relations = bulk_data.begin(entity, rank);

  size_t
  num_rels = bulk_data.num_connectivity(entity, rank);

  stk::mesh::ConnectivityOrdinal const*
  ords = bulk_data.begin_ordinals(entity, rank);

  for (size_t i = 0; i < num_rels; ++i) {
    std::cout << "entity:\t";
    std::cout << bulk_data.identifier(relations[i]) << ",";
    std::cout << bulk_data.entity_rank(relations[i]);
    std::cout << "\tlocal id: ";
    std::cout << ords[i];
    std::cout << '\n';
  }
  return;
}

///
/// Test whether a given source entity and relation are
/// needed in STK to maintain connectivity information.
/// These are relations that connect cells to points.
///
inline
bool
is_needed_for_stk(
    stk::mesh::BulkData & bulk_data,
    stk::mesh::Entity source_entity,
    stk::mesh::EntityRank target_rank,
    stk::mesh::EntityRank const cell_rank)
{
  stk::mesh::EntityRank const
  source_rank = bulk_data.entity_rank(source_entity);

  return (source_rank == stk::topology::ELEMENT_RANK)
      && (target_rank == stk::topology::NODE_RANK);
}

///
/// Add a dash and processor rank to a string. Useful for output
/// file names.
///
inline std::string
parallelize_string(std::string const & string)
{
  std::ostringstream
  oss;

  oss << string;

  int const
  number_processors = Teuchos::GlobalMPISession::getNProc();

  if (number_processors > 1) {

    int const
    number_digits = static_cast<int>(std::log10(number_processors));

    int const
    processor_id = Teuchos::GlobalMPISession::getRank();

    oss << "-";
    oss << std::setfill('0') << std::setw(number_digits) << processor_id;
  }

  return oss.str();
}

//
// Auxiliary for graphviz output
//
inline std::string
entity_label(stk::mesh::EntityRank const rank)
{
  std::ostringstream
  oss;

  switch (rank) {
  default:
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "stk::mesh::Entity rank is invalid: " << rank;
    std::cerr << '\n';
    exit(1);
    break;
  case stk::topology::NODE_RANK:
    oss << "Point";
    break;
  case stk::topology::EDGE_RANK:
    oss << "Segment";
    break;
  case stk::topology::FACE_RANK:
    oss << "Polygon";
    break;
  case stk::topology::ELEMENT_RANK:
    oss << "Polyhedron";
    break;
#if defined(LCM_TOPOLOGY_HIGH_DIMENSIONS)
    case 4:
    oss << "Polychoron";
    break;
    case 5:
    oss << "Polyteron";
    break;
    case 6:
    oss << "Polypeton";
    break;
#endif // LCM_TOPOLOGY_HIGH_DIMENSIONS
  }

  return oss.str();
}

//
// Auxiliary for graphviz output
//
inline std::string
entity_string(stk::mesh::BulkData & bulk_data, stk::mesh::Entity entity)
{
  std::ostringstream
  oss;

  oss << entity_label(bulk_data.entity_rank(entity)) << '-';
  oss << bulk_data.identifier(entity);

  return oss.str();
}

//
// Auxiliary for graphviz output
//
inline std::string
entity_color(
    stk::mesh::EntityRank const rank,
    FractureState const fracture_state)
{
  std::ostringstream
  oss;

  switch (fracture_state) {

  default:
    std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
    std::cerr << '\n';
    std::cerr << "Fracture state is invalid: " << fracture_state;
    std::cerr << '\n';
    exit(1);
    break;

  case CLOSED:
    switch (rank) {
    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "stk::mesh::Entity rank is invalid: " << rank;
      std::cerr << '\n';
      exit(1);
      break;
    case stk::topology::NODE_RANK:
      oss << "6";
      break;
    case stk::topology::EDGE_RANK:
      oss << "4";
      break;
    case stk::topology::FACE_RANK:
      oss << "2";
      break;
    case stk::topology::ELEMENT_RANK:
      oss << "8";
      break;
#if defined(LCM_TOPOLOGY_HIGH_DIMENSIONS)
      case 4:
      oss << "10";
      break;
      case 5:
      oss << "12";
      break;
      case 6:
      oss << "14";
      break;
#endif // LCM_TOPOLOGY_HIGH_DIMENSIONS
    }
    break;

  case OPEN:
    switch (rank) {
    default:
      std::cerr << "ERROR: " << __PRETTY_FUNCTION__;
      std::cerr << '\n';
      std::cerr << "stk::mesh::Entity rank is invalid: " << rank;
      std::cerr << '\n';
      exit(1);
      break;
    case stk::topology::NODE_RANK:
      oss << "5";
      break;
    case stk::topology::EDGE_RANK:
      oss << "3";
      break;
    case stk::topology::FACE_RANK:
      oss << "1";
      break;
    case stk::topology::ELEMENT_RANK:
      oss << "7";
      break;
#if defined(LCM_TOPOLOGY_HIGH_DIMENSIONS)
      case 4:
      oss << "9";
      break;
      case 5:
      oss << "11";
      break;
      case 6:
      oss << "13";
      break;
#endif // LCM_TOPOLOGY_HIGH_DIMENSIONS
    }
    break;
  }

  return oss.str();
}

//
// Auxiliary for graphviz output
//
inline std::string
dot_header()
{
  std::string
  header = "digraph mesh {\n";

  header += "  node [colorscheme=paired12]\n";
  header += "  edge [colorscheme=paired12]\n";

  return header;
}

//
// Auxiliary for graphviz output
//
inline std::string
dot_footer()
{
  return "}";
}

//
// Auxiliary for graphviz output
//
inline std::string
dot_entity(
    stk::mesh::EntityId const id,
    stk::mesh::EntityRank const rank,
    FractureState const fracture_state)
{
  std::ostringstream
  oss;

  oss << "  \"";
  oss << id;
  oss << "_";
  oss << rank;
  oss << "\"";
  oss << " [label=\"";
  //oss << entity_label(rank);
  //oss << " ";
  oss << id;
  oss << "\",style=filled,fillcolor=\"";
  oss << entity_color(rank, fracture_state);
  oss << "\"]\n";

  return oss.str();
}

//
// Auxiliary for graphviz output
//
inline std::string
relation_color(unsigned int const relation_id)
{
  std::ostringstream
  oss;

  switch (relation_id) {
  default:
    oss << 2 * (relation_id + 1);
    break;
  case 0:
    oss << "6";
    break;
  case 1:
    oss << "4";
    break;
  case 2:
    oss << "2";
    break;
  case 3:
    oss << "8";
    break;
  case 4:
    oss << "10";
    break;
  case 5:
    oss << "12";
    break;
  }

  return oss.str();
}

//
// Auxiliary for graphviz output
//
inline std::string
dot_relation(
    stk::mesh::EntityId const source_id,
    stk::mesh::EntityRank const source_rank,
    stk::mesh::EntityId const target_id,
    stk::mesh::EntityRank const target_rank,
    unsigned int const relation_local_id)
{
  std::ostringstream
  oss;

  oss << "  \"";
  oss << source_id;
  oss << "_";
  oss << source_rank;
  oss << "\" -> \"";
  oss << target_id;
  oss << "_";
  oss << target_rank;
  oss << "\" [color=\"";
  oss << relation_color(relation_local_id);
  oss << "\"]\n";

  return oss.str();
}

}  // namespace LCM

#endif // LCM_Topology_Utils_h
