//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Define only if LCM is enabled
#if defined (ALBANY_LCM)

#include <stk_mesh/base/FieldBase.hpp>
#include "Topology.h"

// FIXME: need to extract Topology member functions specific to
// Barycentric subdivision and move into their own header!

namespace LCM
{

void
Topology::divideSegmentsHalf()
{
//get the segment from the original mesh
  std::vector<stk::mesh::Entity>
  initial_entities_1D = getEntitiesByRank(*(getBulkData()), stk::topology::EDGE_RANK);
  std::vector<stk::mesh::Entity> vector_nodes;

//Adding nodes to the divide segments by half
  std::vector<size_t> request1(getSpaceDimension() + 1, 0);
  request1[0] = initial_entities_1D.size();
  addEntities(request1);

  std::vector<stk::mesh::Entity>
  initial_entities_0D = getEntitiesByRank(*(getBulkData()), stk::topology::NODE_RANK);

//add a relation from the former segment to a new node
  for (unsigned int i = 0; i < initial_entities_1D.size(); i++) {
    addRelation(initial_entities_1D[i], initial_entities_0D[i], 2);
  }

//adding segments
  std::vector<size_t> requests_step1_2(getSpaceDimension() + 1, 0);
  requests_step1_2[1] = initial_entities_1D.size();
  addEntities(requests_step1_2);
  std::vector<stk::mesh::Entity>
  modified_entities_1D = getEntitiesByRank(*(getBulkData()), stk::topology::EDGE_RANK);

  for (unsigned int i = 0; i < initial_entities_1D.size(); i++) {
//Look for all the relations for each segment
    stk::mesh::Entity const* _relations = getBulkData()->begin_nodes(initial_entities_1D[i]);
    unsigned const num_relations = getBulkData()->num_nodes(initial_entities_1D[i]);

    //add new relation between the new node and the corresponding node in the original mesh
    for (unsigned j = 0; j < num_relations; j++) {
      if (getLocalRelationId(initial_entities_1D[i],
                             _relations[j]) == 1) {
        addRelation(modified_entities_1D[i], _relations[j], 0);
        removeRelation(initial_entities_1D[i], _relations[j], 1);
      }
      addRelation(modified_entities_1D[i], initial_entities_0D[i], 1); // JGF: Why inside loop?
    }
    // change the blue relation into a green one
    for (unsigned j = 0; j < num_relations; j++) {
      if (getLocalRelationId(initial_entities_1D[i],
                             _relations[j]) == 2) {
        addRelation(initial_entities_1D[i], _relations[j], 1);
        removeRelation(initial_entities_1D[i], _relations[j], 2);
      }
    }
  }

//adding the relation between the new segment and the faces
  for (unsigned int i = 0; i < initial_entities_1D.size(); i++) {
    stk::mesh::Entity const* _relations = getBulkData()->begin_faces(initial_entities_1D[i]);
    unsigned const num_relations = getBulkData()->num_faces(initial_entities_1D[i]);

    for (unsigned int j = 0; j < num_relations; j++) {
      addRelation(_relations[j], modified_entities_1D[i],
                  getNumberLowerRankEntities(_relations[j]));
    }
  }

}

void
Topology::addCentroid()
{
  //get the faces form the original mesh
  std::vector<stk::mesh::Entity>
  initial_entities_2D = getEntitiesByRank(*(getBulkData()), stk::topology::FACE_RANK);

  //Adding nodes to the faces
  std::vector<size_t> request2(getSpaceDimension() + 1, 0);
  request2[0] = initial_entities_2D.size();
  addEntities(request2);
}

void
Topology::connectCentroid()
{
  //get the faces form the original mesh
  std::vector<stk::mesh::Entity>
  initial_entities_2D = getEntitiesByRank(*(getBulkData()), stk::topology::FACE_RANK);

  //get the centroid
  std::vector<stk::mesh::Entity>
  modified_entities_0D = getEntitiesByRank(*(getBulkData()), stk::topology::NODE_RANK);

  //adding new segment
  std::vector<size_t> request2(getSpaceDimension() + 1, 0);
  request2[1] = 6 * initial_entities_2D.size();
  addEntities(request2);
  std::vector<stk::mesh::Entity>
  modified_entities_1D = getEntitiesByRank(*(getBulkData()), stk::topology::EDGE_RANK);

  for (unsigned int i = 0; i < initial_entities_2D.size(); i++) {

    //get boundary nodes
    std::vector<stk::mesh::Entity>
    boundary_entities_0D = getBoundaryEntities(initial_entities_2D[i], stk::topology::NODE_RANK);

    //adding new relation
    for (int j = 0; j < getNumberLowerRankEntities(initial_entities_2D[i]);
        j++) {
      addRelation(modified_entities_1D[6 * i + j],
                  modified_entities_0D[i], 0);
      addRelation(modified_entities_1D[6 * i + j],
                  boundary_entities_0D[j], 1);
    }
  }
}

void
Topology::addNewFaces()
{
  //get the faces form the original mesh
  std::vector<stk::mesh::Entity>
  initial_entities_2D = getEntitiesByRank(*(getBulkData()), stk::topology::FACE_RANK);

  //Adding nodes to the faces
  std::vector<size_t> request2(getSpaceDimension() + 1, 0);
  request2[2] = 6 * initial_entities_2D.size();
  addEntities(request2);

  //get boundary nodes
  std::vector<stk::mesh::Entity>
  boundary_entities_1D = getBoundaryEntities(initial_entities_2D[0], stk::topology::EDGE_RANK);

  for (unsigned i = 0; i < boundary_entities_1D.size(); i++) {
    std::cout << getBulkData()->identifier(boundary_entities_1D[i]) << std::endl;
  }
}
void
Topology::connectNewFaces()
{
  //get the faces form the original mesh
  std::vector<stk::mesh::Entity>
  initial_entities_2D = getEntitiesByRank(*(getBulkData()), stk::topology::FACE_RANK);

}

///
/// \brief Practice creating the barycentric subdivision
///
void
Topology::barycentricSubdivisionAlt()
{
  // Use to assign unique ids
  setHighestIds();
  // Begin mesh update
  getBulkData()->modification_begin();
  divideSegmentsHalf();
  addCentroid();
  connectCentroid();
  addNewFaces();
  connectNewFaces();
  // End mesh update
  getBulkData()->modification_end();
}

} //End of namespace LCM

#endif // #if defined (ALBANY_LCM)

