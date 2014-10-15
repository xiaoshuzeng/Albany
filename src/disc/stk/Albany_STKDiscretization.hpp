//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STKDISCRETIZATION_HPP
#define ALBANY_STKDISCRETIZATION_HPP

#include <vector>
#include <utility>

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Epetra_Comm.h"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"

#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"

#include "Piro_NullSpaceUtils.hpp" // has defn of struct that holds null space info for ML

// Start of STK stuff
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldTraits.hpp>
#ifdef ALBANY_SEACAS
  #include <stk_io/StkMeshIoBroker.hpp>
#endif


namespace Albany {

  struct MeshGraph {

       std::vector<std::size_t> start;
       std::vector<std::size_t> adj;

  };


  struct DOFsStruct {
    Teuchos::RCP<Epetra_Map> node_map;
    Teuchos::RCP<Epetra_Map> overlap_node_map;
    Teuchos::RCP<Epetra_Map> map;
    Teuchos::RCP<Epetra_Map> overlap_map;
    NodalDOFManager dofManager;
    NodalDOFManager overlap_dofManager;
    std::vector<std::vector<int> > wsElNodeEqID_rawVec;
    std::vector<IDArray> wsElNodeEqID;
    std::vector<std::vector<int> > wsElNodeID_rawVec;
    std::vector<IDArray> wsElNodeID;
  };

  struct NodalDOFsStructContainer {
    typedef std::map<std::pair<std::string,int>,  DOFsStruct >  MapOfDOFsStructs;

    MapOfDOFsStructs mapOfDOFsStructs;
    std::map<std::string, MapOfDOFsStructs::const_iterator> fieldToMap;
    const DOFsStruct& getDOFsStruct(const std::string& field_name) const {return fieldToMap.find(field_name)->second->second;}; //TODO handole errors

    void addEmptyDOFsStruct(const std::string& field_name, const std::string& meshPart, int numComps){
    
      if(numComps != 1)
        mapOfDOFsStructs.insert(make_pair(make_pair(meshPart,1),DOFsStruct()));

      fieldToMap[field_name] = mapOfDOFsStructs.insert(make_pair(make_pair(meshPart,numComps),DOFsStruct())).first;
    }
    
  };

  class STKDiscretization : public Albany::AbstractDiscretization {
  public:

    //! Constructor
    STKDiscretization(
       Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct,
       const Teuchos::RCP<const Epetra_Comm>& comm,
       const Teuchos::RCP<Piro::MLRigidBodyModes>& rigidBodyModes = Teuchos::null);


    //! Destructor
    ~STKDiscretization();

    //! Get DOF map
    Teuchos::RCP<const Epetra_Map> getMap() const;

    //! Get overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap() const;

    //! Get field DOF map
    Teuchos::RCP<const Epetra_Map> getMap(const std::string& field_name) const;

    //! Get field overlapped DOF map
    Teuchos::RCP<const Epetra_Map> getOverlapMap(const std::string& field_name) const;

    //! Get Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const;

    //! Get overlap Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getOverlapJacobianGraph() const;

    //! Get Node map
    Teuchos::RCP<const Epetra_Map> getNodeMap() const;

    //! Get overlapped Node map
    Teuchos::RCP<const Epetra_Map> getOverlapNodeMap() const;

    //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
    const NodeSetList& getNodeSets() const { return nodeSets; };
    const NodeSetCoordList& getNodeSetCoords() const { return nodeSetCoords; };

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const SideSetList& getSideSets(const int workset) const { return sideSets[workset]; };

    //! Get connectivity map from elementGID to workset
    WsLIDList& getElemGIDws() { return elemGIDws; };

    //! Get map from (Ws, El, Local Node) -> NodeLID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type& getWsElNodeEqID() const;

    //! Get map from (Ws, Local Node) -> NodeGID
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type& getWsElNodeID() const;

    //! Get IDArray for (Ws, Local Node, nComps) -> NodeLID, works for both scalar and vector fields
    const std::vector<IDArray>& getElNodeEqID(const std::string& field_name) const
        {return nodalDOFsStructContainer.getDOFsStruct(field_name).wsElNodeEqID;}

    //! Retrieve coodinate vector (num_used_nodes * 3)
    Teuchos::ArrayRCP<double>& getCoordinates() const;

    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& getCoords() const;
    const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type& getSphereVolume() const;

    //! Print the coordinates for debugging

    void printCoords() const;

    //! Get stateArrays
    Albany::StateArrays& getStateArrays() {return stateArrays;}

    //! Get nodal parameters state info struct
    const Albany::StateInfoStruct& getNodalParameterSIS() const
      {return stkMeshStruct->getFieldContainer()->getNodalParameterSIS();}

    //! Retrieve Vector (length num worksets) of element block names
    const Albany::WorksetArray<std::string>::type&  getWsEBNames() const;
    //! Retrieve Vector (length num worksets) of physics set index
    const Albany::WorksetArray<int>::type&  getWsPhysIndex() const;

    void writeSolution(const Epetra_Vector& soln, const double time, const bool overlapped = false);

    Teuchos::RCP<Epetra_Vector> getSolutionField() const;

    int getSolutionFieldHistoryDepth() const;
    Teuchos::RCP<Epetra_MultiVector> getSolutionFieldHistory() const;
    Teuchos::RCP<Epetra_MultiVector> getSolutionFieldHistory(int maxStepCount) const;
    void getSolutionFieldHistory(Epetra_MultiVector &result) const;

    void setResidualField(const Epetra_Vector& residual);

    // Retrieve mesh struct
    Teuchos::RCP<Albany::AbstractSTKMeshStruct> getSTKMeshStruct() {return stkMeshStruct;}
    Teuchos::RCP<Albany::AbstractMeshStruct> getMeshStruct() const {return stkMeshStruct;}

    //! Flag if solution has a restart values -- used in Init Cond
    bool hasRestartSolution() const {return stkMeshStruct->hasRestartSolution();}

    //! STK supports MOR
    virtual bool supportsMOR() const { return true; }

    //! If restarting, convenience function to return restart data time
    double restartDataTime() const {return stkMeshStruct->restartDataTime();}

    //! After mesh modification, need to update the element connectivity and nodal coordinates
    void updateMesh(bool shouldTransferIPData = false);

    //! Function that transforms an STK mesh of a unit cube (for FELIX problems)
    void transformMesh();

    //! Close current exodus file in stk_io and create a new one for an adapted mesh and new results
    void reNameExodusOutput(std::string& filename);

   //! Get number of spatial dimensions
    int getNumDim() const { return stkMeshStruct->numDim; }

    //! Get number of total DOFs per node
    int getNumEq() const { return neq; }

    //! Locate nodal dofs in non-overlapping vectors using local indexing
    int getOwnedDOF(const int inode, const int eq) const;

    //! Locate nodal dofs in overlapping vectors using local indexing
    int getOverlapDOF(const int inode, const int eq) const;

    //! Locate nodal dofs using global indexing
    int getGlobalDOF(const int inode, const int eq) const;


    //! used when NetCDF output on a latitude-longitude grid is requested.
    // Each struct contains a latitude/longitude index and it's parametric
    // coordinates in an element.
    struct interp {
      std::pair<double, double> parametric_coords;
      std::pair<unsigned, unsigned> latitude_longitude;
    };

    const stk::mesh::MetaData& getSTKMetaData(){ return metaData; }

    const stk::mesh::BulkData& getSTKBulkData(){ return bulkData; }

  private:

    //! Private to prohibit copying
    STKDiscretization(const STKDiscretization&);

    //! Private to prohibit copying
    STKDiscretization& operator=(const STKDiscretization&);

    inline int gid(const stk::mesh::Entity node) const;

    // Copy values from STK Mesh field to given Epetra_Vector
    void getSolutionField(Epetra_Vector &result) const;

    //! Copy field from STK Mesh field to given Epetra_Vector
    void getField(Epetra_Vector &field_vector, const std::string& field_name) const;

    Teuchos::RCP<Epetra_MultiVector> getSolutionFieldHistoryImpl(int stepCount) const;
    void getSolutionFieldHistoryImpl(Epetra_MultiVector &result) const;

    // Copy solution vector from Epetra_Vector into STK Mesh
    // Here soln is the local (non overlapped) solution
    void setSolutionField(const Epetra_Vector& soln);

    // Copy field vector into STK Mesh field
    void setField(const Epetra_Vector &field_vector, const std::string& field_name, bool overlapped=false);

    // Copy solution vector from Epetra_Vector into STK Mesh
    // Here soln is the local + neighbor (overlapped) solution
    void setOvlpSolutionField(const Epetra_Vector& soln);

    int nonzeroesPerRow(const int neq) const;
    double monotonicTimeLabel(const double time);

    void computeNodalEpetraMaps(bool overlapped);

    //! Process STK mesh for Owned nodal quantitites
    void computeOwnedNodesAndUnknowns();
    //! Process coords for ML
    void setupMLCoords();
    //! Process STK mesh for Overlap nodal quantitites
    void computeOverlapNodesAndUnknowns();
    //! Process STK mesh for CRS Graphs
    void computeGraphs();
    //! Process STK mesh for Workset/Bucket Info
    void computeWorksetInfo();
    //! Process STK mesh for NodeSets
    void computeNodeSets();
    //! Process STK mesh for SideSets
    void computeSideSets();
    //! Call stk_io for creating exodus output file
    void setupExodusOutput();
    //! Call stk_io for creating NetCDF output file
    void setupNetCDFOutput();
    int processNetCDFOutputRequest(const Epetra_Vector&);
    //! Find the local side id number within parent element
    unsigned determine_local_side_id( const stk::mesh::Entity elem , stk::mesh::Entity side );
    //! Call stk_io for creating exodus output file
    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Convert the stk mesh on this processor to a nodal graph using SEACAS
    void meshToGraph();

    double previous_time_label;

  protected:


    //! Stk Mesh Objects
    stk::mesh::MetaData& metaData;
    stk::mesh::BulkData& bulkData;

    //! Epetra communicator
    Teuchos::RCP<const Epetra_Comm> comm;

    //! Node map
    Teuchos::RCP<Epetra_Map> node_map;

    //! Unknown Map
    Teuchos::RCP<Epetra_Map> map;

    NodalDOFsStructContainer nodalDOFsStructContainer;

    //! Overlapped unknown map, and node map
    Teuchos::RCP<Epetra_Map> overlap_map;
    Teuchos::RCP<Epetra_Map> overlap_node_map;

    //! Jacobian matrix graph
    Teuchos::RCP<Epetra_CrsGraph> graph;

    //! Overlapped Jacobian matrix graph
    Teuchos::RCP<Epetra_CrsGraph> overlap_graph;

    //! Processor ID
    unsigned int myPID;

    //! Number of equations (and unknowns) per node
    const unsigned int neq;

    //! Number of elements on this processor
    unsigned int numMyElements;

    //! node sets stored as std::map(string ID, int vector of GIDs)
    Albany::NodeSetList nodeSets;
    Albany::NodeSetCoordList nodeSetCoords;

    //! side sets stored as std::map(string ID, SideArray classes) per workset (std::vector across worksets)
    std::vector<Albany::SideSetList> sideSets;

    //! Connectivity array [workset, element, local-node, Eq] => LID
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type wsElNodeEqID;

    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type wsElNodeID;

    mutable Teuchos::ArrayRCP<double> coordinates;
    Albany::WorksetArray<std::string>::type wsEBNames;
    Albany::WorksetArray<int>::type wsPhysIndex;
    Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type coords;
    Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type sphereVolume;

    //! Connectivity map from elementGID to workset and LID in workset
    WsLIDList  elemGIDws;

    // States: vector of length worksets of a map from field name to shards array
    Albany::StateArrays stateArrays;
    std::vector<std::vector<std::vector<double> > > nodesOnElemStateVec;

    //! list of all owned nodes, saved for setting solution
    std::vector< stk::mesh::Entity > ownednodes ;
    std::vector< stk::mesh::Entity > cells ;

    //! list of all overlap nodes, saved for getting coordinates for mesh motion
    std::vector< stk::mesh::Entity > overlapnodes ;

    //! Number of elements on this processor
    int numOwnedNodes;
    int numOverlapNodes;
    int numGlobalNodes;

    // Needed to pass coordinates to ML.
    Teuchos::RCP<Piro::MLRigidBodyModes> rigidBodyModes;

    int netCDFp;
    int netCDFOutputRequest;
    std::vector<int> varSolns;
    Albany::WorksetArray<Teuchos::ArrayRCP<std::vector<interp> > >::type interpolateData;

    // Storage used in periodic BCs to un-roll coordinates. Pointers saved for destructor.
    std::vector<double*>  toDelete;

    Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct;

    // Used in Exodus writing capability
#ifdef ALBANY_SEACAS
    Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data;

    int outputInterval;

    size_t outputFileIdx;
#endif
    bool interleavedOrdering;

  private:

    MeshGraph nodalGraph;

    // find the location of "value" within the first "count" locations of "vector"
    ssize_t in_list(const std::size_t value, std::size_t count, std::size_t *vector) {

      for(std::size_t i=0; i < count; i++) {
        if(vector[i] == value)
          return i;
      }
       return -1;
    }

    ssize_t in_list(const std::size_t value, std::vector<std::size_t> vector) {

      std::size_t count = vector.size();
      for(std::size_t i=0; i < count; i++) {
        if(vector[i] == value)
          return i;
      }
      return -1;
    }

    ssize_t entity_in_list(stk::mesh::Entity value, std::vector<stk::mesh::Entity> const& arg_vector) {

      std::size_t count = arg_vector.size();
      for(std::size_t i=0; i < count; i++) {
        if(bulkData.identifier(arg_vector[i]) == bulkData.identifier(value))
          return i;
      }
      return -1;
    }

    void printVertexConnectivity();

  };

}

#endif // ALBANY_STKDISCRETIZATION_HPP
