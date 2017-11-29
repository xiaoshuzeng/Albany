//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Albany_AbstractDiscretization.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
DirichletFieldSideEq<EvalT, Traits>::
DirichletFieldSideEq(const Teuchos::ParameterList& p) :
  PHAL::DirichletSideEqBase<EvalT, Traits>(p)
{
  // Get field name
  field_name = p.get<std::string>("Field Name");

  // Get side mesh nodeset name
  nodeSetName = p.get<std::string>("Node Set Name");
}

// **********************************************************************
// Full Specialization: Residual
// **********************************************************************
template<>
void DirichletFieldSideEq<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Grab solution and residual
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  // Grab the field datum
  const Albany::NodalDOFManager& fieldDofManager = workset.disc->getDOFManager(field_name);
  Teuchos::RCP<const Tpetra_Map> fieldNodeMap = workset.disc->getNodeMapT(this->field_name);
  bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == workset.disc->getMapT(this->field_name)->getNodeNumElements());
  int fieldOffset = isFieldScalar ? 0 : this->dof_offset;
  Teuchos::RCP<const Tpetra_Vector> pvecT = workset.distParamLib->get(field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);
  const std::vector<GO>& nsNodesGIDs = ss_disc->getNodeSetGIDs().find(nodeSetName)->second;

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map       = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map = workset.disc->getSideSetsNodeMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map_local = ss_disc->getNodeMapT();
  Teuchos::RCP<const Tpetra_Map> map          = workset.disc->getMapT();
  Teuchos::RCP<const Tpetra_Map> nodes_map    = workset.disc->getNodeMapT();

  // Loop over nodes
  int ss_lid, lid, field_lid, node_lid;
  GO gid, node_gid;
  for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
    ss_lid = nsNodes[inode][this->dof_offset];
    gid = ss_map->getGlobalElement(ss_lid);
    node_gid = nsNodesGIDs[inode];
    node_lid = ss_nodes_map_local->getLocalElement(node_gid);
    node_gid = ss_nodes_map->getGlobalElement(node_lid);
    lid = map->getLocalElement(gid);
    field_lid = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);

    fT_nonconstView[lid] = xT_constView[lid] - pT[field_lid];
  }
}

// **********************************************************************
// Full Specialization: Jacobian
// **********************************************************************
template<>
void DirichletFieldSideEq<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Grab solution, jacobian, and residual
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = workset.JacT;
  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fillResid) {
    fT_nonconstView = fT->get1dViewNonConst();
  }

  // Grab the field datum
  const Albany::NodalDOFManager& fieldDofManager = workset.disc->getDOFManager(field_name);
  Teuchos::RCP<const Tpetra_Map> fieldNodeMap = workset.disc->getNodeMapT(this->field_name);
  bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == workset.disc->getMapT(this->field_name)->getNodeNumElements());
  int fieldOffset = isFieldScalar ? 0 : this->dof_offset;
  Teuchos::RCP<const Tpetra_Vector> pvecT = workset.distParamLib->get(field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);
  const std::vector<GO>& nsNodesGIDs = ss_disc->getNodeSetGIDs().find(nodeSetName)->second;

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map       = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map = workset.disc->getSideSetsNodeMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map_local = ss_disc->getNodeMapT();
  Teuchos::RCP<const Tpetra_Map> map          = workset.disc->getMapT();
  Teuchos::RCP<const Tpetra_Map> nodes_map    = workset.disc->getNodeMapT();

  // Helper structures
  const RealType j_coeff = workset.j_coeff;
  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  // Loop over nodes
  int ss_lid, lid, field_lid, node_lid;
  GO gid, node_gid;
  for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
    // Get the dof GID and its LID in the volume discretization
    ss_lid = nsNodes[inode][this->dof_offset];
    gid = ss_map->getGlobalElement(ss_lid);
    node_gid = nsNodesGIDs[inode];
    node_lid = ss_nodes_map_local->getLocalElement(node_gid);
    node_gid = ss_nodes_map->getGlobalElement(node_lid);
    lid = map->getLocalElement(gid);
    index[0] = lid;

    // Get jacobian entries
    numEntriesT = jacT->getNumEntriesInLocalRow(lid);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);

    jacT->getLocalRowCopy(lid, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    std::fill_n(matrixEntriesT.begin(),numEntriesT,0.0);
    jacT->replaceLocalValues(lid, matrixIndicesT(), matrixEntriesT());
    jacT->replaceLocalValues(lid, index(), value());

    if (fillResid) {
      // Get the datum LID
      field_lid = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
      fT_nonconstView[lid] = xT_constView[lid] - pT[field_lid];
    }
  }
}

// **********************************************************************
// Full Specialization: Tangent
// **********************************************************************
// **********************************************************************
template<>
void DirichletFieldSideEq<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Grab solution, residual, and tangent structures
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = workset.VxT;

  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fT != Teuchos::null) {
    fT_nonconstView = fT->get1dViewNonConst();
  }

  Teuchos::ArrayRCP<const ST> VxT_constView;

  // Grab the field datum
  const Albany::NodalDOFManager& fieldDofManager = workset.disc->getDOFManager(field_name);
  Teuchos::RCP<const Tpetra_Map> fieldNodeMap = workset.disc->getNodeMapT(this->field_name);
  bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == workset.disc->getMapT(this->field_name)->getNodeNumElements());
  int fieldOffset = isFieldScalar ? 0 : this->dof_offset;
  Teuchos::RCP<const Tpetra_Vector> pvecT = workset.distParamLib->get(field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);
  const std::vector<GO>& nsNodesGIDs = ss_disc->getNodeSetGIDs().find(nodeSetName)->second;

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map       = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map = workset.disc->getSideSetsNodeMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map_local = ss_disc->getNodeMapT();
  Teuchos::RCP<const Tpetra_Map> map          = workset.disc->getMapT();
  Teuchos::RCP<const Tpetra_Map> nodes_map    = workset.disc->getNodeMapT();

  // Helper structures
  const RealType j_coeff = workset.j_coeff;

  // Loop over nodes
  int ss_lid, lid, field_lid, node_lid;
  GO gid, node_gid;
  for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
    // Get the dof GID and its LID in the volume discretization
    ss_lid = nsNodes[inode][this->dof_offset];
    gid = ss_map->getGlobalElement(ss_lid);
    node_gid = nsNodesGIDs[inode];
    node_lid = ss_nodes_map_local->getLocalElement(node_gid);
    node_gid = ss_nodes_map->getGlobalElement(node_lid);
    lid = map->getLocalElement(gid);

    if (fT != Teuchos::null) {
      // Get the datum LID
      field_lid = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
      fT_nonconstView[lid] = xT_constView[lid] - pT[field_lid];
    }

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView;
      for (int i=0; i<workset.num_cols_x; i++) {
        JVT_nonconstView = JVT->getDataNonConst(i);
        VxT_constView = VxT->getData(i);
        JVT_nonconstView[lid] = j_coeff*VxT_constView[lid];
      }
    }

    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView;
      for (int i=0; i<workset.num_cols_p; i++) {
        fpT_nonconstView = fpT->getDataNonConst(i);
        fpT_nonconstView[lid] = 0;
      }
    }
  }
}

// **********************************************************************
// Full Specialization: DistParamDeriv
// **********************************************************************
template<>
void DirichletFieldSideEq<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Grab solution, residual, and deriv structures
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  Teuchos::ArrayRCP<ST> fT_nonconstView;
  Teuchos::RCP<const Tpetra_Vector> pvecT = workset.distParamLib->get(field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  if (fT != Teuchos::null) {
    fT_nonconstView = fT->get1dViewNonConst();
  }

  bool isFieldParameter =  workset.dist_param_deriv_name == this->field_name;

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);
  const std::vector<GO>& nsNodesGIDs = ss_disc->getNodeSetGIDs().find(nodeSetName)->second;

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map       = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map = workset.disc->getSideSetsNodeMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> ss_nodes_map_local = ss_disc->getNodeMapT();
  Teuchos::RCP<const Tpetra_Map> map          = workset.disc->getMapT();
  Teuchos::RCP<const Tpetra_Map> nodes_map    = workset.disc->getNodeMapT();

  // Helper structures
  const RealType j_coeff = workset.j_coeff;
  const Albany::NodalDOFManager& fieldDofManager = workset.disc->getDOFManager(field_name);
  Teuchos::RCP<const Tpetra_Map> fieldNodeMap = workset.disc->getNodeMapT(this->field_name);
  bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == workset.disc->getMapT(this->field_name)->getNodeNumElements());
  int fieldOffset = isFieldScalar ? 0 : this->dof_offset;

  // Loop over nodes
  int ss_lid, lid, field_lid, node_lid;
  GO gid, node_gid;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = workset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;
    if(isFieldParameter) {
      // Grab the field datum
      const Albany::NodalDOFManager& fieldDofManager = workset.disc->getDOFManager(field_name);
      Teuchos::RCP<const Tpetra_Map> fieldNodeMap = workset.disc->getNodeMapT(this->field_name);
      bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == workset.disc->getMapT(this->field_name)->getNodeNumElements());
      int fieldOffset = isFieldScalar ? 0 : this->dof_offset;

      // Loop over nodes
      int ss_lid, lid, field_lid;
      GO gid, node_gid;
      for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
        // Get the dof GID and its LID in the volume discretization
        ss_lid = nsNodes[inode][this->dof_offset];
        gid = ss_map->getGlobalElement(ss_lid);
        node_gid = nsNodesGIDs[inode];
        node_lid = ss_nodes_map_local->getLocalElement(node_gid);
        node_gid = ss_nodes_map->getGlobalElement(node_lid);
        lid = map->getLocalElement(gid);
        field_lid = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);

        for (int col=0; col<num_cols; ++col) {
          VpT_nonconstView = VpT->getDataNonConst(col);
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[field_lid] -= VpT_nonconstView[lid];
          VpT_nonconstView[lid] = 0.0;
        }
      }
    } else {
      for (unsigned int inode = 0; inode < nsNodes.size(); ++inode) {
        // Get the dof GID and its LID in the volume discretization
        ss_lid = nsNodes[inode][this->dof_offset];
        gid = ss_map->getGlobalElement(ss_lid);
        node_gid = nsNodesGIDs[inode];
        node_lid = ss_nodes_map_local->getLocalElement(node_gid);
        node_gid = ss_nodes_map->getGlobalElement(node_lid);
        lid = map->getLocalElement(gid);

        for (int col=0; col<num_cols; ++col) {
          VpT_nonconstView = VpT->getDataNonConst(col);
          VpT_nonconstView[lid] = 0.0;
         }
      }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    if(isFieldParameter) {
      // Grab the field datum
      const Albany::NodalDOFManager& fieldDofManager = workset.disc->getDOFManager(field_name);
      Teuchos::RCP<const Tpetra_Map> fieldNodeMap = workset.disc->getNodeMapT(this->field_name);
      bool isFieldScalar = (fieldNodeMap->getNodeNumElements() == workset.disc->getMapT(this->field_name)->getNodeNumElements());
      int fieldOffset = isFieldScalar ? 0 : this->dof_offset;
      for (unsigned int inode = 0; inode < nsNodes.size(); ++inode) {
        // Get the dof GID and its LID in the volume discretization
        ss_lid = nsNodes[inode][this->dof_offset];
        gid = ss_map->getGlobalElement(ss_lid);
        node_gid = nsNodesGIDs[inode];
        node_lid = ss_nodes_map_local->getLocalElement(node_gid);
        node_gid = ss_nodes_map->getGlobalElement(node_lid);
        lid = map->getLocalElement(gid);
        int lfield = fieldDofManager.getLocalDOF(fieldNodeMap->getLocalElement(node_gid),fieldOffset);
        for (int col=0; col<num_cols; ++col) {
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[lid] = -double(col == lfield);
        }
      }
    } else {
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        // Get the dof GID and its LID in the volume discretization
        ss_lid = nsNodes[inode][this->dof_offset];
        gid = ss_map->getGlobalElement(ss_lid);
        node_gid = nsNodesGIDs[inode];
        node_lid = ss_nodes_map_local->getLocalElement(node_gid);
        node_gid = ss_nodes_map->getGlobalElement(node_lid);
        lid = map->getLocalElement(gid);

        for (int col=0; col<num_cols; ++col) {
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[lid] = 0.0;
        }
      }
    }
  }
}

} // namespace PHAL
