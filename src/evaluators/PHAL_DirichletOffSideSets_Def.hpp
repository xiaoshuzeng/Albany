//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Albany_NodalDOFManager.hpp"

namespace PHAL
{

template<typename EvalT,typename Traits>
DirichletOffSideSets<EvalT, Traits>::
DirichletOffSideSets(Teuchos::ParameterList& p) :
  DirichletSideEqBase<EvalT, Traits>(p)
{
  sideSetNames = *p.get<Teuchos::RCP<std::vector<std::string>>>("Side Set Names");
  value = p.get<double>("Dirichlet Value");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
               ("Parameter Library", Teuchos::null);
  this->registerSacadoParameter(this->dirName, paramLib);
}

template<typename EvalT,typename Traits>
void DirichletOffSideSets<EvalT, Traits>::
gather_rows(typename Traits::EvalData workset)
{
  if (rows.size()>0) {
    // Do this only once
    return;
  }

  // Gather all node IDs from all the stored sidesets
  // NOTE: would like to do this in postRegistrationSetup once and for all
  //       but AlbanyTraits::SetupData is a string, not the workset, so we
  //       cannot access the discretization
  Teuchos::RCP<const Tpetra_Map> node_map = workset.disc->getNodeMapT();
  LO numLocalNodes  = node_map->getNodeNumElements();
  GO numGlobalNodes = node_map->getGlobalNumElements();

  // Create a helper NodalDOFManager to help us recovering dof ids from the node id
  // and the dof offset
  dof_manager = Teuchos::rcp(new Albany::NodalDOFManager());
  bool interleaved = workset.disc->getMeshStruct()->getMeshSpecs()[0]->interleavedOrdering;
  dof_manager->setup(workset.disc->getNumEq(),numLocalNodes,numGlobalNodes,interleaved);

  for (const auto& ss_name : sideSetNames) {
    // Grab the side discretization
    Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(ss_name);

    // Grab the map of the side nodes with the volume discretization GID
    Teuchos::RCP<const Tpetra_Map> ss_node_map = workset.disc->getSideSetsNodeMapT().at(ss_name);
    LO numSideLocalNodes  = ss_node_map->getNodeNumElements();

    for (LO i=0; i<numSideLocalNodes; ++i) {
      // Get gid of this node from ss map, and get local id in the volume discretization
      GO node_gid = ss_node_map->getGlobalElement(i);
      LO node_lid = node_map->getLocalElement(node_gid);

      // Get local dof id in the full volume discretization
      rows.insert(dof_manager->getLocalDOF(node_lid,this->dof_offset));
    }
  }
}

// **********************************************************************
// Full Specialization: Residual
// **********************************************************************
template<>
void DirichletOffSideSets<PHAL::AlbanyTraits::Residual, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Gather all node IDs from all the stored nodesets
  // TODO: do this in postRegistrationSetup once and for all
  gather_rows(workset);

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  Teuchos::RCP<const Tpetra_Map> nodes = workset.disc->getNodeMapT();
  for (int inode=0; inode<static_cast<int>(nodes->getNodeNumElements()); ++inode)
  {
    int row = dof_manager->getLocalDOF(inode,this->dof_offset);
    if (rows.find(row)==rows.end())
    {
      // This node is NOT on the input nodeset: proceed to set the BC
      fT_nonconstView[row] = xT_constView[row] - this->value;
    }
  }
}

// **********************************************************************
// Full specialization: Jacobian
// **********************************************************************
template<>
void DirichletOffSideSets<PHAL::AlbanyTraits::Jacobian, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Gather all node IDs from all the stored nodesets
  // TODO: do this in postRegistrationSetup once and for all
  gather_rows(workset);

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = workset.JacT;

  const RealType j_coeff = workset.j_coeff;
  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;

  if (fillResid) {
    fT_nonconstView = fT->get1dViewNonConst();
  }

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  Teuchos::RCP<const Tpetra_Map> nodes = workset.disc->getNodeMapT();
  for (int inode=0; inode<static_cast<int>(nodes->getNodeNumElements()); ++inode)
  {
    int row = dof_manager->getLocalDOF(inode,this->dof_offset);
    if (rows.find(row)==rows.end())
    {
      // This node is NOT on the input nodeset: proceed to set the BC
      index[0] = row;

      numEntriesT = jacT->getNumEntriesInLocalRow(row);
      matrixEntriesT.resize(numEntriesT);
      matrixIndicesT.resize(numEntriesT);

      jacT->getLocalRowCopy(row, matrixIndicesT(), matrixEntriesT(), numEntriesT);

      for (int i=0; i<numEntriesT; i++) {
        matrixEntriesT[i]=0;
      }

      jacT->replaceLocalValues(row, matrixIndicesT(), matrixEntriesT());
      jacT->replaceLocalValues(row, index(), value());

      if (fillResid) {
        fT_nonconstView[row] = xT_constView[row] - this->value.val();
      }
    }
  }
}

// **********************************************************************
// Full specialization: Tangent
// **********************************************************************
template<>
void DirichletOffSideSets<PHAL::AlbanyTraits::Tangent, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Gather all node IDs from all the stored nodesets
  // TODO: do this in postRegistrationSetup once and for all
  gather_rows(workset);

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = workset.VxT;

  Teuchos::ArrayRCP<const ST> VxT_constView;
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fT != Teuchos::null) {
    fT_nonconstView = fT->get1dViewNonConst();
  }

  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  const RealType j_coeff = workset.j_coeff;

  Teuchos::RCP<const Tpetra_Map> nodes = workset.disc->getNodeMapT();
  for (int inode=0; inode<static_cast<int>(nodes->getNodeNumElements()); ++inode)
  {
    int row = dof_manager->getLocalDOF(inode,this->dof_offset);
    if (rows.find(row)==rows.end())
    {
      // This node is NOT on the input nodeset: proceed to set the BC
      if (fT != Teuchos::null) {
        fT_nonconstView[row] = xT_constView[row] - this->value.val();
      }

      if (JVT != Teuchos::null)
      {
        Teuchos::ArrayRCP<ST> JVT_nonconstView;
        for (int i=0; i<workset.num_cols_x; i++)
        {
          JVT_nonconstView = JVT->getDataNonConst(i);
          VxT_constView = VxT->getData(i);
          JVT_nonconstView[row] = j_coeff*VxT_constView[row];
        }
      }

      if (fpT != Teuchos::null)
      {
        Teuchos::ArrayRCP<ST> fpT_nonconstView;
        for (int i=0; i<workset.num_cols_p; i++)
        {
          fpT_nonconstView = fpT->getDataNonConst(i);
          fpT_nonconstView[row] = -this->value.dx(workset.param_offset+i);
        }
      }
    }
  }
}

// **********************************************************************
// Full specialization: DistParamDeriv
// **********************************************************************
template<>
void DirichletOffSideSets<PHAL::AlbanyTraits::DistParamDeriv, PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  // Gather all node IDs from all the stored nodesets
  // TODO: do this in postRegistrationSetup once and for all
  gather_rows(workset);

  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  //non-const view of fpVT
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans)
  {
    Teuchos::RCP<Tpetra_MultiVector> VpT = workset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;

    Teuchos::RCP<const Tpetra_Map> nodes = workset.disc->getNodeMapT();
    for (int inode=0; inode<static_cast<int>(nodes->getNodeNumElements()); ++inode)
    {
      int row = dof_manager->getLocalDOF(inode,this->dof_offset);
      if (rows.find(row)==rows.end())
      {
        // It's a row not on the given node sets

        for (int col=0; col<num_cols; ++col)
        {
          //(*Vp)[col][row] = 0.0;
          VpT_nonconstView = VpT->getDataNonConst(col);
          VpT_nonconstView[row] = 0.0;
        }
      }
    }
  }
  // for (df/dp)*V we zero out corresponding entries in df/dp
  else
  {
    Teuchos::RCP<const Tpetra_Map> nodes = workset.disc->getNodeMapT();
    for (int inode=0; inode<static_cast<int>(nodes->getNodeNumElements()); ++inode)
    {
      int row = dof_manager->getLocalDOF(inode,this->dof_offset);
      if (rows.find(row)==rows.end())
      {
        // It's a row not on the given node sets

        for (int col=0; col<num_cols; ++col)
        {
          //(*fpV)[col][row] = 0.0;
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[row] = 0.0;
        }
      }
    }
  }
}

} // Namespace PHAL
