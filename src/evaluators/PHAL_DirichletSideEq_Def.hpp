//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Teuchos_TestForException.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "Albany_AbstractDiscretization.hpp"

namespace PHAL {

// ========================== DirichletSideEqBase ====================== //

template<typename EvalT,typename Traits>
DirichletSideEqBase<EvalT, Traits>::
DirichletSideEqBase(const Teuchos::ParameterList& p)
{
  // The derived DirichletOffSideSets does not need this; uses a set of strings instead
  sideSetName = p.isParameter("Side Set Name") ? p.get<std::string>("Side Set Name") : "";

  dirName = p.get< std::string >("Dirichlet Name");
  Teuchos::RCP<PHX::DataLayout> dummy(new PHX::MDALayout<Dummy>(0));
  PHX::Tag<ScalarT> fieldTag(dirName, dummy);

  dof_offset = p.get<int>("DOF offset");

  this->addEvaluatedField(fieldTag);

  this->setName(dirName+PHX::typeAsString<EvalT>());
}

// ========================== DirichletSideEq ====================== //
template<typename EvalT,typename Traits>
DirichletSideEq<EvalT, Traits>::
DirichletSideEq(Teuchos::ParameterList& p)
 : DirichletSideEqBase<EvalT,Traits>(p)
{
  value = p.get<RealType>("Dirichlet Value");

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
               ("Parameter Library", Teuchos::null);

  this->registerSacadoParameter(this->dirName, paramLib);

  nodeSetName = p.get<std::string>("Node Set Name");
}

// Specializations for evaluate fields

// **********************************************************************
// Full Specialization: Residual
// **********************************************************************
template<>
void DirichletSideEq<PHAL::AlbanyTraits::Residual,PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> map = workset.disc->getMapT();

  int ss_lid, lid;
  GO gid;
  for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
    ss_lid = nsNodes[inode][this->dof_offset];
    gid = ss_map->getGlobalElement(ss_lid);
    lid = map->getLocalElement(gid);
    fT_nonconstView[lid] = xT_constView[lid] - this->value;
  }
}

// **********************************************************************
// Full Specialization: Jacobian
// **********************************************************************
template<>
void DirichletSideEq<PHAL::AlbanyTraits::Jacobian,PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = workset.JacT;

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> map = workset.disc->getMapT();

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

  int ss_lid, lid;
  GO gid;
  for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
    ss_lid = nsNodes[inode][this->dof_offset];
    gid = ss_map->getGlobalElement(ss_lid);
    lid = map->getLocalElement(gid);
    index[0] = lid;

    numEntriesT = jacT->getNumEntriesInLocalRow(lid);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);

    jacT->getLocalRowCopy(lid, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    std::fill_n(matrixEntriesT.begin(),numEntriesT,0.0);
    jacT->replaceLocalValues(lid, matrixIndicesT(), matrixEntriesT());
    jacT->replaceLocalValues(lid, index(), value());

    if (fillResid) {
      fT_nonconstView[lid] = xT_constView[lid] - this->value.val();
    }
  }
}

// **********************************************************************
// Full Specialization: Tangent
// **********************************************************************
template<>
void DirichletSideEq<PHAL::AlbanyTraits::Tangent,PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = workset.VxT;

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> map = workset.disc->getMapT();

  Teuchos::ArrayRCP<const ST> VxT_constView;
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fT != Teuchos::null) {
    fT_nonconstView = fT->get1dViewNonConst();
  }
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  const RealType j_coeff = workset.j_coeff;

  LO ss_lid, lid;
  GO gid;
  for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
    ss_lid = nsNodes[inode][this->dof_offset];
    gid = ss_map->getGlobalElement(ss_lid);
    lid = map->getLocalElement(gid);

    if (fT != Teuchos::null) {
      fT_nonconstView[lid] = xT_constView[lid] - this->value.val();
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
        fpT_nonconstView[lid] = -this->value.dx(workset.param_offset+i);
      }
    }
  }
}
// **********************************************************************
// Full Specialization: DistParamDeriv
// **********************************************************************
template<>
void DirichletSideEq<PHAL::AlbanyTraits::DistParamDeriv,PHAL::AlbanyTraits>::
evaluateFields(typename PHAL::AlbanyTraits::EvalData workset)
{
  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  //non-const view of fpVT
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  // Grab the side discretization
  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = workset.disc->getSideSetDiscretizations().at(this->sideSetName);

  // Grab the side's nodeset from the side discretization
  const std::vector<std::vector<GO> >& nsNodes = ss_disc->getNodeSets().at(nodeSetName);

  // Grab the map of the side nodes with the volume discretization GID
  Teuchos::RCP<const Tpetra_Map> ss_map = workset.disc->getSideSetsMapT().at(sideSetName);
  Teuchos::RCP<const Tpetra_Map> map = workset.disc->getMapT();

  LO ss_lid, lid;
  GO gid;
  if (trans) {
    // For (df/dp)^T*V we zero out corresponding entries in V
    Teuchos::RCP<Tpetra_MultiVector> VpT = workset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;
    for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
      ss_lid = nsNodes[inode][this->dof_offset];
      gid = ss_map->getGlobalElement(ss_lid);
      lid = map->getLocalElement(gid);

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][lid] = 0.0;
        VpT_nonconstView = VpT->getDataNonConst(col);
        VpT_nonconstView[lid] = 0.0;
       }
    }
  } else {
    // for (df/dp)*V we zero out corresponding entries in df/dp
    for (unsigned int inode=0; inode<nsNodes.size(); ++inode) {
      ss_lid = nsNodes[inode][this->dof_offset];
      gid = ss_map->getGlobalElement(ss_lid);
      lid = map->getLocalElement(gid);

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][lid] = 0.0;
        fpVT_nonconstView = fpVT->getDataNonConst(col);
        fpVT_nonconstView[lid] = 0.0;
      }
    }
  }
}

} // namespace PHAL
