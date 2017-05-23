//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_DirichletCondensed.hpp"
#include <Tpetra_RowMatrixTransposer_decl.hpp>

namespace PHAL {

// ****************************************************************************
// DEFAULT - DO NOTHING 
// ****************************************************************************

template <typename EvalT, typename Traits>
DirichletCondensed<EvalT, Traits>::
DirichletCondensed(Teuchos::ParameterList& p)
  : PHAL::DirichletBase<EvalT, Traits>(p) {
}

template <typename EvalT, typename Traits>
void DirichletCondensed<EvalT, Traits>::
evaluateFields(typename Traits::EvalData d) {
}

// ****************************************************************************
// RESIDUAL SPECIALIZATION
// ****************************************************************************

template <typename Traits>
DirichletCondensed<PHAL::AlbanyTraits::Residual, Traits>::
DirichletCondensed(Teuchos::ParameterList& p)
  : PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p) {
}

template <typename Traits>
void DirichletCondensed<PHAL::AlbanyTraits::Residual, Traits>::
preEvaluate(PreEvalData workset) {

  // set the DBCs to the overlapped solution vector before evaluations
  // this really needs to be revisited
  auto owned_x = workset.xT;
  auto overlapped_x = workset.overlapped_xT;
  auto importer = workset.x_importerT;
  auto ns_nodes = workset.nodeSets->find(this->nodeSetID)->second;
  auto tmp_x = Teuchos::rcp(new Tpetra_Vector(owned_x->getMap()), true);
  tmp_x->update(1.0, *owned_x, 0.0);
  for (unsigned node = 0; node < ns_nodes.size(); ++node) {
    LO row = ns_nodes[node][this->offset];
    tmp_x->replaceLocalValue(row, this->value);
  }
  overlapped_x->doImport(*tmp_x, *importer, Tpetra::INSERT);
}

template <typename Traits>
void DirichletCondensed<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(EvalData workset) {

  // specify the residual to be exactly zero at DBC rows
  auto f = workset.fT;
  auto ns_nodes = workset.nodeSets->find(this->nodeSetID)->second;
  for (unsigned node = 0; node < ns_nodes.size(); ++node) {
    LO row = ns_nodes[node][this->offset];
    f->replaceLocalValue(row, 0.0);
  }

}

// ****************************************************************************
// JACOBIAN SPECIALIZATION
// ****************************************************************************

template <typename Traits>
DirichletCondensed<PHAL::AlbanyTraits::Jacobian, Traits>::
DirichletCondensed(Teuchos::ParameterList& p)
  : PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p) {
}

template <typename Traits>
void DirichletCondensed<PHAL::AlbanyTraits::Jacobian, Traits>::
preEvaluate(PreEvalData workset) {

  // set the DBCs to the overlapped solution vector before evaluations
  auto owned_x = workset.xT;
  auto overlapped_x = workset.overlapped_xT;
  auto importer = workset.x_importerT;
  auto ns_nodes = workset.nodeSets->find(this->nodeSetID)->second;
  auto tmp_x = Teuchos::rcp(new Tpetra_Vector(owned_x->getMap()), true);
  tmp_x->update(1.0, *owned_x, 0.0);
  for (unsigned node = 0; node < ns_nodes.size(); ++node) {
    LO row = ns_nodes[node][this->offset];
    tmp_x->replaceLocalValue(row, this->value.val());
  }
  overlapped_x->doImport(*tmp_x, *importer, Tpetra::INSERT);
}

template <typename Traits>
void DirichletCondensed<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(EvalData workset) {

  using Transposer = Tpetra::RowMatrixTransposer<ST, LO, GO, KokkosNode>;

  auto f = workset.fT;
  auto J = workset.JacT;
  auto ns_nodes = workset.nodeSets->find(this->nodeSetID)->second;
  bool fill_resid = Teuchos::nonnull(f);

  {  // zero out the row, leaving the diagonal unchanged
    size_t num_entries;
    Teuchos::Array<LO> indices;
    Teuchos::Array<ST> entries;
    for (unsigned node = 0; node < ns_nodes.size(); ++node) {
      LO row = ns_nodes[node][this->offset];
      num_entries = J->getNumEntriesInLocalRow(row);
      indices.resize(num_entries);
      entries.resize(num_entries);
      J->getLocalRowCopy(row, indices(), entries(), num_entries);
      for (size_t c = 0; c < num_entries; ++c)
        if (indices[c] != row)
          entries[c] = 0.0;
      J->replaceLocalValues(row, indices(), entries());
      if (fill_resid)
        f->replaceLocalValue(row, 0.0);
    }
  }

  { // zero out the column, leaving the diagonal unchanged
    auto transposer = Teuchos::rcp(new Transposer(J));
    auto JT = transposer->createTranspose();
    size_t num_entries;
    Teuchos::Array<GO> indices;
    Teuchos::Array<ST> entries;
    Teuchos::Array<GO> index(1);
    Teuchos::Array<ST> entry(1);
    entry[0] = 0.0;
    for (unsigned node = 0; node < ns_nodes.size(); ++node) {
      LO row_lid = ns_nodes[node][this->offset];
      GO row_gid = f->getMap()->getGlobalElement(row_lid);
      num_entries = JT->getNumEntriesInGlobalRow(row_gid);
      indices.resize(num_entries);
      entries.resize(num_entries);
      JT->getGlobalRowCopy(row_gid, indices(), entries(), num_entries);
      for (size_t r = 0; r < num_entries; ++r)
        if (indices[r] != row_gid)
          J->replaceGlobalValues(indices[r], index(), entry());
    }
  }

}

} // namespace PHAL
