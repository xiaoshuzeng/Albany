//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_DirichletCondensed.hpp"

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
}

template <typename Traits>
void DirichletCondensed<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(EvalData workset) {
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
}

template <typename Traits>
void DirichletCondensed<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(EvalData workset) {
}

} // namespace PHAL
