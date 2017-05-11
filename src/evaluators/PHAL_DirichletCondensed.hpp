//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DIRICHLET_CONDENSED_HPP
#define PHAL_DIRICHLET_CONDENSED_HPP

#include "PHAL_Dirichlet.hpp"

namespace PHAL {

// ****************************************************************************
// DEFAULT - DO NOTHING 
// ****************************************************************************

template <typename EvalT, typename Traits>
class DirichletCondensed
  : public DirichletBase<EvalT, Traits> {
 public:
   DirichletCondensed(Teuchos::ParameterList& p);
   void evaluateFields(typename Traits::EvalData d);
};

// ****************************************************************************
// RESIDUAL SPECIALIZATION
// ****************************************************************************

template <typename Traits>
class DirichletCondensed<PHAL::AlbanyTraits::Residual, Traits>
  : public DirichletBase<PHAL::AlbanyTraits::Residual, Traits> {

  public:

    using PreEvalData = typename Traits::PreEvalData;
    using EvalData = typename Traits::EvalData;
    using ScalarT = PHAL::AlbanyTraits::Residual::ScalarT;

    DirichletCondensed(Teuchos::ParameterList& p);
    void preEvaluate(PreEvalData workset);
    void evaluateFields(EvalData workset);
};

// ****************************************************************************
// JACOBIAN SPECIALIZATION
// ****************************************************************************

template <typename Traits>
class DirichletCondensed<PHAL::AlbanyTraits::Jacobian, Traits>
  : public DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits> {

  public:

    using PreEvalData = typename Traits::PreEvalData;
    using EvalData = typename Traits::EvalData;
    using ScalarT = PHAL::AlbanyTraits::Jacobian::ScalarT;

    DirichletCondensed(Teuchos::ParameterList& p);
    void preEvaluate(PreEvalData workset);
    void evaluateFields(EvalData workset);
};

} // namespace PHAL

#endif
