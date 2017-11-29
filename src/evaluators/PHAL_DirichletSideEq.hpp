//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DIRICHLET_SIDE_EQ_HPP
#define PHAL_DIRICHLET_SIDE_EQ_HPP

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"

#include "Teuchos_ParameterList.hpp"

#include "Sacado_ParameterAccessor.hpp"

#include <string>

namespace PHAL {

template<typename EvalT, typename Traits>
class DirichletSideEqBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  using ScalarT = typename EvalT::ScalarT;

  DirichletSideEqBase(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData /*d*/,
                             PHX::FieldManager<Traits>& /*vm*/) {}

  // This function will be overloaded
  virtual void evaluateFields(typename Traits::EvalData d) = 0;

protected:

  std::string dirName;
  std::string sideSetName;

  int dof_offset;
};

// ===================== Single-value Dirichlet for Side Eq ==================== //


template<typename EvalT, typename Traits>
class DirichletSideEq : public DirichletSideEqBase<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
public:

  using ScalarT = typename EvalT::ScalarT;

  DirichletSideEq (Teuchos::ParameterList& p);

  void evaluateFields(typename Traits::EvalData d);

  virtual ScalarT& getValue(const std::string &n) {
    if (n==this->dirName) {
      return value;
    } else {
      static ScalarT dummy;
      return dummy;
    }
  }

private:

  ScalarT value;

  std::string nodeSetName;
};

} // namespace PHAL

#endif // PHAL_DIRICHLET_SIDE_EQ_HPP
