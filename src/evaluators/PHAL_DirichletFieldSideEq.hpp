//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DIRICHLET_FIELD_SIDE_EQ_HPP
#define PHAL_DIRICHLET_FIELD_SIDE_EQ_HPP

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_DirichletSideEq.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
class DirichletFieldSideEq : public PHAL::DirichletSideEqBase<EvalT, Traits>
{
public:
  DirichletFieldSideEq(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

protected:
  std::string field_name;
  std::string nodeSetName;
};

} // namespace PHAL

#endif // PHAL_DIRICHLET_FIELD_SIDE_EQ_HPP
