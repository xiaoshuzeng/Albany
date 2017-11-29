//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef PHAL_DIRICHLET_OFF_SIDE_SET_HPP
#define PHAL_DIRICHLET_OFF_SIDE_SET_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Sacado_ParameterAccessor.hpp"

#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_DirichletSideEq.hpp"

namespace Albany {
  class NodalDOFManager;
}

namespace PHAL {

/** \brief Dirichlet evaluator for nodes outside a given (set of) side set(s)

    This evaluator is needed when the given problem has equations defined only on a side set.
    In that case, the Jacobian entries on the remaining nodes (not on the side set) MUST be
    handled (typically, J(dof,dof)=1, J(dof,:)=0 and res(dof)=x-datum), otherwise the linear
    solvers may complain (nan).
*/

template<typename EvalT, typename Traits>
class DirichletOffSideSets : public DirichletSideEqBase<EvalT,Traits>,
                             public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
public:
  using ScalarT = typename EvalT::ScalarT;

  DirichletOffSideSets(Teuchos::ParameterList& p);
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

  void gather_rows(typename Traits::EvalData d);

  std::set<GO> rows;
  std::vector<std::string> sideSetNames;

  ScalarT value;

  Teuchos::RCP<Albany::NodalDOFManager> dof_manager;
};

} // Namespace PHAL

#endif // PHAL_DIRICHLET_OFF_SIDE_SET_HPP
