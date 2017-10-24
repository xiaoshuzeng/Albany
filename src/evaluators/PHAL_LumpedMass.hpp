//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_LUMPED_MASS_HPP
#define PHAL_LUMPED_MASS_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace PHAL
{

/** \brief Lumped Mass Evaluator

    This evaluator evaluates the local lumped mass

    The lumped mass is evaluated as a field, each entry corresponding
    to the diagonal entry of the lumped mass matrix.

    NOTE: We offer two choices for mass lumping:

      - ROW_SUM: (M_lumped)_ii = sum_j M_ij. This only works for linear elements
      - DIAG_SCALING: (M_lumped)_ii = M_ii / (sum_j M_jj). This works for lagrangian elements only.
*/

template<typename EvalT, typename Traits>
class LumpedMass : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  LumpedMass (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsOnCells (typename Traits::EvalData workset);
  void evaluateFieldsOnSides (typename Traits::EvalData workset);

  typedef typename EvalT::MeshScalarT   MeshScalarT;

  enum LumpingType {
    ROW_SUM      = 1,
    DIAG_SCALING = 2
  };

  LumpingType  lumping_type;

  int numNodes;
  int numQPs;

  bool onSide;
  std::string sideSetName;

  // Inputs:
  PHX::MDField<const MeshScalarT>    BF;
  PHX::MDField<const MeshScalarT>    w_measure;

  // Output:
  PHX::MDField<MeshScalarT>          lumped_mass;
};

} // namespace PHAL

#endif // PHAL_LUMPED_MASS_HPP
