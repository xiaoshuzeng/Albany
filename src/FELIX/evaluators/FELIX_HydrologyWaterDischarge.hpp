//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_WATER_DISCHARGE_HPP
#define FELIX_HYDROLOGY_WATER_DISCHARGE_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokes>
class HydrologyWaterDischarge : public PHX::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT       ScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT;
  typedef typename std::conditional<HasThicknessEqn,ScalarT,ParamScalarT>::type hScalarT;

  HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input:
  PHX::MDField<const ScalarT>       gradPhi;
  PHX::MDField<const ScalarT>       gradPhiNorm;
  PHX::MDField<const hScalarT>      h;
  PHX::MDField<const ScalarT,Dim>   regularizationParam;

  // Output:
  PHX::MDField<ScalarT>   q;

  int numQPs;
  int numDim;
  std::string   sideSetName;

  double k_0;

  // These two would enable a more general case, where instead of (h^3 \grad \Phi)
  // we would have (h^\alpha |\grad \Phi|^\beta \grad \Phi)
  //double alpha;
  //double beta;

  bool needsGradPhiNorm;
  bool regularize;
};

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_WATER_DISCHARGE_HPP
