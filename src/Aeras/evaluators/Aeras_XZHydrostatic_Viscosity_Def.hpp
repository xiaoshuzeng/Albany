//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_Viscosity<EvalT, Traits>::
XZHydrostatic_Viscosity(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  GradBF     (p.get<std::string>("Gradient BF Name"),  dl->node_qp_gradient),
  GradV      (p.get<std::string>("Gradient Vel Name"), dl->qp_gradient_level),
  Viscosity  (p.get<std::string>("Viscosity Name"),    dl->qp_gradient_level),

  coefficient(p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Viscosity",0.0)),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2))

{
  this->addDependentField(GradBF);
  this->addDependentField(GradV);
  this->addEvaluatedField(Viscosity);
  this->setName("Aeras::XZHydrostatic_Viscosity"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Viscosity<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(GradBF    , fm);
  this->utils.setFieldData(GradV     , fm);
  this->utils.setFieldData(Viscosity , fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_Viscosity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        for (int dim=0; dim<numDims; dim++) {
          Viscosity(cell,qp,level,dim) = 0;
          for (int node=0; node < numNodes; ++node) {
            Viscosity(cell,qp,level,dim) += coefficient * GradV(cell,qp,level,dim)*GradBF(cell, node, qp, dim);
          }
        }
      }
    }
  }
}

}
