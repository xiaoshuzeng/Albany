//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokes>
HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, IsStokes>::
HydrologyWaterDischarge (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  gradPhi (p.get<std::string> ("Hydraulic Potential Gradient Variable Name"), dl->qp_gradient),
  h       (p.get<std::string> ("Water Thickness Variable Name"), dl->qp_scalar),
  q       (p.get<std::string> ("Water Discharge Variable Name"), dl->qp_gradient)
{
  if (IsStokes) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, std::logic_error,
                                "Error! For coupling with StokesFO, the Layouts structure must be that of the basal side.\n");

    sideSetName = p.get<std::string>("Side Set Name");

    numQPs  = dl->qp_gradient->dimension(2);
    numDim  = dl->qp_gradient->dimension(3);
  } else {
    numQPs  = dl->qp_gradient->dimension(1);
    numDim  = dl->qp_gradient->dimension(2);
  }

  this->addDependentField(gradPhi);
  this->addDependentField(h);

  this->addEvaluatedField(q);

  // Setting parameters
  Teuchos::ParameterList& hydrology = *p.get<Teuchos::ParameterList*>("FELIX Hydrology");
  Teuchos::ParameterList& physics   = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  double rho_w = physics.get<double>("Water Density");
  double g     = physics.get<double>("Gravity Acceleration");
  k_0   = hydrology.get<double>("Transmissivity");
  //alpha = hydrology.get<double>("Water Thickness Exponent (alpha)",3);
  //beta  = hydrology.get<double>("Potential Gradient Exponent (beta)",2) - 2.0;

  k_0 /= (rho_w * g);

  regularize = hydrology.get<bool>("Regularize With Continuation", false);
  if (regularize)
  {
    regularizationParam = PHX::MDField<ScalarT,Dim>(p.get<std::string>("Regularization Parameter Name"),dl->shared_param);
    this->addDependentField(regularizationParam);
  }

/*
  needsGradPhiNorm = false;
  if (beta!=0.0)
  {
    needsGradPhiNorm = true;
    gradPhiNorm = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string>("Hydraulic Potential Gradient Norm QP Variable Name"), dl->qp_scalar);
    this->addDependentField(gradPhiNorm);
  }
*/
  this->setName("HydrologyWaterDischarge"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokes>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, IsStokes>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(gradPhi,fm);
  this->utils.setFieldData(h,fm);
  if (regularize)
  {
    this->utils.setFieldData(regularizationParam,fm);
  }
/*
  if (needsGradPhiNorm)
  {
    this->utils.setFieldData(gradPhiNorm,fm);
  }
*/
  this->utils.setFieldData(q,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokes>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, IsStokes>::evaluateFields (typename Traits::EvalData workset)
{
  if (IsStokes) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokes>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, IsStokes>::evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT regularization(0.0);
  if (regularize)
  {
    regularization = std::pow(10.0,-10*regularizationParam(0));
  }

/*
  // q = - k h^3 \nabla (phiH-N)
  if (needsGradPhiNorm)
  {
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
          q(cell,qp,dim) = -k_0 * std::pow(h(cell,qp),alpha) * std::pow(gradPhiNorm(cell,qp),beta) * gradPhi(cell,qp,dim);
        }
      }
    }
  }
  else
*/
  {
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
//          q(cell,qp,dim) = -k_0 * std::pow(h(cell,qp),alpha) * gradPhi(cell,qp,dim);
          q(cell,qp,dim) = -k_0 * std::pow(h(cell,qp)+regularization,3) * gradPhi(cell,qp,dim);
        }
      }
    }
  }
}

template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokes>
void HydrologyWaterDischarge<EvalT, Traits, HasThicknessEqn, IsStokes>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  ScalarT regularization(0.0);
  if (regularize)
  {
    regularization = std::pow(10.0,-10*regularizationParam(0));
  }

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
/*
  if (needsGradPhiNorm)
  {
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int qp=0; qp < numQPs; ++qp)
      {
        for (int dim(0); dim<numDim; ++dim)
        {
          q(cell,side,qp,dim) = -k_0 * std::pow(h(cell,side,qp),alpha) * std::pow(gradPhiNorm(cell,side,qp),beta) * gradPhi(cell,side,qp,dim);
        }
      }
    }
  }
  else
*/
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    for (int qp=0; qp < numQPs; ++qp)
    {
      for (int dim(0); dim<numDim; ++dim)
      {
        //q(cell,side,qp,dim) = -k_0 * std::pow(h(cell,side,qp),alpha) * gradPhi(cell,side,qp,dim);
        q(cell,side,qp,dim) = -k_0 * std::pow(h(cell,side,qp)+regularization,3) * gradPhi(cell,side,qp,dim);
      }
    }
  }
}

} // Namespace FELIX
