//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp" 

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

const double pi = 3.1415926535897932385;
 
//**********************************************************************
template<typename EvalT, typename Traits>
ShallowWaterSource<EvalT, Traits>::
ShallowWaterSource(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  U        (p.get<std::string> ("QP Variable Name"), dl->qp_vector),
  UNodal   (p.get<std::string> ("Nodal Variable Name"), dl->node_vector),
  Ugrad    (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_vecgradient),
  UDot     (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_vector),
  mountainHeight  (p.get<std::string> ("Aeras Surface Height QP Variable Name"), dl->qp_scalar),
  jacobian_inv  (p.get<std::string>  ("Jacobian Inv Name"), dl->qp_tensor ),
  jacobian_det  (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  weighted_measure (p.get<std::string>  ("Weights Name"),   dl->qp_scalar ),
  jacobian  (p.get<std::string>  ("Jacobian Name"), dl->qp_tensor ),
  intrepidBasis (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  spatialDim(p.get<std::size_t>("spatialDim")),
  sphere_coord  (p.get<std::string>  ("Spherical Coord Name"), dl->qp_gradient ),
  gravity (Aeras::ShallowWaterConstants::self().gravity),
  Omega(2.0*(Aeras::ShallowWaterConstants::self().pi)/(24.*3600.)),
  //OG should source be node vector? like residual?
  source    (p.get<std::string> ("Shallow Water Source QP Variable Name"), dl->qp_vector)
{
  Teuchos::ParameterList* shallowWaterList =
   p.get<Teuchos::ParameterList*>("Parameter List");

   std::string sourceTypeString = shallowWaterList->get<std::string>("SourceType", "None");

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  const bool invalidString = (sourceTypeString != "None" && sourceTypeString != "TC4");
  TEUCHOS_TEST_FOR_EXCEPTION( invalidString,
                                  std::logic_error,
                                  "Unknown shallow water source string of " << sourceTypeString
                                          << " encountered. " << std::endl);

  if (sourceTypeString == "None"){
    sourceType = NONE;
  }
  else if (sourceTypeString == "TC4") {
   *out << "Setting TC4 source.  To be implemented by Oksana." << std::endl;
   sourceType = TC4;

  }

  this->addDependentField(sphere_coord);
  this->addEvaluatedField(source);
  
  
  //should i move it under if statement?
  this->addDependentField(U);
  this->addDependentField(UNodal);
  this->addDependentField(Ugrad);
  this->addDependentField(UDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  //this->addDependentField(GradBF);
  this->addDependentField(mountainHeight);
  
  this->addDependentField(weighted_measure);
  this->addDependentField(jacobian);
  this->addDependentField(jacobian_inv);
  this->addDependentField(jacobian_det);
  
  
  

  std::vector<PHX::DataLayout::size_type> dims;
  
  //dl->qp_gradient->dimensions(dims);
  //numQPs  = dims[1];
  //numDims = dims[2];
  //dl->qp_vector->dimensions(dims);
  //vecDim  = dims[2]; //# of dofs/node

  this->setName("ShallowWaterSource"+PHX::TypeString<EvalT>::value);
  
  
  //why dims from grad phi? what is in dims[0]
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
  
  refWeights        .resize               (numQPs);
  grad_at_cub_points.resize     (numNodes, numQPs, 2);
  refPoints         .resize               (numQPs, 2);
  nodal_jacobian.resize(numNodes, 2, 2);
  nodal_inv_jacobian.resize(numNodes, 2, 2);
  nodal_det_j.resize(numNodes);
  
  cubature->getCubature(refPoints, refWeights);
  
  //?
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);
  
  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];
  
  std::vector<PHX::DataLayout::size_type> gradDims;
  wGradBF.fieldTag().dataLayout().dimensions(gradDims);
  
  
  gradDims.clear();
  Ugrad.fieldTag().dataLayout().dimensions(gradDims);
  
  
  
  
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(source,fm);
  this->utils.setFieldData(sphere_coord,fm);
  
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(UNodal,fm);
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(UDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  //this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(mountainHeight,fm);
  
  this->utils.setFieldData(weighted_measure, fm);
  this->utils.setFieldData(jacobian, fm);
  this->utils.setFieldData(jacobian_inv, fm);
  this->utils.setFieldData(jacobian_det, fm);
  
  
}

//**********************************************************************
//A concrete (non-virtual) implementation of getValue is needed for code to compile. 
//Do we really need it though for this problem...?
template<typename EvalT,typename Traits>
typename ShallowWaterSource<EvalT,Traits>::ScalarT& 
ShallowWaterSource<EvalT,Traits>::getValue(const std::string &n)
{
  static ScalarT junk(0);
  return junk;
}


//**********************************************************************
template<typename EvalT, typename Traits>
void ShallowWaterSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if (sourceType == NONE) {
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        for (std::size_t i = 0; i < vecDim; ++i) { //loop over # of dofs/node
          source(cell, qp, i) = 0;
        }
      }
    }
  }
  else if(sourceType == TC4) {
  std::cout << "In evaluateFields for TC4." << std::endl; 
  const RealType time = workset.current_time; //current time from workset
    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t qp = 0; qp < numQPs; ++qp) {
        for (std::size_t i = 0; i < vecDim; ++i) { //loop over # of dofs/node
          const MeshScalarT theta = sphere_coord(cell, qp, 0);
          const MeshScalarT lambda = sphere_coord(cell, qp, 1);
          source(cell, qp, i) = 0.0; //set this to some function involving time 
        }
      }
    }
  }
}
}
