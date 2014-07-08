//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec         (p.get<std::string>  ( "Coordinate Vector Name"    ), dl->vertices_vector ),
  weighted_measure (p.get<std::string>  ( "Weights Name"              ), dl->qp_scalar ),
  jacobian_det     (p.get<std::string>  ( "Jacobian Det Name"         ), dl->qp_scalar ),
  BF               (p.get<std::string>  ( "BF Name"                   ), dl->node_qp_scalar),
  wBF              (p.get<std::string>  ( "Weighted BF Name"          ), dl->node_qp_scalar),
  GradBF           (p.get<std::string>  ( "Gradient BF Name"          ), dl->node_qp_gradient),
  wGradBF          (p.get<std::string>  ( "Weighted Gradient BF Name" ), dl->node_qp_gradient),
  topoName         (p.get<std::string>  ( "Topology Variable Name"    )),
  cubature         (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  intrepidBasis    (p.get<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis") ),
  cellType         (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type"))
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(jacobian_det);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  int containerSize = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];


  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  // Allocate Temporary FieldContainers
  val_at_cub_points.resize(numNodes, numQPs);
  grad_at_cub_points.resize(numNodes, numQPs, numDims);
  refPoints.resize(numQPs, numDims);
  refWeights.resize(numQPs);
  jacobian.resize(containerSize, numQPs, numDims, numDims);
  jacobian_inv.resize(containerSize, numQPs, numDims, numDims);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(val_at_cub_points, refPoints, Intrepid::OPERATOR_VALUE);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid::OPERATOR_GRAD);

  this->setName("ComputeBasisFunctions"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(jacobian_det,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions_ElementTopo<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  /** The allocated size of the Field Containers must currently 
    * match the full workset size of the allocated PHX Fields, 
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */
  
  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!
  Intrepid::CellTools<RealType>::setJacobian(this->jacobian, this->refPoints, 
                                             this->coordVec, *(this->cellType));
  Intrepid::CellTools<MeshScalarT>::setJacobianInv(this->jacobian_inv, this->jacobian);
  Intrepid::CellTools<MeshScalarT>::setJacobianDet(this->jacobian_det, this->jacobian);

  Intrepid::FunctionSpaceTools::HGRADtransformVALUE<RealType>
    (this->BF, this->val_at_cub_points);
  Intrepid::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>
    (this->GradBF, this->jacobian_inv, this->grad_at_cub_points);

  Intrepid::FunctionSpaceTools::computeCellMeasure<MeshScalarT>
    (this->weighted_measure, this->jacobian_det, this->refWeights);

  // adjust weighted_measure to account for topology
  int nQPs = this->numQPs;
  Albany::MDArray topology = (*workset.stateArrayPtr)[this->topoName];
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < nQPs; ++qp)
      this->weighted_measure(cell,qp) *= topology(cell);
 
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (this->wBF, this->weighted_measure, this->BF);
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (this->wGradBF, this->weighted_measure, this->GradBF);



}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions_NodeTopo<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  /** The allocated size of the Field Containers must currently 
    * match the full workset size of the allocated PHX Fields, 
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int containerSize = workset.numCells;
    */
  
  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!
  Intrepid::CellTools<RealType>::setJacobian(this->jacobian, this->refPoints, 
                                             this->coordVec, *(this->cellType));
  Intrepid::CellTools<MeshScalarT>::setJacobianInv(this->jacobian_inv, this->jacobian);
  Intrepid::CellTools<MeshScalarT>::setJacobianDet(this->jacobian_det, this->jacobian);

  Intrepid::FunctionSpaceTools::HGRADtransformVALUE<RealType>
    (this->BF, this->val_at_cub_points);
  Intrepid::FunctionSpaceTools::HGRADtransformGRAD<MeshScalarT>
    (this->GradBF, this->jacobian_inv, this->grad_at_cub_points);

  Intrepid::FunctionSpaceTools::computeCellMeasure<MeshScalarT>
    (this->weighted_measure, this->jacobian_det, this->refWeights);

  // adjust weighted_measure to account for topology
  int nQPs = this->numQPs;
  int nNodes = this->numNodes;
  Albany::MDArray topology = (*workset.stateArrayPtr)[this->topoName];
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < nQPs; ++qp){
      RealType topo_weight = 0.0;
      for (std::size_t node=0; node < nNodes; ++node)
        topo_weight += topology(cell,node) * this->BF(cell, node, qp);
      this->weighted_measure(cell,qp) *= topo_weight;
    }
 
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (this->wBF, this->weighted_measure, this->BF);
  Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
    (this->wGradBF, this->weighted_measure, this->GradBF);



}

}  // end ATO namespace
