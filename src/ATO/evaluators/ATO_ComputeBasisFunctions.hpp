//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_COMPUTEBASISFUNCTIONS_HPP
#define ATO_COMPUTEBASISFUNCTIONS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"


namespace ATO {

/** \brief Finite Element Interpolation Evaluator (Base Class)

    This evaluator interpolates nodal DOF values to quad points.  Gauss weights are 
    modified based on the topology.

*/

template<typename EvalT, typename Traits>
class ComputeBasisFunctions : 
       public PHX::EvaluatorWithBaseImpl<Traits>,
       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ComputeBasisFunctions(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d);

protected:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numVertices, numDims, numNodes, numQPs;

  std::string topoName;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType> val_at_cub_points;
  Intrepid::FieldContainer<RealType> grad_at_cub_points;
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;
  Intrepid::FieldContainer<MeshScalarT> jacobian;
  Intrepid::FieldContainer<MeshScalarT> jacobian_inv;

  // Output:
  //! Basis Functions at quadrature points
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det; 
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
};



/** \brief Finite Element Interpolation Evaluator (w/ Element Based Topology)

    This evaluator interpolates nodal DOF values to quad points.  Gauss weights are 
    multiplied by the element based topology value.

*/

template<typename EvalT, typename Traits>
class ComputeBasisFunctions_ElementTopo : public PHX::EvaluatorWithBaseImpl<Traits>,
                                          public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  ComputeBasisFunctions_ElementTopo(const Teuchos::ParameterList& p,
                                    const Teuchos::RCP<Albany::Layouts>& dl):
                                    ComputeBasisFunctions(p,dl){}

  void evaluateFields(typename Traits::EvalData d);
};

/** \brief Finite Element Interpolation Evaluator (w/ Node Based Topology)

    This evaluator interpolates nodal DOF values to quad points.  Gauss weights are 
    multiplied by the node based topology value.

*/

template<typename EvalT, typename Traits>
class ComputeBasisFunctions_NodeTopo : public PHX::EvaluatorWithBaseImpl<Traits>,
                                       public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  ComputeBasisFunctions_NodeTopo(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl):
                                 ComputeBasisFunctions(p,dl){}

  void evaluateFields(typename Traits::EvalData d);
};

}

#endif
