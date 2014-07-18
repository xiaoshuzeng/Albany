//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SHALLOWWATERSOURCE_HPP
#define AERAS_SHALLOWWATERSOURCE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 
#include "Albany_Layouts.hpp"

namespace Aeras {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ShallowWaterSource : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>,
		    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  ShallowWaterSource(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n); 
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

private:


  // Input:// og what does this mean, input???
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> sphere_coord;

  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> U;  //vecDim works but its really Dim+1
  PHX::MDField<ScalarT,Cell,Node,VecDim> UNodal;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> UDot;
          
  PHX::MDField<ScalarT,Cell,QuadPoint> mountainHeight;
          
  PHX::MDField<MeshScalarT,Cell,QuadPoint> weighted_measure;
          
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> jacobian_inv;
  PHX::MDField<MeshScalarT,Cell,QuadPoint> jacobian_det;
  Intrepid::FieldContainer<RealType>    grad_at_cub_points;

  enum SOURCETYPE {NONE, TC4};
  SOURCETYPE sourceType;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> source;

          
          
  std::size_t numQPs, numDims, numNodes, vecDim, spatialDim;
          
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  Intrepid::FieldContainer<RealType>    refPoints;
  Intrepid::FieldContainer<RealType>    refWeights;
  Intrepid::FieldContainer<MeshScalarT>  nodal_jacobian;
  Intrepid::FieldContainer<MeshScalarT>  nodal_inv_jacobian;
  Intrepid::FieldContainer<MeshScalarT>  nodal_det_j;
          
          
  //PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> source;
          
  ScalarT gravity; // gravity parameter -- Sacado-ized for sensitivities
  ScalarT Omega;   //rotation of earth  -- Sacado-ized for sensitivities
          
  void divergence(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
                          std::size_t cell, Intrepid::FieldContainer<ScalarT>  & div);
          
  //gradient returns vector in physical basis
  void gradient(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
                        std::size_t cell, Intrepid::FieldContainer<ScalarT>  & gradField);
          
  // curl only returns the component in the radial direction
  void curl(const Intrepid::FieldContainer<ScalarT>  & fieldAtNodes,
                    std::size_t cell, Intrepid::FieldContainer<ScalarT>  & curl);
          
  void fill_nodal_metrics(std::size_t cell);
          
  void get_coriolis(std::size_t cell, Intrepid::FieldContainer<ScalarT>  & coriolis);
          
          
          
};

}

#endif
