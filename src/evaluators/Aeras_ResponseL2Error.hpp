//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_RESPONSE_L2_ERROR_HPP
#define AERAS_RESPONSE_L2_ERROR_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"


namespace Aeras {
/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseL2Error : 
    public PHAL::SeparableScatterScalarResponse<EvalT,Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ResponseL2Error(Teuchos::ParameterList& p,
			  const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    PHX::MDField<ScalarT,Cell,Node,VecDim> flow_state_field; //flow state field at nodes
    PHX::MDField<MeshScalarT> sphere_coord;
    PHX::MDField<MeshScalarT> weighted_measure;
    //! Basis Functions
    PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
    PHX::DataLayout::size_type field_rank;
    std::vector<PHX::DataLayout::size_type> field_dims;
    Teuchos::Array<int> field_components;
    std::size_t numQPs, numDims, numNodes, vecDim;

    std::vector<std::string> ebNames;    
  };
	
}

#endif
