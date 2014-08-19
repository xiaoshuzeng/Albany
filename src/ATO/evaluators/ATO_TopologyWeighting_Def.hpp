//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

// JR:  currently hardwired for element centered topology.

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
TopologyWeighting<EvalT, Traits>::
TopologyWeighting(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
{
  const Teuchos::ParameterList& topoParams = p.get<Teuchos::ParameterList>("Topology");

  ATO::TopoToolsFactory topoFactory;
  topoTools = topoFactory.create(topoParams);

  topoName = topoParams.get<std::string>("Topology Name");

  std::string strLayout = p.get<std::string>("Variable Layout");
 
  Teuchos::RCP<PHX::DataLayout> layout;
  if(strLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(strLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown variable layout " << strLayout <<
                               "!" << std::endl << "Options are (QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<ScalarT> _unWeightedVar(p.get<std::string>("Unweighted Variable Name"), layout);
  unWeightedVar = _unWeightedVar;
  PHX::MDField<ScalarT> _weightedVar(p.get<std::string>("Weighted Variable Name"), layout);
  weightedVar = _weightedVar;


  // Pull out numQPs and numDims from a Layout
  std::vector<int> dims;
  layout->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(unWeightedVar);
  this->addEvaluatedField(weightedVar);

  this->setName("Topology Weighting"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void TopologyWeighting<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(unWeightedVar,fm);
  this->utils.setFieldData(weightedVar,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TopologyWeighting<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::vector<int> dims;
  unWeightedVar.dimensions(dims);
  int size = dims.size();

  Albany::MDArray topo = (*workset.stateArrayPtr)[this->topoName];

  switch(size) {
    case 3:
      for(int cell=0; cell<dims[0]; cell++){
        double P = topoTools->Penalize(topo(cell));
        for(int qp=0; qp<dims[1]; qp++)
          for(int i=0; i<dims[2]; i++)
            weightedVar(cell,qp,i) = P*unWeightedVar(cell,qp,i);
      }
      break;
    case 4:
      for(int cell=0; cell<dims[0]; cell++){
        double P = topoTools->Penalize(topo(cell));
        for(int qp=0; qp<dims[1]; qp++)
          for(int i=0; i<dims[2]; i++)
            for(int j=0; j<dims[3]; j++)
              weightedVar(cell,qp,i,j) = P*unWeightedVar(cell,qp,i,j);
      }
      break;
    default:
     TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>4, Teuchos::Exceptions::InvalidParameter,
       "Unexpected array dimensions in TopologyWeighting:" << size << std::endl);


}

//**********************************************************************
}

}
