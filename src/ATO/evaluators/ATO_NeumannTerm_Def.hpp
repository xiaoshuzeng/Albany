//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
NeumannTerm<EvalT, Traits>::
NeumannTerm(const Teuchos::ParameterList& p) :
  outVector        (p.get<std::string>                   ("Neumann Force Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Neumann Force Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Neumann Force Data Layout");
  std::vector<PHX::Device::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  boundaryForceVector = p.get<Teuchos::Array<double> >("Traction Vector");

  this->addEvaluatedField(outVector);

  this->setName("NeumannTerm"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void NeumannTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(outVector,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NeumannTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t qp=0; qp < numQPs; ++qp)
      for (std::size_t i=0; i < numDims; ++i)
        outVector(cell,qp,i) = boundaryForceVector[i];
}

//**********************************************************************
}

