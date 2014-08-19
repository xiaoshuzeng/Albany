//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Aggregator.hpp"
#include "Teuchos_TestForException.hpp"

namespace ATO {


//**********************************************************************
Teuchos::RCP<Aggregator> AggregatorFactory::create(const Teuchos::ParameterList& aggregatorParams)
{
  // if there's only one response function ...
  
  Teuchos::Array<std::string> objectives = 
    aggregatorParams.get<Teuchos::Array<std::string> >("Objectives");

  if (objectives.size() == 1) return Teuchos::rcp(new Aggregator_PassThru(aggregatorParams));

  TEUCHOS_TEST_FOR_EXCEPTION(
    true, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Objective aggregators not implemented yet." << std::endl);

/*
  std::string weightingType = aggregatorParams.get<std::string>("Weighting");
  if( weightingType == "Uniform"  )  return Teuchos::rcp(new Aggregator_Uniform(aggregatorParams));
  if( weightingType == "Weighted" )  return Teuchos::rcp(new Aggregator_Weighted(aggregatorParams));
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Weighting type " << weightingType << " Unknown!" << std::endl 
      << "Valid weighting types are (Uniform, Weighted)" << std::endl);
*/
}

//**********************************************************************
Aggregator::Aggregator(const Teuchos::ParameterList& aggregatorParams)
{ 
  aggregatedVariablesNames = aggregatorParams.get<Teuchos::Array<std::string> >("Objectives");
}

//**********************************************************************
Aggregator_PassThru::Aggregator_PassThru(const Teuchos::ParameterList& aggregatorParams) :
Aggregator(aggregatorParams)
{ 
  variableName = aggregatedVariablesNames[0];
}


}

