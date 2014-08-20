//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_Aggregator_HPP
#define ATO_Aggregator_HPP

#include "Albany_StateManager.hpp"

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"

// this design assumes that all data are in the same state manager.  This may be
// a bad assumption.  If so, instead of sending in a state manager pointer to the
// Evaluate*(...) functions, create an 'AddAggratedVariable' function that takes
// a pointer to the associated state manager.  Then, Evaluate*(...) doesn't take any
// arguments, and the class has a list of pointers to the needed state managers.
//

namespace ATO {

class Aggregator 
/** \brief Combines objectives.

    This class reads objectives from response functions and combines them into
  a single objective for optimization.

*/
{

public:

  Aggregator(const Teuchos::ParameterList& aggregatorParams);
  virtual ~Aggregator(){};

  virtual void Evaluate(Albany::StateManager& stateManager)=0;
  virtual void EvaluateObjective(Albany::StateManager& stateManager)=0;
  virtual void EvaluateObjectiveDerivative(Albany::StateManager& stateManager)=0;

  virtual std::string getOutputVariableName(){return outputVariableName;}
protected:

  Teuchos::Array<std::string> aggregatedVariablesNames;
  std::string outputVariableName;

};


/* not implemented yet.
class Aggregator_Uniform : public Aggregator {
 public:
  Aggregator_Uniform(const Teuchos::ParameterList& aggregatorParams);
  virtual void EvaluateObjective();
  virtual void EvaluateObjectiveDerivative();
};

class Aggregator_Weighted : public Aggregator {
 public:
  Aggregator_Weighted(const Teuchos::ParameterList& aggregatorParams);
  virtual void EvaluateObjective();
  virtual void EvaluateObjectiveDerivative();
 private:
  std::vector<double> aggregatedVariablesWeights;
};
*/

class Aggregator_PassThru : public Aggregator {
 public:
  Aggregator_PassThru(const Teuchos::ParameterList& aggregatorParams);
  void Evaluate(Albany::StateManager& stateManager){}
  void EvaluateObjective(Albany::StateManager& stateManager){}
  void EvaluateObjectiveDerivative(Albany::StateManager& stateManager){}
  std::string getOutputVariableName(){return variableName;}
 private:
  std::string variableName;
};


class AggregatorFactory {
public:
  Teuchos::RCP<Aggregator> create(const Teuchos::ParameterList& aggregatorParams);
};


}
#endif
