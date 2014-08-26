//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_Optimizer_HPP
#define ATO_Optimizer_HPP

#include "Albany_StateManager.hpp"

#include <string>
#include <vector>

#include "Teuchos_ParameterList.hpp"


namespace ATO {

class Solver;
class OptInterface;

class Optimizer 
/** \brief Optimizer wrapper

    This class provides a very basic container for topological optimization algorithms.

*/
{
 public:
  Optimizer(const Teuchos::ParameterList& optimizerParams);
  virtual ~Optimizer(){};

  virtual void Optimize()=0;
  virtual void Initialize()=0;
  virtual void SetInterface(Solver*);
 protected:

  OptInterface* solverInterface;

  double _optConvTol;
  double _optMaxIter;

};


class Optimizer_OC : public Optimizer {
 public:
  Optimizer_OC(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_OC();
  void Optimize();
  void Initialize();
 private:
  void computeUpdatedTopology();
  double computeNorm();

  double* p;
  double* p_last;
  double f;
  double* dfdp;
  int numOptDofs;

  double _volConvTol;
  double _volMaxIter;
  double _minDensity;
  double _initLambda;
  double _moveLimit;
  double _stabExponent;
  double _volConstraint;
  double _optVolume;

};


class OptimizerFactory {
 public:
  Teuchos::RCP<Optimizer> create(const Teuchos::ParameterList& optimizerParams);
};


}
#endif
