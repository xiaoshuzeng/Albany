//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Optimizer.hpp"
#include "Teuchos_TestForException.hpp"
#include "ATO_Solver.hpp"
#include <algorithm>

namespace ATO {


/**********************************************************************/
Teuchos::RCP<Optimizer> 
OptimizerFactory::create(const Teuchos::ParameterList& optimizerParams)
/**********************************************************************/
{
  std::string optMethod = optimizerParams.get<std::string>("Method");
  if( optMethod == "OC"  )  return Teuchos::rcp(new Optimizer_OC(optimizerParams));
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Optimization method: " << optMethod << " Unknown!" << std::endl 
      << "Valid options are (OC)" << std::endl);
}

/**********************************************************************/
Optimizer::Optimizer(const Teuchos::ParameterList& optimizerParams)
/**********************************************************************/
{ 
  _optConvTol = optimizerParams.get<double>("Optimization Convergence Tolerance");
  _optMaxIter = optimizerParams.get<int>("Optimization Maximum Iterations");

  solverInterface = NULL;
}

/**********************************************************************/
Optimizer_OC::Optimizer_OC(const Teuchos::ParameterList& optimizerParams) :
Optimizer(optimizerParams)
/**********************************************************************/
{ 
  _volConvTol    = optimizerParams.get<double>("Volume Enforcement Convergence Tolerance");
  _volMaxIter    = optimizerParams.get<int>("Volume Enforcement Maximum Iterations");
  _minDensity    = optimizerParams.get<double>("Minimum Density");
  _initLambda    = optimizerParams.get<double>("Volume Multiplier Initial Guess");
  _volConstraint = optimizerParams.get<double>("Volume Fraction Constraint");
  _moveLimit     = optimizerParams.get<double>("Move Limiter");
  _stabExponent  = optimizerParams.get<double>("Stabilization Parameter");
}

/**********************************************************************/
void
Optimizer::SetInterface(Solver* mySolverInterface)
/**********************************************************************/
{
  solverInterface = dynamic_cast<OptInterface*>(mySolverInterface);
  TEUCHOS_TEST_FOR_EXCEPTION(
    solverInterface == NULL, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error! Dynamic cast of Solver* to OptInterface* failed." << std::endl);
}

/******************************************************************************/
Optimizer_OC::~Optimizer_OC()
/******************************************************************************/
{
  if( p      ) delete [] p;
  if( p_last ) delete [] p_last;
  if( dfdp   ) delete [] dfdp;
}

/******************************************************************************/
double
Optimizer_OC::computeNorm()
/******************************************************************************/
{
  double norm = 0.0;
  for(int i=0; i<numOptDofs; i++){
    norm += pow(p[i]-p_last[i],2);
  }
  return (norm > 0.0) ? sqrt(norm) : 0.0;
}

/******************************************************************************/
void
Optimizer_OC::Initialize()
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION (
    solverInterface == NULL, Teuchos::Exceptions::InvalidParameter, 
    std::endl << "Error! Optimizer requires valid Solver Interface" << std::endl);

  numOptDofs = solverInterface->GetNumOptDofs();

  p      = new double[numOptDofs];
  p_last = new double[numOptDofs];
  dfdp   = new double[numOptDofs];

  std::fill_n(p,      numOptDofs, _volConstraint);
  std::fill_n(p_last, numOptDofs, 0.0);
  std::fill_n(dfdp,   numOptDofs, 0.0);

  solverInterface->ComputeVolume(_optVolume);
}

/******************************************************************************/
void
Optimizer_OC::Optimize()
/******************************************************************************/
{

  int iter=0;
  bool optimization_converged = false;

  while(!optimization_converged && iter < _optMaxIter) {

    solverInterface->ComputeObjective(p, f, dfdp);

    computeUpdatedTopology();

    // check for convergence
    double delta_p = computeNorm();
    if( delta_p < _optConvTol ) optimization_converged = true;

    iter++;
  }

  return;
}



/******************************************************************************/
void
Optimizer_OC::computeUpdatedTopology()
/******************************************************************************/
{

  // find multiplier that enforces volume constraint
  const double maxDensity = 1.0;
  double vmid, v1=0.0;
  double v2=_initLambda;
  int niters=0;

  for(int i=0; i<numOptDofs; i++)
    p_last[i] = p[i];

  double vol = 0.0;
  do {
    TEUCHOS_TEST_FOR_EXCEPTION(
      niters > _volMaxIter, Teuchos::Exceptions::InvalidParameter, 
      std::endl << "Enforcement of volume constraint failed:  Exceeded max iterations" 
      << std::endl);

    vol = 0.0;
    vmid = (v2+v1)/2.0;

    // update topology
    for(int i=0; i<numOptDofs; i++) {
      double be = dfdp[i]/vmid;
      double p_old = p_last[i];
      double p_new = p_old*pow(be,_stabExponent);
      // limit change
      double dval = p_new - p_old;
      if( fabs(dval) > _moveLimit) p_new = p_old+fabs(dval)/dval*_moveLimit;
      // enforce limits
      if( p_new < _minDensity ) p_new = _minDensity;
      if( p_new > maxDensity ) p_new = maxDensity;
      p[i] = p_new;
    }

    // compute new volume
    solverInterface->ComputeVolume(p, vol);
    if( (vol - _volConstraint*_optVolume) > 0.0 ) v1 = vmid;
    else v2 = vmid;
    niters++;
  } while ( fabs(vol - _volConstraint*_optVolume) > _volConvTol );
}


}

