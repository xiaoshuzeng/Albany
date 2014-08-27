//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Solver.hpp"
#include "ATO_OptimizationProblem.hpp"

/* GAH FIXME - Silence warning:
TRILINOS_DIR/../../../include/pecos_global_defs.hpp:17:0: warning: 
        "BOOST_MATH_PROMOTE_DOUBLE_POLICY" redefined [enabled by default]
Please remove when issue is resolved
*/
#undef BOOST_MATH_PROMOTE_DOUBLE_POLICY

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"

/******************************************************************************/
ATO::Solver::
Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
       const Teuchos::RCP<const Epetra_Comm>& comm,
       const Teuchos::RCP<const Epetra_Vector>& initial_guess)
: _solverComm(comm), _mainAppParams(appParams)
/******************************************************************************/
{
  zeroSet();


  ///*** PROCESS TOP LEVEL PROBLEM ***///

  // Validate Problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");
  _numPhysics = problemParams.get<int>("Number of Subproblems", 1);
  problemParams.validateParameters(*getValidProblemParameters(),0);

  // Parse and create aggregator
  Teuchos::ParameterList& aggregatorParams = 
    problemParams.get<Teuchos::ParameterList>("Objective Aggregator");
  ATO::AggregatorFactory aggregatorFactory;
  _aggregator = aggregatorFactory.create(aggregatorParams);

  // Parse and create optimizer
  Teuchos::ParameterList& optimizerParams = 
    problemParams.get<Teuchos::ParameterList>("Topological Optimization");
  ATO::OptimizerFactory optimizerFactory;
  _optimizer = optimizerFactory.create(optimizerParams);
  _optimizer->SetInterface(this);


  // Parse topology info
  Teuchos::ParameterList& topoParams = problemParams.get<Teuchos::ParameterList>("Topology");
  _topoCentering = topoParams.get<std::string>("Centering");
  _topoName = topoParams.get<std::string>("Topology Name");

  // Get and set the default Piro parameters from a file, if given
  std::string piroFilename  = problemParams.get<std::string>("Piro Defaults Filename", "");
  if(piroFilename.length() > 0) {
    const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(*comm);
    Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mpiComm);
    Teuchos::RCP<Teuchos::ParameterList> defaultPiroParams = 
      Teuchos::createParameterList("Default Piro Parameters");
    Teuchos::updateParametersFromXmlFileAndBroadcast(piroFilename, defaultPiroParams.ptr(), *tcomm);
    Teuchos::ParameterList& piroList = appParams->sublist("Piro", false);
    piroList.setParametersNotAlreadySet(*defaultPiroParams);
  }
  
  // set verbosity
  _is_verbose = (comm->MyPID() == 0) && problemParams.get<bool>("Verbose Output", false);

  // set optimization parameters
  _stabilizationExponent = problemParams.get<double>("Stabilization Exponent",0.5);
  _moveLimiter           = problemParams.get<double>("Move Limiter",0.2);
  _volumeConstraint      = problemParams.get<double>("Volume Constraint",0.5);
  



  ///*** PROCESS SUBPROBLEM(S) ***///
   
  _subProblemAppParams.resize(_numPhysics);
  _subProblem.resize(_numPhysics);
  for(int i=0; i<_numPhysics; i++){

    _subProblemAppParams[i] = createInputFile(appParams, i);
    _subProblem[i] = CreateSubSolver( _subProblemAppParams[i], *_solverComm);

    // ensure that all subproblems are topology based (i.e., optimizable)
    Teuchos::RCP<Albany::AbstractProblem> problem = _subProblem[i].app->getProblem();
    ATO::OptimizationProblem* atoProblem = 
      dynamic_cast<ATO::OptimizationProblem*>(problem.get());
    TEUCHOS_TEST_FOR_EXCEPTION( 
      atoProblem == NULL, Teuchos::Exceptions::InvalidParameter, std::endl 
      << "Error!  Requested subproblem does not support topologies." << std::endl);
  }


  // store a pointer to the first problem as an ATO::OptimizationProblem for callbacks
  Teuchos::RCP<Albany::AbstractProblem> problem = _subProblem[0].app->getProblem();
  _atoProblem = dynamic_cast<ATO::OptimizationProblem*>(problem.get());
  _atoProblem->setDiscretization(_subProblem[0].app->getDiscretization());

  _atoProblem->InitTopOpt();
  


  // get solution map from first subproblem
  const SolverSubSolver& sub = _subProblem[0];
  Teuchos::RCP<const Epetra_Map> sub_x_map = sub.app->getMap();
  TEUCHOS_TEST_FOR_EXCEPT( sub_x_map == Teuchos::null );
  _epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif
}


/******************************************************************************/
void
ATO::Solver::zeroSet()
/******************************************************************************/
{
  // set parameters and responses
  _num_parameters = 0; //TEV: assume no parameters or responses for now...
  _num_responses  = 0; //TEV: assume no parameters or responses for now...
}

  
/******************************************************************************/
void
ATO::Solver::evalModel(const InArgs& inArgs,
                       const OutArgs& outArgs ) const
/******************************************************************************/
{


  if(_is_verbose){
    Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
    *out << "*** Performing Topology Optimization Loop ***" << std::endl;
  }

  _optimizer->Initialize();

  _optimizer->Optimize();
 
}



/******************************************************************************/
///*************** SOLVER - OPTIMIZER INTERFACE FUNCTIONS *******************///
/******************************************************************************/


/******************************************************************************/
void
ATO::Solver::ComputeObjective(double* p, double& f, double* dfdp)
/******************************************************************************/
{
  for(int i=0; i<_numPhysics; i++){
    // copy data from p into each stateManager
    Albany::StateManager& stateMgr = _subProblem[i].app->getStateMgr();
    copyTopologyIntoStateMgr( p, stateMgr );

    // enforce PDE constraints
    _subProblem[i].model->evalModel((*_subProblem[i].params_in),
                                    (*_subProblem[i].responses_out));
  }


  // aggregate responses into a single objective
  // JR: This design clearly doesn't work for multiple subproblems.  Each 
  // subproblem has it's own state manager (I think) in which case we can't
  // send the state manager of the first subproblem as an argument.  Perhaps
  // during setup the subproblem data can be sent to the aggregator so 
  // the call below doesn't need arguments.  Where does the result go?  In
  // the statemanager of the first physics?  Can we get all subproblems to 
  // use the same statemanager?
  _aggregator->Evaluate(_subProblem[0].app->getStateMgr());

  
  
  // copy objective (f) and first derivative wrt the topology (dfdp) out 
  // of stateManager
  copyObjectiveFromStateMgr( f, dfdp );
  
}

/******************************************************************************/
void
ATO::Solver::copyTopologyIntoStateMgr( double* p, Albany::StateManager& stateMgr )
/******************************************************************************/
{
  // JR: This only works for element topology.  Generalize.
  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;

  int numWorksets = dest.size();

  int wsOffset = 0;
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& wsTopo = dest[ws][_topoName];
    int wsSize = wsTopo.size();
    for(int i=0; i<wsSize; i++)
      wsTopo[i] = p[wsOffset+i];
    wsOffset += wsSize;
  }
}

/******************************************************************************/
void
ATO::Solver::copyObjectiveFromStateMgr( double& f, double* dfdp )
/******************************************************************************/
{
  // JR: This only works for element topology.  Generalize.
  
  // f and dfdp are stored in subProblem[0]
  Albany::StateManager& stateMgr = _subProblem[0].app->getStateMgr();
  Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
  Albany::StateArrayVec& src = stateArrays.elemStateArrays;

  int numWorksets = src.size();

  std::string objName = _aggregator->getOutputVariableName();

  int wsOffset = 0;
  for(int ws=0; ws<numWorksets; ws++){
    Albany::MDArray& dfdpSrc = src[ws][objName];
    int wsSize = dfdpSrc.size();
    for(int i=0; i<wsSize; i++)
      dfdp[wsOffset+i] = dfdpSrc[i];
    wsOffset += wsSize;
  }
}
/******************************************************************************/
void
ATO::Solver::ComputeVolume(double& v)
/******************************************************************************/
{
  return _atoProblem->ComputeVolume(v);
}


/******************************************************************************/
void
ATO::Solver::ComputeVolume(double* p, double& v, double* dvdp)
/******************************************************************************/
{
  return _atoProblem->ComputeVolume(p, v, dvdp);
}

/******************************************************************************/
void
ATO::Solver::ComputeConstraint(double* p, double& c, double* dcdp)
/******************************************************************************/
{
}

/******************************************************************************/
int
ATO::Solver::GetNumOptDofs()
/******************************************************************************/
{
  if( _topoCentering == "Element" ){
    Albany::StateManager& stateMgr = _subProblem[0].app->getStateMgr();
    Albany::StateArrays& stateArrays = stateMgr.getStateArrays();
    Albany::StateArrayVec& dest = stateArrays.elemStateArrays;

    int numWorksets = dest.size();

    int numTotalElems = 0;
    for(int ws=0; ws<numWorksets; ws++){
      Albany::MDArray& wsTopo = dest[ws][_topoName];
      int wsSize = wsTopo.size();
      numTotalElems += wsSize;
    }
    return numTotalElems;
    
  } else
  if( _topoCentering == "Node" ){
    return _subProblem[0].app->getDiscretization()->getNodeMap()->NumMyElements();
  }
}

/******************************************************************************/
///*********************** SETUP AND UTILITY FUNCTIONS **********************///
/******************************************************************************/


/******************************************************************************/
ATO::SolverSubSolver
ATO::Solver::CreateSubSolver( const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                              const Epetra_Comm& comm,
                              const Teuchos::RCP<const Epetra_Vector>& initial_guess) const
/******************************************************************************/
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  ATO::SolverSubSolver ret; //value to return

  const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(comm);

  RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  *out << "ATO Solver creating solver from " << appParams->name()
       << " parameter list" << std::endl;

  //! Create solver factory, which reads xml input filen
  Albany::SolverFactory slvrfctry(appParams, mpiComm);

  //! Create solver and application objects via solver factory
  RCP<Epetra_Comm> appComm = Albany::createEpetraCommFromMpiComm(mpiComm);
  ret.model = slvrfctry.createAndGetAlbanyApp(ret.app, appComm, appComm, initial_guess);


  ret.params_in = rcp(new EpetraExt::ModelEvaluator::InArgs);
  ret.responses_out = rcp(new EpetraExt::ModelEvaluator::OutArgs);

  *(ret.params_in) = ret.model->createInArgs();
  *(ret.responses_out) = ret.model->createOutArgs();
  int ss_num_p = ret.params_in->Np();     // Number of *vectors* of parameters
  int ss_num_g = ret.responses_out->Ng(); // Number of *vectors* of responses
  RCP<Epetra_Vector> p1;
  RCP<Epetra_Vector> g1;

  if (ss_num_p > 0)
    p1 = rcp(new Epetra_Vector(*(ret.model->get_p_init(0))));
  if (ss_num_g > 1)
    g1 = rcp(new Epetra_Vector(*(ret.model->get_g_map(0))));
  RCP<Epetra_Vector> xfinal =
    rcp(new Epetra_Vector(*(ret.model->get_g_map(ss_num_g-1)),true) );

  // Sensitivity Analysis stuff
  bool supportsSensitivities = false;
  RCP<Epetra_MultiVector> dgdp;

  if (ss_num_p>0 && ss_num_g>1) {
    supportsSensitivities =
      !ret.responses_out->supports(EpetraExt::ModelEvaluator::OUT_ARG_DgDp, 0, 0).none();

    if (supportsSensitivities) {
      if (p1->GlobalLength() > 0)
        dgdp = rcp(new Epetra_MultiVector(g1->Map(), p1->GlobalLength() ));
      else
        supportsSensitivities = false;
    }
  }

  if (ss_num_p > 0)  ret.params_in->set_p(0,p1);
  if (ss_num_g > 1)  ret.responses_out->set_g(0,g1);
  ret.responses_out->set_g(ss_num_g-1,xfinal);

  if (supportsSensitivities) ret.responses_out->set_DgDp(0,0,dgdp);

  return ret;
}

/******************************************************************************/
Teuchos::RCP<Teuchos::ParameterList> 
ATO::Solver::createInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams, int physIndex) const
/******************************************************************************/
{   


  ///*** CREATE INPUT FILE FOR SUBPROBLEM: ***///
  

  // Get physics (pde) problem sublist, i.e., Physics Problem N, where N = physIndex.
  std::stringstream physStream;
  physStream << "Physics Problem " << physIndex;
  Teuchos::ParameterList& physics_subList = appParams->sublist("Problem").sublist(physStream.str(), false);

  // Create input parameter list for physics app which mimics a separate input file
  std::stringstream appStream;
  appStream << "Parameters for Subapplication " << physIndex;
  Teuchos::RCP<Teuchos::ParameterList> physics_appParams = Teuchos::createParameterList(appStream.str());

  // get reference to Problem ParameterList in new input file and initialize it 
  // from Parameters in Physics Problem N.
  Teuchos::ParameterList& physics_probParams = physics_appParams->sublist("Problem",false);
  physics_probParams.setParameters(physics_subList);

  // Add topology information
  Teuchos::ParameterList& topoParams = 
    appParams->sublist("Problem").get<Teuchos::ParameterList>("Topology");
  physics_probParams.set<Teuchos::ParameterList>("Topology",topoParams);

  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& physics_discList = physics_appParams->sublist("Discretization", false);
  physics_discList.setParameters(discList);

  // Piro sublist processing
  physics_appParams->set("Piro",appParams->sublist("Piro"));



  ///*** VERIFY SUBPROBLEM: ***///


  // extract physics and dimension of the subproblem
  Teuchos::ParameterList& subProblemParams = appParams->sublist("Problem").sublist(physStream.str());
  std::string problemName = subProblemParams.get<std::string>("Name");
  // "xD" where x = 1, 2, or 3
  std::string problemDimStr = problemName.substr( problemName.length()-2 );
  //remove " xD" where x = 1, 2, or 3
  std::string problemNameBase = problemName.substr( 0, problemName.length()-3 );
  
  //// check dimensions
  int numDimensions = 0;
  if(problemDimStr == "1D") numDimensions = 1;
  else if(problemDimStr == "2D") numDimensions = 2;
  else if(problemDimStr == "3D") numDimensions = 3;
  else TEUCHOS_TEST_FOR_EXCEPTION (
         true, Teuchos::Exceptions::InvalidParameter, std::endl 
         << "Error!  Cannot extract dimension from problem name: " << problemName << std::endl);
  TEUCHOS_TEST_FOR_EXCEPTION (
    numDimensions == 1, Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  Topology optimization is not avaliable in 1D." << std::endl);

  //// See if requested physics work with ATO (add your physics here)
  std::vector<std::string> ATOablePhysics;
  ATOablePhysics.push_back( "LinearElasticity" );
  
  std::vector<std::string>::iterator it;
  it = std::find(ATOablePhysics.begin(), ATOablePhysics.end(), problemNameBase);
  TEUCHOS_TEST_FOR_EXCEPTION (
    it == ATOablePhysics.end(), Teuchos::Exceptions::InvalidParameter, std::endl 
    << "Error!  Invalid problem base name: " << problemNameBase << std::endl);
  
  
  return physics_appParams;

}

/******************************************************************************/
Teuchos::RCP<const Teuchos::ParameterList>
ATO::Solver::getValidProblemParameters() const
/******************************************************************************/
{

  Teuchos::RCP<Teuchos::ParameterList> validPL = 
    Teuchos::createParameterList("ValidTopologicalOptimizationProblemParams");

  // Basic set-up
  validPL->set<int>("Number of Subproblems", 1, "Number of PDE constraint problems");
  validPL->set<bool>("Verbose Output", false, "Enable detailed output mode");
  validPL->set<std::string>("Name", "", "String to designate Problem");

  // Specify physics problem(s)
  for(int i=0; i<_numPhysics; i++){
    std::stringstream physStream; physStream << "Physics Problem " << i;
    validPL->sublist(physStream.str(), false, "");
  }

  // Specify aggregator
  validPL->sublist("Objective Aggregator", false, "");

  // Specify optimizer
  validPL->sublist("Topological Optimization", false, "");

  // Specify responses
  validPL->sublist("Topology", false, "");

  // Physics solver options
  validPL->set<std::string>(
       "Piro Defaults Filename", "", 
       "An xml file containing a default Piro parameterlist and its sublists");

  // Candidate for deprecation.
  validPL->set<std::string>(
       "Solution Method", "Steady", 
       "Flag for Steady, Transient, or Continuation");

  return validPL;
}





/******************************************************************************/
///*************                   BOILERPLATE                  *************///
/******************************************************************************/



/******************************************************************************/
ATO::Solver::~Solver() { }
/******************************************************************************/


/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_x_map() const
/******************************************************************************/
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_f_map() const
/******************************************************************************/
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

/******************************************************************************/
EpetraExt::ModelEvaluator::InArgs 
ATO::Solver::createInArgs() const
/******************************************************************************/
{
  EpetraExt::ModelEvaluator::InArgsSetup inArgs;
  inArgs.setModelEvalDescription("ATO Solver Model Evaluator Description");
  inArgs.set_Np(_num_parameters);
  return inArgs;
}

/******************************************************************************/
EpetraExt::ModelEvaluator::OutArgs 
ATO::Solver::createOutArgs() const
/******************************************************************************/
{
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("ATO Solver Multipurpose Model Evaluator");
  outArgs.set_Np_Ng(_num_parameters, _num_responses+1);  //TODO: is the +1 necessary still??
  return outArgs;
}

/******************************************************************************/
Teuchos::RCP<const Epetra_Map> ATO::Solver::get_g_map(int j) const
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(j > _num_responses || j < 0, Teuchos::Exceptions::InvalidParameter,
                     std::endl <<
                     "Error in ATO::Solver::get_g_map():  " <<
                     "Invalid response index j = " <<
                     j << std::endl);
  //TEV: Hardwired for now
  int _num_responses = 0;
  if      (j <  _num_responses) return _epetra_response_map;  //no index because num_g == 1 so j must be zero
  else if (j == _num_responses) return _epetra_x_map;
  return Teuchos::null;
}

/******************************************************************************/
ATO::SolverSubSolverData
ATO::Solver::CreateSubSolverData(const ATO::SolverSubSolver& sub) const
/******************************************************************************/
{
  ATO::SolverSubSolverData ret;
  if( sub.params_in->Np() > 0 && sub.responses_out->Ng() > 0 ) {
    ret.deriv_support = sub.model->createOutArgs().supports(OUT_ARG_DgDp, 0, 0);
  }
  else ret.deriv_support = EpetraExt::ModelEvaluator::DerivativeSupport();

  ret.Np = sub.params_in->Np();
  ret.pLength = std::vector<int>(ret.Np);
  for(int i=0; i<ret.Np; i++) {
    Teuchos::RCP<const Epetra_Vector> solver_p = sub.params_in->get_p(i);
    if(solver_p != Teuchos::null) ret.pLength[i] = solver_p->MyLength();  //uses local length (need to modify to work with distributed params)
    else ret.pLength[i] = 0;
  }

  ret.Ng = sub.responses_out->Ng();
  ret.gLength = std::vector<int>(ret.Ng);
  for(int i=0; i<ret.Ng; i++) {
    Teuchos::RCP<const Epetra_Vector> solver_g = sub.responses_out->get_g(i);
    if(solver_g != Teuchos::null) ret.gLength[i] = solver_g->MyLength(); //uses local length (need to modify to work with distributed responses)
    else ret.gLength[i] = 0;
  }

  if(ret.Np > 0) {
    Teuchos::RCP<const Epetra_Vector> p_init =
      sub.model->get_p_init(0); //only first p vector used - in the future could make ret.p_init an array of Np vectors
    if(p_init != Teuchos::null) ret.p_init = Teuchos::rcp(new const Epetra_Vector(*p_init)); //copy
    else ret.p_init = Teuchos::null;
  }
  else ret.p_init = Teuchos::null;

  return ret;
}

