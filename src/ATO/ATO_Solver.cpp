//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Solver.hpp"
/*TEV
#include "QCAD_CoupledPoissonSchrodinger.hpp"
#include "Piro_Epetra_LOCASolver.hpp"
TEV*/

/* GAH FIXME - Silence warning:
TRILINOS_DIR/../../../include/pecos_global_defs.hpp:17:0: warning: 
        "BOOST_MATH_PROMOTE_DOUBLE_POLICY" redefined [enabled by default]
Please remove when issue is resolved
*/
#undef BOOST_MATH_PROMOTE_DOUBLE_POLICY

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

/*TEV
#include "Stokhos.hpp"
#include "Stokhos_Epetra.hpp"
#include "Sacado_PCE_OrthogPoly.hpp"


#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_EigendataInfoStruct.hpp"
TEV*/

/*TEV
#ifdef ALBANY_CI
#include "AnasaziConfigDefs.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziBlockDavidsonSolMgr.hpp"
#include "AnasaziBasicOutputManager.hpp"
#endif
TEV*/



/*TEV
namespace ATO {
  
  void SolveModel(const SolverSubSolver& ss);
  void SolveModel(const SolverSubSolver& ss, 
		  Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates);
  void SolveModel(const QCAD::SolverSubSolver& ss, 
		  Teuchos::RCP<Albany::EigendataStruct>& pInitialEData, 
		  Teuchos::RCP<Albany::EigendataStruct>& pFinalEData);
  void SolveModel(const SolverSubSolver& ss, 
		  Albany::StateArrays*& pInitialStates, Albany::StateArrays*& pFinalStates,
		  Teuchos::RCP<Albany::EigendataStruct>& pInitialEData,
		  Teuchos::RCP<Albany::EigendataStruct>& pFinalEData);



  void CopyStateToContainer(Albany::StateArrays& src,
			    std::string stateNameToCopy,
			    std::vector<Intrepid::FieldContainer<RealType> >& dest);
  void CopyContainerToState(std::vector<Intrepid::FieldContainer<RealType> >& src,
			    Albany::StateArrays& dest,
			    std::string stateNameOfCopy);
  void CopyContainer(std::vector<Intrepid::FieldContainer<RealType> >& src,
		     std::vector<Intrepid::FieldContainer<RealType> >& dest);
  void AddContainerToContainer(std::vector<Intrepid::FieldContainer<RealType> >& src,
			       std::vector<Intrepid::FieldContainer<RealType> >& dest,
			       double srcFactor, double thisFactor); // dest = thisFactor * dest + srcFactor * src
  void AddContainerToState(std::vector<Intrepid::FieldContainer<RealType> >& src,
			    Albany::StateArrays& dest,
			   std::string stateName, double srcFactor, double thisFactor); // dest[stateName] = thisFactor * dest[stateName] + srcFactor * src

  
  void CopyState(Albany::StateArrays& src, Albany::StateArrays& dest,  std::string stateNameToCopy);
  void AddStateToState(Albany::StateArrays& src, std::string srcStateNameToAdd, 
		       Albany::StateArrays& dest, std::string destStateNameToAddTo);
  void SubtractStateFromState(Albany::StateArrays& src, std::string srcStateNameToSubtract,
			      Albany::StateArrays& dest, std::string destStateNameToSubtractFrom);
  
  double getMaxDifference(Albany::StateArrays& states, 
			  std::vector<Intrepid::FieldContainer<RealType> >& prevState,
			  std::string stateName);

  double getNorm2Difference(Albany::StateArrays& states,   
			    std::vector<Intrepid::FieldContainer<RealType> >& prevState,
			    std::string stateName);
  double getNorm2(std::vector<Intrepid::FieldContainer<RealType> >& container, const Teuchos::RCP<const Epetra_Comm>& comm);
  int getElementCount(std::vector<Intrepid::FieldContainer<RealType> >& container);
  
  void ResetEigensolverShift(const Teuchos::RCP<EpetraExt::ModelEvaluator>& Solver, double newShift,
			     Teuchos::RCP<Teuchos::ParameterList>& eigList);
  double GetEigensolverShift(const SolverSubSolver& ss, double pcBelowMinPotential);
  void   SetPreviousDensityMixing(const Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> inArgs, double mixingFactor);


  //String processing helper functions
  std::vector<std::string> string_split(const std::string& s, char delim, bool bProtect=false);
  std::string string_remove_whitespace(const std::string& s);
  std::vector<std::string> string_parse_function(const std::string& s);
  std::map<std::string,std::string> string_parse_arrayref(const std::string& s);
  std::vector<int> string_expand_compoundindex(const std::string& indexStr, int min_index, int max_index);

}
TEV*/



ATO::Solver::
Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
       const Teuchos::RCP<const Epetra_Comm>& comm,
       const Teuchos::RCP<const Epetra_Vector>& initial_guess)
: _solverComm(comm), _mainAppParams(appParams)
{
  using std::string;

  // Get sub-problem input xml files from problem parameters
  Teuchos::ParameterList& problemParams = appParams->sublist("Problem");

  // Validate Problem parameters against list for this specific problem
  problemParams.validateParameters(*getValidProblemParameters(),0);

  string problemName = problemParams.get<string>("Name");
  string problemNameBase = problemName.substr( 0, problemName.length()-3 ); //remove " xD" where x = 1, 2, or 3
  string problemDimStr = problemName.substr( problemName.length()-2 ); // "xD" where x = 1, 2, or 3
  
  if(problemDimStr == "1D") numDims = 1;
  else if(problemDimStr == "2D") numDims = 2;
  else if(problemDimStr == "3D") numDims = 3;
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				   << "Error!  Cannot extract dimension from problem name: "
				   << problemName << std::endl);

  if( !(problemNameBase == "Poisson" || problemNameBase == "Schrodinger" || problemNameBase == "Schrodinger CI" ||
	problemNameBase == "Poisson Schrodinger" || problemNameBase == "Poisson Schrodinger CI"))
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				<< "Error!  Invalid problem base name: "
				<< problemNameBase << std::endl);
  

  // Check if "verbose" mode is enabled
  // bVerbose = (comm->MyPID() == 0) && problemParams.get<bool>("Verbose Output", false);

/*TEV
  eigensolverName = problemParams.get<string>("Schrodinger Eigensolver","LOBPCG");
  bRealEvecs = problemParams.get<bool>("Eigenvectors are Real",false);

  // Get problem parameters used for iterating Poisson-Schrodinger loop
  if(problemNameBase == "Poisson Schrodinger" || problemNameBase == "Poisson Schrodinger CI") {
    bUseIntegratedPS = problemParams.get<bool>("Use Integrated Poisson Schrodinger",true);
    maxIter = problemParams.get<int>("Maximum PS Iterations", 100);
    shiftPercentBelowMin = problemParams.get<double>("Eigensolver Percent Shift Below Potential Min", 1.0);
    ps_converge_tol = problemParams.get<double>("Iterative PS Convergence Tolerance", 1e-6);
    fixedPSOcc = problemParams.get<double>("Iterative PS Fixed Occupation", -1.0);
  }

  // Get problem parameters used for Poisson-Schrodinger-CI mode
  if(problemNameBase == "Poisson Schrodinger CI") {
    minCIParticles = problemParams.get<int>("Minimum CI Particles",0);
    maxCIParticles = problemParams.get<int>("Maximum CI Particles",10);
    bUseTotalSpinSymmetry = problemParams.get<bool>("Use S2 Symmetry in CI", false);
  }

  // Get problem parameters used for Schrodinger-CI mode
  if(problemNameBase == "Schrodinger CI") {
    nCIParticles = problemParams.get<int>("CI Particles");
    nCIExcitations = problemParams.get<int>("CI Excitations");
    assert(nCIParticles >= nCIExcitations);
  }

  // Get the number of eigenvectors - needed for all problems-modes except "Poisson"
  nEigenvectors = 0;
  if(problemNameBase != "Poisson") {
    nEigenvectors = problemParams.get<int>("Number of Eigenvalues");
  }

  // Get and set the default Piro parameters from a file, if given
  std::string piroFilename  = problemParams.get<std::string>("Piro Defaults Filename", "");
  if(piroFilename.length() > 0) {
    const Albany_MPI_Comm mpiComm = Albany::getMpiCommFromEpetraComm(*comm);
    Teuchos::RCP<Teuchos::Comm<int> > tcomm = Albany::createTeuchosCommFromMpiComm(mpiComm);
    Teuchos::RCP<Teuchos::ParameterList> defaultPiroParams = Teuchos::createParameterList("Default Piro Parameters");
    Teuchos::updateParametersFromXmlFileAndBroadcast(piroFilename, defaultPiroParams.ptr(), *tcomm);
    Teuchos::ParameterList& piroList = appParams->sublist("Piro", false);
    piroList.setParametersNotAlreadySet(*defaultPiroParams);
  }
    

  // Get name of output exodus file specified in Discretization section
  std::string outputExo = appParams->sublist("Discretization").get<std::string>("Exodus Output File Name");

  // Create Solver parameter lists based on problem name
  if( problemNameBase == "Poisson" ) {
    subProblemAppParams["Poisson"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "none",
							    debug_poissonXML, outputExo);
    defaultSubSolver = "Poisson";
  }

  else if( problemNameBase == "Schrodinger" ) {
    subProblemAppParams["Schrodinger"] = createSchrodingerInputFile(appParams, numDims, nEigenvectors, "none",
								    debug_schroXML, debug_schroExo);
    defaultSubSolver = "Schrodinger";
  }

  else if( problemNameBase == "Poisson Schrodinger" ) {
    subProblemAppParams["InitPoisson"] = createPoissonInputFile(appParams, numDims, nEigenvectors, "initial poisson",
								debug_initpoissonXML, debug_initpoissonExo);
    subProblemAppParams["Poisson"]     = createPoissonInputFile(appParams, numDims, nEigenvectors, "couple to schrodinger",
								debug_poissonXML, debug_poissonExo);
    subProblemAppParams["Schrodinger"] = createSchrodingerInputFile(appParams, numDims, nEigenvectors, "couple to poisson",
								    debug_schroXML, debug_schroExo);
    if(bUseIntegratedPS) {
      subProblemAppParams["PoissonSchrodinger"] = createPoissonSchrodingerInputFile(appParams, numDims, nEigenvectors,
										    debug_psXML, outputExo);
      defaultSubSolver = "PoissonSchrodinger";
    }
    else defaultSubSolver = "Poisson";    
  }

  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in ATO::Solver constructor:  " <<
				  "Invalid problem name base: " << problemNameBase << std::endl);

  //Save the initial guess passed to the solver
  saved_initial_guess = initial_guess;

  //Temporarily create a map of sub-solvers, used for obtaining the initial parameter vector and 
  //   for figuring out what types of derivatives are supported.
  std::map<std::string, SolverSubSolverData> subSolversData;
  std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> >::const_iterator itp;
  for(itp = subProblemAppParams.begin(); itp != subProblemAppParams.end(); ++itp) {
    const std::string& name = itp->first;
    const Teuchos::RCP<Teuchos::ParameterList>& param_list = itp->second;
    const SolverSubSolver& sub = CreateSubSolver( param_list , *comm);

    subSolversData[ name ] = CreateSubSolverData( sub );

    // Create Epetra map for solution vector (second response vector).  Assume 
    //  each subSolver has the same map, so just get the first one.
    if(itp == subProblemAppParams.begin()) {
      Teuchos::RCP<const Epetra_Map> sub_x_map = sub.app->getMap();
      TEUCHOS_TEST_FOR_EXCEPT( sub_x_map == Teuchos::null );
      epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));
    }    
  }

  //Determine whether we should support DgDp (all sub-solvers must support DpDg for QCAD::Solver to)
  bSupportDpDg = true;  
  std::map<std::string, SolverSubSolverData>::const_iterator it;
  for(it = subSolversData.begin(); it != subSolversData.end(); ++it) {
    deriv_support = (it->second).deriv_support;
    if(deriv_support.none()) { bSupportDpDg = false; break; } //test if p=0, g=0 DgDp is supported
  }

  // We support all dg/dp layouts model supports, plus the linear op layout
  if(bSupportDpDg) deriv_support.plus(DERIV_LINEAR_OP);

  //Setup Parameter and responses maps
  
  // input file can have 
  //    <Parameter name="Parameter 0" type="string" value="Poisson[0]" />
  //    <Parameter name="Parameter 1" type="string" value="Poisson[1:3]" />
  //
  //    <Parameter name="Response 0" type="string" value="Poisson[0] # charge" />
  //    <Parameter name="Response 0" type="string" value="Schrodinger[1,3]" />
  //    <Parameter name="Response 0" type="string" value="=dist(Poisson[1:4],Poisson[4:7]) # distance example" />

  
  Teuchos::ParameterList& paramList = problemParams.sublist("Parameters");
  setupParameterMapping(paramList, defaultSubSolver, subSolversData);

  Teuchos::ParameterList& responseList = problemParams.sublist("Response Functions");
  setupResponseMapping(responseList, defaultSubSolver, nEigenvectors, subSolversData);

  num_p = (nParameters > 0) ? 1 : 0; // Only use first parameter (p) vector, if there are any parameters
  num_g = (responseFns.size() > 0) ? 1 : 0; // Only use first response vector (but really one more than num_g -- 2nd holds solution vector)


#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif

  // Create Epetra map for parameter vector (only one since num_p always == 1)
  epetra_param_map = Teuchos::rcp(new Epetra_LocalMap(static_cast<GlobalIndex>(nParameters), 0, *comm));

  // Create Epetra map for (first) response vector
  epetra_response_map = Teuchos::rcp(new Epetra_LocalMap(static_cast<GlobalIndex>(nResponseDoubles), 0, *comm));
     //ANDY: if (nResponseDoubles > 0) needed ??

  // Get vector of initial parameter values
  epetra_param_vec = Teuchos::rcp(new Epetra_Vector(*(epetra_param_map)));

  // Take initial value from the first (if multiple) parameter 
  //    fns for each given parameter
  for(std::size_t i=0; i<nParameters; i++) {
    (*epetra_param_vec)[i] = paramFnVecs[i][0]->getInitialParam(subSolversData);
  }
TEV*/
}

ATO::Solver::~Solver()
{
}

Teuchos::RCP<const Epetra_Map> ATO::Solver::get_x_map() const
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

Teuchos::RCP<const Epetra_Map> ATO::Solver::get_f_map() const
{
  Teuchos::RCP<const Epetra_Map> dummy;
  return dummy;
}

EpetraExt::ModelEvaluator::InArgs 
ATO::Solver::createInArgs() const
{
  EpetraExt::ModelEvaluator::InArgsSetup inArgs;
  inArgs.setModelEvalDescription("ATO Solver Model Evaluator Description");
/*TEV
  inArgs.set_Np(_num_p);
  return inArgs;
TEV*/
}

EpetraExt::ModelEvaluator::OutArgs 
ATO::Solver::createOutArgs() const
{
  //Based on Piro_Epetra_NOXSolver.cpp implementation
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("ATO Solver Multipurpose Model Evaluator");
  // Ng is 1 bigger then model-Ng so that the solution vector can be an outarg
  //TEV outArgs.set_Np_Ng(_num_p, num_g+1);  //TODO: is the +1 necessary still??
  //Derivative info 
  //TEVif(bSupportDpDg) {
  //TEV  for (int i=0; i<num_g; i++) {
  //TEV    for (int j=0; j<num_p; j++)
  //TEV      outArgs.setSupports(OUT_ARG_DgDp, i, j, deriv_support);
  //TEV  }
  //TEV}
  return outArgs;
}

Teuchos::RCP<const Teuchos::ParameterList>
ATO::Solver::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList("ValidTopologicalOptimizationProblemParams");

  validPL->set<std::string>("Name", "", "String to designate Problem");

  // Candidates for deprecation. Pertain to the solution rather than the problem definition.
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");

  return validPL;
}


void 
ATO::Solver::evalModel(const InArgs& inArgs,
                       const OutArgs& outArgs ) const
{
  std::map<std::string, SolverSubSolver> subSolvers;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

/*TEV
  if(_is_verbose) {
    if(_num_p > 0) {   // or could use: (inArgs.Np() > 0)
      Teuchos::RCP<const Epetra_Vector> p = inArgs.get_p(0); //only use *first* param vector
      *out << "BEGIN ATO Solver Parameters:" << std::endl;
      for(std::size_t i=0; i<nParameters; i++)
	*out << "  Parameter " << i << " = " << (*p)[i] << std::endl;
      *out << "END ATO Solver Parameters" << std::endl;
    }
  }
   
  if( problemNameBase == "Poisson" ) {
      if(_is_verbose) *out << "QCAD Solve: Simple Poisson solve" << std::endl;

      //Create Poisson solver & fill its parameters
      subSolvers[ "Poisson" ] = CreateSubSolver( getSubSolverParams("Poisson") , *solverComm, saved_initial_guess);
      fillSingleSubSolverParams(inArgs, "Poisson", subSolvers[ "Poisson" ]);

      QCAD::SolveModel(subSolvers["Poisson"]);
      eigenvalueResponses.resize(0); // no eigenvalues in the Poisson problem
  }

  else if( problemNameBase == "Schrodinger" ) {
      if(_is_verbose) *out << "QCAD Solve: Simple Schrodinger solve" << std::endl;

      //Create Schrodinger solver & fill its parameters
      subSolvers[ "Schrodinger" ] = CreateSubSolver( getSubSolverParams("Schrodinger") , *solverComm); // no initial guess
      fillSingleSubSolverParams(inArgs, "Schrodinger", subSolvers[ "Schrodinger" ]);

      Teuchos::RCP<Albany::EigendataStruct> eigenData = Teuchos::null;
      Teuchos::RCP<Albany::EigendataStruct> eigenDataNull = Teuchos::null;

      QCAD::SolveModel(subSolvers["Schrodinger"], eigenDataNull, eigenData);
      eigenvalueResponses = *(eigenData->eigenvalueRe); // copy eigenvalues to member variable
      for(std::size_t i=0; i<eigenvalueResponses.size(); ++i) eigenvalueResponses[i] *= -1; //apply minus sign (b/c of eigenval convention)

      // Create final observer to output evecs and solution
      Teuchos::RCP<Epetra_Vector> solnVec = subSolvers["Schrodinger"].responses_out->get_g(1); //get the *first* response vector (solution)
      Teuchos::RCP<MultiSolution_Observer> final_obs = 
	Teuchos::rcp(new QCAD::MultiSolution_Observer(subSolvers["Schrodinger"].app, mainAppParams)); 
      final_obs->observeSolution(*solnVec, "ZeroSolution", eigenData, 0.0);
  }

  else if( problemNameBase == "Poisson Schrodinger" )
    evalPoissonSchrodingerModel(inArgs, outArgs, eigenvalueResponses, subSolvers);

  else if( problemNameBase == "Schrodinger CI" )
    evalCIModel(inArgs, outArgs, eigenvalueResponses, subSolvers);

  else if( problemNameBase == "Poisson Schrodinger CI" )
    evalPoissonCIModel(inArgs, outArgs, eigenvalueResponses, subSolvers);

  if(num_g > 0) {
    // update main solver's responses using sub-solver response values
    Teuchos::RCP<Epetra_Vector> g = outArgs.get_g(0); //only use *first* response vector
    Teuchos::RCP<Epetra_MultiVector> dgdp = Teuchos::null;
    
    if(num_p > 0 && !outArgs.supports(OUT_ARG_DgDp, 0, 0).none()) 
      dgdp = outArgs.get_DgDp(0,0).getMultiVector();
    
    int offset = 0;
    std::vector<Teuchos::RCP<QCAD::SolverResponseFn> >::const_iterator rit;
    
    for(rit = responseFns.begin(); rit != responseFns.end(); rit++) {
      (*rit)->fillSolverResponses( *g, dgdp, offset, subSolvers, paramFnVecs, bSupportDpDg, eigenvalueResponses);
      offset += (*rit)->getNumDoubles();
    }
    
    if(bVerbose) {
      *out << "BEGIN QCAD Solver Responses:" << std::endl;
      for(int i=0; i< g->MyLength(); i++)
	*out << "  Response " << i << " = " << (*g)[i] << std::endl;
      *out << "END QCAD Solver Responses" << std::endl;
      
    }
  }
TEV*/
}
