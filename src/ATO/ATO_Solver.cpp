//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_Solver.hpp"

/* GAH FIXME - Silence warning:
TRILINOS_DIR/../../../include/pecos_global_defs.hpp:17:0: warning: 
        "BOOST_MATH_PROMOTE_DOUBLE_POLICY" redefined [enabled by default]
Please remove when issue is resolved
*/
#undef BOOST_MATH_PROMOTE_DOUBLE_POLICY

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Albany_SolverFactory.hpp"

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
  string problemDimStr = problemName.substr( problemName.length()-2 ); // "xD" where x = 1, 2, or 3
  problemNameBase = problemName.substr( 0, problemName.length()-3 ); //remove " xD" where x = 1, 2, or 3
  
  if(problemDimStr == "1D") numDims = 1;
  else if(problemDimStr == "2D") numDims = 2;
  else if(problemDimStr == "3D") numDims = 3;
  else TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				   << "Error!  Cannot extract dimension from problem name: "
				   << problemName << std::endl);

  // set problem (pde constraint)
  if( !(problemNameBase == "Elasticity" ))
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
				<< "Error!  Invalid problem base name: "
				<< problemNameBase << std::endl);
  
  // set verbosity
  _is_verbose = (comm->MyPID() == 0) && problemParams.get<bool>("Verbose Output", false);
  
  // set parameters and responses
  _num_parameters = 0; //TEV: assume no parameters or responses for now...
  _num_responses  = 0; //TEV: assume no parameters or responses for now...

  // Get name of output exodus file specified in Discretization section
  std::string outputExo = appParams->sublist("Discretization").get<std::string>("Exodus Output File Name");

  // Create Solver parameter lists based on problem name
  if( problemNameBase == "Elasticity" ) {
    subProblemAppParams["Elasticity"] = createElasticityInputFile( appParams, 
                                                                   numDims, 
                                                                   "none",
                                                                   outputExo );
    defaultSubSolver = "Elasticity";
  }
  else TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in ATO::Solver constructor:  " <<
				  "Invalid problem name base: " << problemNameBase << std::endl);

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

  //Save the initial guess passed to the solver
//TEV  saved_initial_guess = initial_guess;

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
      _epetra_x_map = Teuchos::rcp(new Epetra_Map( *sub_x_map ));
    }    
  }

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  typedef int GlobalIndex;
#else
  typedef long long GlobalIndex;
#endif
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
  inArgs.set_Np(_num_parameters);
  return inArgs;
}

EpetraExt::ModelEvaluator::OutArgs 
ATO::Solver::createOutArgs() const
{
  EpetraExt::ModelEvaluator::OutArgsSetup outArgs;
  outArgs.setModelEvalDescription("ATO Solver Multipurpose Model Evaluator");
  outArgs.set_Np_Ng(_num_parameters, _num_responses+1);  //TODO: is the +1 necessary still??
  return outArgs;
}

Teuchos::RCP<const Teuchos::ParameterList>
ATO::Solver::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = Teuchos::createParameterList("ValidTopologicalOptimizationProblemParams");

  validPL->set<std::string>( "Name", "", "String to designate Problem");
  validPL->set<bool>( "Verbose Output", false, "Enable detailed output mode");
  validPL->set<std::string>( "Piro Defaults Filename", "", "An xml file containing a default Piro parameterlist and its sublists");

  validPL->sublist("Elasticity Problem", false, "");


  // Candidates for deprecation. Pertain to the solution rather than the problem definition.
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");

  return validPL;
}

Teuchos::RCP<const Epetra_Map> ATO::Solver::get_g_map(int j) const
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

void 
ATO::Solver::evalModel(const InArgs& inArgs,
                       const OutArgs& outArgs ) const
{
  std::map<std::string, SolverSubSolver> subSolvers;
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

/*TEV
  if( problemNameBase == "Elasticity" ) {
      if(_is_verbose) *out << "ATO Solve: Elasticity" << std::endl;

      //Create Poisson solver & fill its parameters
      subSolvers[ "Elasticity" ] = CreateSubSolver( getSubSolverParams("Elasticity"), 
                                                   *_solverComm, 
                                                    0x0);
      fillSingleSubSolverParams(inArgs, "Elasticity", subSolvers[ "Elasticity" ]);

      ATO::SolveModel(subSolvers["Elasticity"]);
  }
TEV*/
}

ATO::SolverSubSolver
ATO::Solver::CreateSubSolver( const Teuchos::RCP<Teuchos::ParameterList> appParams, 
                              const Epetra_Comm& comm,
                              const Teuchos::RCP<const Epetra_Vector>& initial_guess) const
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

ATO::SolverSubSolverData
ATO::Solver::CreateSubSolverData(const ATO::SolverSubSolver& sub) const
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

  
Teuchos::RCP<Teuchos::ParameterList> 
ATO::Solver::createElasticityInputFile( const Teuchos::RCP<Teuchos::ParameterList>& appParams,
                                        int numDims, 
                                        const std::string& specialProcessing,
                                        const std::string& exoOutputFile ) const
{   

  Teuchos::ParameterList& physics_problemParams = appParams->sublist("Problem");
    
  // Get physics (pde) problem sublists
  Teuchos::ParameterList& physics_subList = physics_problemParams.sublist("Elasticity Problem", false);
   
  // Create input parameter list for physics app which mimics a separate input file
  Teuchos::RCP<Teuchos::ParameterList> physics_appParams =
    Teuchos::createParameterList("Elasticity Subapplication Parameters - " + specialProcessing);
  Teuchos::ParameterList& physics_probParams = physics_appParams->sublist("Problem",false);


  physics_probParams.set("Solution Method", "ATO Elasticity");
  std::ostringstream name;
  name << "Elasticity " << numDims << "D";
  physics_probParams.set("Name", name.str());

/*TEV
  poisson_probParams.set("Name", QCAD::strdim("Poisson",numDims));
  poisson_probParams.set("Phalanx Graph Visualization Detail", vizDetail);
  poisson_probParams.set("Length Unit In Meters",lenUnit);
  poisson_probParams.set("Energy Unit In Electron Volts",energyUnit);
  poisson_probParams.set("MaterialDB Filename", matrlFile);
  if(Temp >= 0) poisson_probParams.set("Temperature",Temp);

  // Poisson Source sublist processing
  {
    Teuchos::ParameterList auto_sourceList;
    auto_sourceList.set("Factor",1.0);
    auto_sourceList.set("Device","elementblocks");

    if(specialProcessing == "initial poisson") {
      auto_sourceList.set("Quantum Region Source", "semiclassical");
      auto_sourceList.set("Non Quantum Region Source", "semiclassical");

    } else if (specialProcessing == "couple to schrodinger") {
      auto_sourceList.set("Quantum Region Source", "schrodinger");
      auto_sourceList.set("Non Quantum Region Source", bQBOnly ? "semiclassical" : "schrodinger");
      auto_sourceList.set("Eigenvectors to Import", nEigen);
      auto_sourceList.set("Eigenvectors are Real", bRealEvecs);
      auto_sourceList.set("Use predictor-corrector method", bUsePCMethod);
      auto_sourceList.set("Include exchange-correlation potential", bXCPot);
      auto_sourceList.set("Fixed Quantum Occupation", fixedPSOcc);

    } else if (specialProcessing == "Coulomb") {
      auto_sourceList.set("Quantum Region Source", "coulomb");
      auto_sourceList.set("Non Quantum Region Source", "none");
      auto_sourceList.set("Imaginary Part of Coulomb Source", false);
      auto_sourceList.set("Eigenvectors to Import", nEigen);
      auto_sourceList.set("Eigenvectors are Real", bRealEvecs);

    } else if (specialProcessing != "none")
      TEUCHOS_TEST_FOR_EXCEPTION( true, Teuchos::Exceptions::InvalidParameter,
                                  "Invalid special processing for Elasticity input: " << specialProcessing);

    Teuchos::ParameterList& sourceList = poisson_probParams.sublist("Poisson Source", false);
    if(poisson_subList.isSublist("Poisson Source"))
      sourceList.setParameters( poisson_subList.sublist("Poisson Source") );
    sourceList.setParametersNotAlreadySet( auto_sourceList );
  }

  // Permittivity sublist processing
  {
    Teuchos::ParameterList auto_permList;
    auto_permList.set("Permittivity Type","Block Dependent");

    Teuchos::ParameterList& permList = poisson_probParams.sublist("Permittivity", false);
    if(poisson_subList.isSublist("Permittivity"))
      permList.setParameters( poisson_subList.sublist("Permittivity") );
    permList.setParametersNotAlreadySet( auto_permList );
  }


  // Dirichlet BC sublist processing
  if(poisson_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& poisson_dbcList = poisson_probParams.sublist("Dirichlet BCs", false);
    poisson_dbcList.setParameters(poisson_subList.sublist("Dirichlet BCs"));
  }
  else if(schro_subList.isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList& poisson_dbcList = poisson_probParams.sublist("Dirichlet BCs", false);
    const Teuchos::ParameterList& schro_dbcList = schro_subList.sublist("Dirichlet BCs");
    Teuchos::ParameterList::ConstIterator it; double* dummy = NULL;
    for(it = schro_dbcList.begin(); it != schro_dbcList.end(); ++it) {
      std::string dbcName = schro_dbcList.name(it);
      std::size_t k = dbcName.find("psi");
      if( k != std::string::npos ) {
        dbcName.replace(k, 3, "Phi");  // replace Phi -> psi
        poisson_dbcList.set( dbcName, schro_dbcList.entry(it).getValue(dummy) ); //copy all schrodinger DBCs
      }
    }
  }

  // Parameters sublist processing
  if( specialProcessing == "Coulomb" || specialProcessing == "Coulomb imaginary" ) {
    //! Add source eigenvector indices as parameters
    Teuchos::ParameterList& paramList = poisson_probParams.sublist("Parameters", false);
    if(poisson_subList.isSublist("Parameters"))
      paramList.setParameters(poisson_subList.sublist("Parameters"));

    int nParams = paramList.get<int>("Number", 0);
    paramList.set("Number", nParams + 2); //assumes Source Eigenvector X are not already params
    paramList.set(Albany::strint("Parameter",nParams), "Source Eigenvector 1");
    paramList.set(Albany::strint("Parameter",nParams+1), "Source Eigenvector 2");
  }
  else if(specialProcessing == "couple to schrodinger" ) {
    Teuchos::ParameterList& paramList = poisson_probParams.sublist("Parameters", false);
    if(poisson_subList.isSublist("Parameters"))
      paramList.setParameters(poisson_subList.sublist("Parameters"));

    int nParams = paramList.get<int>("Number", 0);
    paramList.set("Number", nParams + 1);
    paramList.set(Albany::strint("Parameter",nParams), "Previous Quantum Density Mixing Factor");
  }
  else {
    Teuchos::ParameterList& poisson_paramsList = poisson_probParams.sublist("Parameters", false);
    if(poisson_subList.isSublist("Parameters"))
      poisson_paramsList.setParameters(poisson_subList.sublist("Parameters"));
    else poisson_paramsList.set("Number", 0);
  }

  // Initial Condition sublist processing: copy list from main problem if it exists
  if(problemParams.isSublist("Initial Condition"))
  {
    Teuchos::ParameterList& icList = poisson_probParams.sublist("Initial Condition", false);
    icList.setParameters( problemParams.sublist("Initial Condition") );
  }

  // Discretization sublist processing
  Teuchos::ParameterList& discList = appParams->sublist("Discretization");
  Teuchos::ParameterList& poisson_discList = poisson_appParams->sublist("Discretization", false);
  poisson_discList.setParameters(discList);
  if(exoOutputFile.length() > 0)
    poisson_discList.set("Exodus Output File Name",exoOutputFile);
  else poisson_discList.remove("Exodus Output File Name",false);
TEV*/

  return physics_appParams;

}


