//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_SOLVER_H
#define ATO_SOLVER_H

#include <iostream>

#include "LOCA.H"
#include "LOCA_Epetra.H"
#include "Epetra_Vector.h"
#include "Epetra_LocalMap.h"
#include "LOCA_Epetra_ModelEvaluatorInterface.H"
#include <NOX_Epetra_MultiVector.H>

#include "Albany_ModelEvaluator.hpp"
#include "Albany_Utils.hpp"
#include "Piro_Epetra_StokhosNOXObserver.hpp"

namespace ATO {
/*
  class SolverParamFn;
  class SolverResponseFn;
  class SolverSubSolver;
  class SolverSubSolverData;
*/

/** \brief Epetra-based Model Evaluator for ATO solver
 *
 */

  class Solver : public EpetraExt::ModelEvaluator {
  public:

    /** \name Constructors/initializers */
    //@{

      Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
	     const Teuchos::RCP<const Epetra_Comm>& comm,
	     const Teuchos::RCP<const Epetra_Vector>& initial_guess);
    //@}

    ~Solver();

    virtual Teuchos::RCP<const Epetra_Map> get_x_map() const;         //pure virtual from EpetraExt::ModelEvaluator
    virtual Teuchos::RCP<const Epetra_Map> get_f_map() const;         //pure virtual from EpetraExt::ModelEvaluator
    virtual EpetraExt::ModelEvaluator::InArgs createInArgs() const;   //pure virtual from EpetraExt::ModelEvaluator
    virtual EpetraExt::ModelEvaluator::OutArgs createOutArgs() const; //pure virtual from EpetraExt::ModelEvaluator
    void evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const; //pure virtual from EpetraExt::ModelEvaluator

  private:
    // data
    int numDims;
    int _num_p;

    bool _is_verbose;

    Teuchos::RCP<const Epetra_Comm> _solverComm;
    Teuchos::RCP<Teuchos::ParameterList> _mainAppParams;

    // methods
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

/*TEV
    Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;
    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    Teuchos::RCP<const Epetra_Vector> get_x_init() const;
    Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;

*/


  private:
/*
    Teuchos::RCP<Teuchos::ParameterList> createPoissonInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
								int numDims, int nEigen, const std::string& specialProcessing,
								const std::string& xmlOutputFile, const std::string& exoOutputFile) const;
    Teuchos::RCP<Teuchos::ParameterList> createSchrodingerInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
								    int numDims, int nEigen, const std::string& specialProcessing,
								    const std::string& xmlOutputFile, const std::string& exoOutputFile) const;
    Teuchos::RCP<Teuchos::ParameterList> createPoissonSchrodingerInputFile(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
									   int numDims, int nEigen, const std::string& xmlOutputFile,
									   const std::string& exoOutputFile) const;

    void evalPoissonSchrodingerModel(const InArgs& inArgs, const OutArgs& outArgs,
				     std::vector<double>& eigenvalResponses, std::map<std::string, SolverSubSolver>& subSolvers) const;
    void evalPoissonCIModel(const InArgs& inArgs, const OutArgs& outArgs,
			    std::vector<double>& eigenvalResponses, std::map<std::string, SolverSubSolver>& subSolvers) const;
    void evalCIModel(const InArgs& inArgs, const OutArgs& outArgs, 
		     std::vector<double>& eigenvalResponses, std::map<std::string, SolverSubSolver>& subSolvers) const;

    bool doPSLoop(const std::string& mode, const InArgs& inArgs, std::map<std::string, SolverSubSolver>& subSolvers, 
		  Teuchos::RCP<Albany::EigendataStruct>& eigenDataResult, bool bPrintNumOfQuantumElectrons) const;

    void setupParameterMapping(const Teuchos::ParameterList& list, const std::string& defaultSubSolver,
			       const std::map<std::string, SolverSubSolverData>& subSolversData);
    void setupResponseMapping(const Teuchos::ParameterList& list, const std::string& defaultSubSolver, int nEigenvalues,
			      const std::map<std::string, SolverSubSolverData>& subSolversData);

    void fillSingleSubSolverParams(const InArgs& inArgs, const std::string& name, 
				   QCAD::SolverSubSolver& subSolver, int nLeaveOffEnd=0) const;

    SolverSubSolver CreateSubSolver(const Teuchos::RCP<Teuchos::ParameterList> appParams, const Epetra_Comm& comm,
				    const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null) const;

    SolverSubSolverData CreateSubSolverData(const QCAD::SolverSubSolver& sub) const;


    const Teuchos::RCP<Teuchos::ParameterList>& getSubSolverParams(const std::string& name) const;
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void printResponses(const QCAD::SolverSubSolver& solver, 
			const std::string& solverName, 
			Teuchos::RCP<Teuchos::FancyOStream> out) const;
    
  private:
    int numDims;
    std::string problemNameBase;
    std::string defaultSubSolver;
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> > subProblemAppParams;

    std::vector< std::vector<Teuchos::RCP<SolverParamFn> > > paramFnVecs;
    std::vector<Teuchos::RCP<SolverResponseFn> > responseFns;

    std::size_t maxIter;
    std::size_t nParameters;
    std::size_t nResponseDoubles;    

    std::string iterationMethod;
    int  nEigenvectors; //used in Poisson-CI coupling

    int num_p, num_g;
    Teuchos::RCP<Epetra_LocalMap> epetra_param_map;
    Teuchos::RCP<Epetra_LocalMap> epetra_response_map;
    Teuchos::RCP<Epetra_Map> epetra_x_map;

    Teuchos::RCP<Epetra_Vector> epetra_param_vec;
    DerivativeSupport deriv_support;

    Teuchos::RCP<const Epetra_Vector> saved_initial_guess;

    bool bVerbose;
    bool bSupportDpDg;
    bool bRealEvecs;

    std::string eigensolverName;
    double ps_converge_tol;
    double shiftPercentBelowMin;  // for eigensolver shift-invert: shift point == minPotential * (1 + shiftPercent/100)
    int    minCIParticles;        // the minimum number of particles allowed to be used in CI calculation
    int    maxCIParticles;        // the maximum number of particles allowed to be used in CI calculation
    int    nCIParticles;          // the number of particles used in CI calculation
    int    nCIExcitations;        // the number of excitations used in CI calculation
    double fixedPSOcc;
    bool   bUseIntegratedPS;
    bool   bUseTotalSpinSymmetry; // use S2 symmetry in CI calculation
*/
  };


/*
  // helper classes - maybe nest inside Solver?
  class SolverParamFn {
  public:
    SolverParamFn(const std::string& fnString, 
		  const std::map<std::string, SolverSubSolverData>& subSolversData);
    ~SolverParamFn() {};

    void fillSingleSubSolverParams(double parameterValue, const std::string& subSolverName,
				   SolverSubSolver& subSolver) const;

    void fillSubSolverParams(double parameterValue, 
			     const std::map<std::string, SolverSubSolver>& subSolvers) const;

    double getInitialParam(const std::map<std::string, SolverSubSolverData>& subSolversData) const;

    std::string getTargetName() const { return targetName; }
    std::vector<int> getTargetIndices() const { return targetIndices; }
    std::size_t getNumFilters() const { return filters.size(); }
    double getFilterScaling() const;

  protected:
    std::string targetName;
    std::vector<int> targetIndices;
    std::vector< std::vector<std::string> > filters;
  };

  class SolverResponseFn {
  public:
    SolverResponseFn(const std::string& fnString,
		     const std::map<std::string, SolverSubSolverData>& subSolversData,
		     int nEigenvalues);
    ~SolverResponseFn() {};

    void fillSolverResponses(Epetra_Vector& g, Teuchos::RCP<Epetra_MultiVector>& dgdp, int offset,
			     const std::map<std::string, SolverSubSolver>& subSolvers,
			     const std::vector<std::vector<Teuchos::RCP<SolverParamFn> > >& paramFnVecs,
			     bool bSupportDpDg, const std::vector<double>& eigenvalueResponses) const;

    std::size_t getNumDoubles() const { return numDoubles; }

  protected:
    struct ArrayRef { 
      std::string name; 
      std::vector<int> indices;
    };

    std::string fnName;
    std::vector<ArrayRef> params;
    std::size_t numDoubles; //number of doubles produced by this response
  };


*/
  class SolverSubSolver {
  public:
    Teuchos::RCP<Albany::Application> app;
    Teuchos::RCP<EpetraExt::ModelEvaluator> model;
    Teuchos::RCP<EpetraExt::ModelEvaluator::InArgs> params_in;
    Teuchos::RCP<EpetraExt::ModelEvaluator::OutArgs> responses_out;
    void freeUp() { app = Teuchos::null; model = Teuchos::null; }
  };

  class SolverSubSolverData {
  public:
    int Np;
    std::vector<int> pLength;
    Teuchos::RCP<const Epetra_Vector> p_init;
    EpetraExt::ModelEvaluator::DerivativeSupport deriv_support;
  };
}
#endif
