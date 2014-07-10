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
  class SolverSubSolver;
  class SolverSubSolverData;

  class Solver : public EpetraExt::ModelEvaluator {
  public:

      Solver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
	     const Teuchos::RCP<const Epetra_Comm>& comm,
	     const Teuchos::RCP<const Epetra_Vector>& initial_guess);

    ~Solver();

    virtual Teuchos::RCP<const Epetra_Map> get_x_map() const;         //pure virtual from EpetraExt::ModelEvaluator
    virtual Teuchos::RCP<const Epetra_Map> get_f_map() const;         //pure virtual from EpetraExt::ModelEvaluator
    virtual EpetraExt::ModelEvaluator::InArgs createInArgs() const;   //pure virtual from EpetraExt::ModelEvaluator
    virtual EpetraExt::ModelEvaluator::OutArgs createOutArgs() const; //pure virtual from EpetraExt::ModelEvaluator
    void evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const; //pure virtual from EpetraExt::ModelEvaluator

  private:
    // data
    int  numDims;
    int _num_parameters; // for sensitiviy analysis(?)
    int _num_responses;  //  ditto
    Teuchos::RCP<Epetra_LocalMap> _epetra_param_map;
    Teuchos::RCP<Epetra_LocalMap> _epetra_response_map;
    Teuchos::RCP<Epetra_Map>      _epetra_x_map;

    bool _is_verbose;

    std::string defaultSubSolver;
    std::string problemNameBase;
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList> > subProblemAppParams;

    Teuchos::RCP<const Epetra_Comm> _solverComm;
    Teuchos::RCP<Teuchos::ParameterList> _mainAppParams;

    // methods
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;

    SolverSubSolver CreateSubSolver(const Teuchos::RCP<Teuchos::ParameterList> appParams, const Epetra_Comm& comm,
				    const Teuchos::RCP<const Epetra_Vector>& initial_guess  = Teuchos::null) const;

    SolverSubSolverData CreateSubSolverData(const ATO::SolverSubSolver& sub) const;

  };

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
    int Ng;
    std::vector<int> pLength;
    std::vector<int> gLength;
    Teuchos::RCP<const Epetra_Vector> p_init;
    EpetraExt::ModelEvaluator::DerivativeSupport deriv_support;
  };

}
#endif
