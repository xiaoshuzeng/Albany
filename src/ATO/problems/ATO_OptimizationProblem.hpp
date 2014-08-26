//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_OPTIMIZATION_PROBLEM_HPP
#define ATO_OPTIMIZATION_PROBLEM_HPP

#include "Albany_AbstractProblem.hpp"

namespace ATO {

class OptimizationProblem :
public virtual Albany::AbstractProblem {

  public:
   OptimizationProblem( const Teuchos::RCP<Teuchos::ParameterList>& _params,
                        const Teuchos::RCP<ParamLib>& _paramLib,
                        const int _numDim);


   void ComputeVolume(double* p, double& v, double* dvdp=NULL);
   void ComputeVolume(double& v);
   void setDiscretization(Teuchos::RCP<Albany::AbstractDiscretization> _disc)
          {disc = _disc;}

   void InitTopOpt();

  protected:
   void setupTopOpt( Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  _meshSpecs,
                     Albany::StateManager& _stateMgr);

   Teuchos::RCP<Albany::AbstractDiscretization> disc;
   Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;
   Albany::StateManager* stateMgr;

   std::vector<Teuchos::RCP<shards::CellTopology> > cellTypes;
   std::vector<Teuchos::RCP<Intrepid::Cubature<double> > > cubatures;

   std::vector<Intrepid::FieldContainer<double> > refPoints;
   std::vector<Intrepid::FieldContainer<double> > refWeights;
   std::vector<Intrepid::FieldContainer<double> > weighted_measure;

};

}

#endif
