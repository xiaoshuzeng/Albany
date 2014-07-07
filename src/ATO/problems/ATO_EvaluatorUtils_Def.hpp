//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ATO_EvaluatorUtils.hpp"
#include "Albany_DataTypes.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

#include "ATO_ComputeBasisFunctions.hpp"

/********************  Problem Utils Class  ******************************/

template<typename EvalT, typename Traits>
ATO::EvaluatorUtils<EvalT,Traits>::EvaluatorUtils(
     Teuchos::RCP<Albany::Layouts> dl_) :
     dl(dl_)
{
}

template<typename EvalT, typename Traits>
Teuchos::RCP< PHX::Evaluator<Traits> >
ATO::EvaluatorUtils<EvalT,Traits>::constructComputeBasisFunctionsEvaluator(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<shards::CellTopology>& cellType,
    const Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis,
    const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature)
{
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::ParameterList;
    using std::string;

    RCP<ParameterList> p = rcp(new ParameterList("Compute Basis Functions"));

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<string>("Coordinate Vector Name","Coord Vec");
    p->set< RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set< RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > > ("Intrepid Basis", intrepidBasis);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
 
    // Outputs: BF, weightBF, Grad BF, weighted-Grad BF, all in physical space
    p->set<string>( "Weights Name",              "Weights"      );
    p->set<string>( "Jacobian Det Name",         "Jacobian Det" );
    p->set<string>( "BF Name",                   "BF"           );
    p->set<string>( "Weighted BF Name",          "wBF"          );
    p->set<string>( "Gradient BF Name",          "Grad BF"      );
    p->set<string>( "Weighted Gradient BF Name", "wGrad BF"     );

    // get topology info.  Throws Teuchos::Exceptions::InvalidParameter if not set.
    std::string& centering = params->get<std::string>("Topology Centering");
    std::string& topology  = params->get<std::string>("Topology Variable Name");

    if( centering == "Element" ){

      p->set<string>( "Topology Variable Name", topology );
      return rcp(new ATO::ComputeBasisFunctions_ElementTopo<EvalT,Traits>(*p,dl));

    if( centering == "Node" ){

      p->set<string>( "Topology Variable Name", topology );
      return rcp(new ATO::ComputeBasisFunctions_NodeTopo<EvalT,Traits>(*p,dl));

    } else {

      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                                 std::endl <<
                                 "Error!  Unknown centering " << centering <<
                                 "!" << std::endl << "Options are (Element, Node)" <<
                                 std::endl);
 
    }
    
    
}
