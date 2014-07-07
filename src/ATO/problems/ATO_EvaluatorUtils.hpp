//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_EVALUATORUTILS_HPP
#define ATO_EVALUATORUTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Phalanx.hpp"
#include "Albany_DataTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "Teuchos_VerboseObject.hpp"

#include "Albany_ProblemUtils.hpp"

#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"


namespace ATO {
  /*!
   * \brief Generic Functions to construct evaluators more succinctly
   */
  template<typename EvalT, typename Traits>
  class EvaluatorUtils {

   public:

    EvaluatorUtils(Teuchos::RCP<Albany::Layouts> dl);

    //! Function to create parameter list for construction of ComputeBasisFunctions
    //! evaluator with standard Field names
    Teuchos::RCP< PHX::Evaluator<Traits> > 
    constructComputeBasisFunctionsEvaluator(
      const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<shards::CellTopology>& cellType,
      const Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis,
      const Teuchos::RCP<Intrepid::Cubature<RealType> > cubature);

  private:

    //! Struct of PHX::DataLayout objects defined all together.
    Teuchos::RCP<Albany::Layouts> dl;

  };
}

#endif 
