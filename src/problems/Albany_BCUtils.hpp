//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_BCUTILS_HPP
#define ALBANY_BCUTILS_HPP

#include <string>
#include <vector>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Phalanx_Evaluator_TemplateManager.hpp>

#include "Albany_DataTypes.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "Albany_BCTraits.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_FactoryTraits.hpp"

namespace Albany {

/*!
 * \brief Generic Functions to help define BC Field Manager
 */

template <typename BCTraits>
class BCUtils {
 public:
  BCUtils() {}

  //! Type of traits class being used
  typedef BCTraits traits_type;

  //! Function to check if the BC section of input file is
  //! present
  static bool
  haveBCSpecified(const Teuchos::RCP<Teuchos::ParameterList>& params) {
    // If the BC sublist is not in the input file,
    // side/node sets can be contained in the Exodus file but are not defined in
    // the problem statement.
    // This is OK, just return

    return params->isSublist(traits_type::bcParamsPl);
  }

  Teuchos::Array<Teuchos::Array<int>>
  getOffsets() const {
    return offsets_;
  }

  bool useSDBCs() const { return use_sdbcs_; }

  //! This versions will be implemented by Dirichlet Traits
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
    constructBCEvaluators(
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& bcNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib, const int numEqn = 0);

  void constructBCEvaluators(
      Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>& fm,
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& bcNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib, const int numEqn = 0);

  //! This version will be implemented by Neumann Traits
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  constructBCEvaluators(
      const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
      const std::vector<std::string>& bcNames,
      const Teuchos::ArrayRCP<std::string>& dof_names, bool isVectorField,
      int offsetToFirstDOF, const std::vector<std::string>& conditions,
      const Teuchos::Array<Teuchos::Array<int>>& offsets,
      const Teuchos::RCP<Albany::Layouts>& dl,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib,
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

  void constructBCEvaluators(
      Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>& fm,
      const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
      const std::vector<std::string>& bcNames,
      const Teuchos::ArrayRCP<std::string>& dof_names, bool isVectorField,
      int offsetToFirstDOF, const std::vector<std::string>& conditions,
      const Teuchos::Array<Teuchos::Array<int>>& offsets,
      const Teuchos::RCP<Albany::Layouts>& dl,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib,
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

  //! This version will be implemented by Side Eq Dirichlet Traits
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
  constructBCEvaluators(
      const std::vector<std::string>& sideSetNames,
      const std::map<std::string,std::vector<std::string>>& nodeSetNames,
      const std::vector<std::string>& dofNames,
      const std::vector<int>& dofOffsets,
      const Teuchos::RCP<Albany::Layouts>& dl,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib);

  void constructBCEvaluators(
      Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>& fm,
      const std::vector<std::string>& sideSetNames,
      const std::map<std::string,std::vector<std::string>>& nodeSetNames,
      const std::vector<std::string>& dofNames,
      const std::vector<int>& dofOffsets,
      const Teuchos::RCP<Albany::Layouts>& dl,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib);

 private:
  //! Builds the list

  //! This version will be implemented by Dirichlet Traits
  void
  buildEvaluatorsList(
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
          evaluatorss_to_build,
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& bcNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib, const int numEqn);

  //! This version will be implemented by Neumann Traits
  void
  buildEvaluatorsList(
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
          evaluators_to_build,
      const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
      const std::vector<std::string>& bcNames,
      const Teuchos::ArrayRCP<std::string>& dof_names, bool isVectorField,
      int offsetToFirstDOF, const std::vector<std::string>& conditions,
      const Teuchos::Array<Teuchos::Array<int>>& offsets,
      const Teuchos::RCP<Albany::Layouts>& dl,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib,
      const Teuchos::RCP<Albany::MaterialDatabase>& materialDB = Teuchos::null);

  //! This version will be implemented by Side Eq Dirichlet Traits
  void
  buildEvaluatorsList(
      std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
          evaluatorss_to_build,
      const std::vector<std::string>& sideSetNames,
      const std::map<std::string,std::vector<std::string>>& nodeSetNames,
      const std::vector<std::string>& dofNames,
      const std::vector<int>& dofOffsets,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib);

  //! Generic implementation of Field Manager construction function
  void buildFieldManager(
      Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>& fm,
      const Teuchos::RCP<std::vector<
          Teuchos::RCP<PHX::Evaluator_TemplateManager<PHAL::AlbanyTraits>>>>
          evaluators,
      std::string& allBC, Teuchos::RCP<PHX::DataLayout>& dummy);

 protected:
   Teuchos::Array<Teuchos::Array<int>> offsets_;
   bool use_sdbcs_{false};
};

} // namespace Albany

#endif // ALBANY_BCUTILS_HPP
