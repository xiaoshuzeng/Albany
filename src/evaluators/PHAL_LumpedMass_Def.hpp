//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
LumpedMass<EvalT, Traits>::
LumpedMass (const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl)
 : BF          (p.get<std::string>("BF Name"), dl->node_qp_scalar)
 , w_measure   (p.get<std::string>("Weighted Measure Name"), dl->qp_scalar)
 , lumped_mass (p.get<std::string>("Lumped Mass Name"), dl->node_scalar)
{
  if (dl->isSideLayouts) {
    onSide   = true;
    sideSetName = p.get<std::string>("Side Set Name");

    numNodes = dl->node_qp_scalar->dimension(2);
    numQPs   = dl->node_qp_scalar->dimension(3);
  } else {
    onSide   = false;
    numNodes = dl->node_qp_scalar->dimension(1);
    numQPs   = dl->node_qp_scalar->dimension(2);
  }

  const std::string& lumping_type_str = p.get<std::string>("Lumping Type");
  if (lumping_type_str=="Row Sum") {
    lumping_type = ROW_SUM;
  } else if (lumping_type_str=="Diagonal Scaling") {
    lumping_type = DIAG_SCALING;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
                                "Error! Invalid choice for 'Lumping Type'. Supported options: 'Row Sum', 'Diagonal Scaling'.\n");
  }

  this->addDependentField(BF);
  this->addDependentField(w_measure);
  this->addEvaluatedField(lumped_mass);

  this->setName("LumpedMass" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LumpedMass<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(lumped_mass,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LumpedMass<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (onSide) {
    evaluateFieldsOnSides (workset);
  } else {
    evaluateFieldsOnCells (workset);
  }
}

template<typename EvalT, typename Traits>
void LumpedMass<EvalT, Traits>::evaluateFieldsOnCells (typename Traits::EvalData workset)
{
  switch (lumping_type) {

    case ROW_SUM:
      for (int cell=0; cell<workset.numCells; ++cell) {
        for (int inode=0; inode<numNodes; ++inode) {
          MeshScalarT mass = 0;
          for (int jnode=0; jnode<numNodes; ++jnode) {
            for (int qp=0; qp<numQPs; ++qp) {
              mass += BF(cell,inode,qp)*BF(cell,jnode,qp)*w_measure(cell,qp);
            }
          }
          lumped_mass(cell,inode) = mass;
        }
      }
      break;

    case DIAG_SCALING:
      for (int cell=0; cell<workset.numCells; ++cell) {
        for (int inode=0; inode<numNodes; ++inode) {
          MeshScalarT mass_ii = 0;
          for (int qp=0; qp<numQPs; ++qp) {
            mass_ii += std::pow(BF(cell,inode,qp),2)*w_measure(cell,qp);
          }
          MeshScalarT mass_jj_sum = 0;
          for (int jnode=0; jnode<numNodes; ++jnode) {
            for (int qp=0; qp<numQPs; ++qp) {
              mass_jj_sum += std::pow(BF(cell,jnode,qp),2)*w_measure(cell,qp);
            }
          }
          lumped_mass(cell,inode) = mass_ii/mass_jj_sum;
        }
      }
     break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unexpected error in LumpedMass. Please, report to developers.\n");
  }
}

template<typename EvalT, typename Traits>
void LumpedMass<EvalT, Traits>::evaluateFieldsOnSides (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    switch (lumping_type) {

      case ROW_SUM:
        for (int inode=0; inode<numNodes; ++inode) {
          MeshScalarT mass = 0;
          for (int jnode=0; jnode<numNodes; ++jnode) {
            for (int qp=0; qp<numQPs; ++qp) {
              mass += BF(cell,side,inode,qp)*BF(cell,side,jnode,qp)*w_measure(cell,side,qp);
            }
          }
          lumped_mass(cell,side,inode) = mass;
        }

        break;

      case DIAG_SCALING:
        for (int inode=0; inode<numNodes; ++inode) {
          MeshScalarT mass_ii = 0;
          for (int qp=0; qp<numQPs; ++qp) {
            mass_ii += std::pow(BF(cell,side,inode,qp),2)*w_measure(cell,side,qp);
          }
          MeshScalarT mass_jj_sum = 0;
          for (int jnode=0; jnode<numNodes; ++jnode) {
            for (int qp=0; qp<numQPs; ++qp) {
              mass_jj_sum += std::pow(BF(cell,side,jnode,qp),2)*w_measure(cell,side,qp);
            }
          }
          lumped_mass(cell,side,inode) = mass_ii/mass_jj_sum;
        }
       break;

      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Unexpected error in LumpedMass. Please, report to developers.\n");
    }
  }
}

} // namespace PHAL
