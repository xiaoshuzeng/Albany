//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherScalarNodalParameterBase<EvalT,Traits>::
GatherScalarNodalParameterBase(const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  param_name = p.get<std::string>("Parameter Name");
  val = PHX::MDField<ScalarT,Cell,Node>(param_name,dl->node_scalar);
  numNodes = 0;

  this->addEvaluatedField(val);

  this->setName("Gather Nodal Parameter"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameterBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val,fm);
  numNodes = val.dimension(1);
}

// **********************************************************************

template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,dl)
{
}

template<typename EvalT, typename Traits>
GatherScalarNodalParameter<EvalT, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<EvalT, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherScalarNodalParameter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Epetra_Vector> pvec =
    workset.distParamLib->get(this->param_name)->overlapped_vector();

  const Albany::IDArray&  wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  for (std::size_t cell=0; cell < workset.numCells; ++cell )
    for (std::size_t node = 0; node < this->numNodes; ++node)
    {
      int lid = wsElDofs((int)cell,(int)node,0);
      (this->val)(cell,node) = (lid >= 0 ) ? (*pvec)[wsElDofs((int)cell,(int)node,0)] : 0;
    }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
}

template<typename Traits>
GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherScalarNodalParameter(const Teuchos::ParameterList& p) :
  GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
}

// **********************************************************************
template<typename Traits>
void GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Distributed parameter vector
  Teuchos::RCP<const Epetra_Vector> pvec =
    workset.distParamLib->get(this->param_name)->overlapped_vector();

  const Albany::IDArray&  wsElDofs = workset.distParamLib->get(this->param_name)->workset_elem_dofs()[workset.wsIndex];

  // Are we differentiating w.r.t. this parameter?
  bool is_active = (workset.dist_param_deriv_name == this->param_name);

  // If active, intialize data needed for differentiation
  if (is_active) {
    const int num_deriv = this->numNodes;
    const int num_nodes_res = this->numNodes;
    bool trans = workset.transpose_dist_param_deriv;
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& resID  = workset.wsElNodeEqID[cell];

      for (std::size_t node = 0; node < num_deriv; ++node) {

        // Initialize Fad type for parameter value
        int id = wsElDofs((int)cell,(int)node,0);
        double pvec_id = (id >=0) ? (*pvec)[id] : 0;
        ScalarT v(num_deriv, node, pvec_id);
        v.setUpdateValue(!workset.ignore_residual);
        (this->val)(cell,node) = v;
      }

      if(workset.Vp != Teuchos::null) {
        const Epetra_MultiVector& Vp = *(workset.Vp);
        const int num_cols = Vp.NumVectors();


        Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
                workset.local_Vp[cell];


        if(trans) {
          local_Vp.resize(num_nodes_res*workset.numEqs);
          for (std::size_t node = 0; node < num_nodes_res; ++node) {
            // Store Vp entries
            const Teuchos::ArrayRCP<int>& eqID  = resID[node];
            for (std::size_t eq = 0; eq < workset.numEqs; eq++) {
              local_Vp[node*workset.numEqs+eq].resize(num_cols);
              int id = eqID[eq];
              for (int col=0; col<num_cols; ++col)
                local_Vp[node*workset.numEqs+eq][col] = Vp[col][id];
            }
          }
        }
        else {
          local_Vp.resize(num_deriv);
          for (std::size_t node = 0; node < num_deriv; ++node) {
            int id = wsElDofs((int)cell,(int)node,0);
            local_Vp[node].resize(num_cols);
            for (int col=0; col<num_cols; ++col)
              local_Vp[node][col] = (id >=0) ?  Vp[col][id] : 0;
          }
        }
      }
    }
  }

  // If not active, just set the parameter value in the phalanx field
  else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
         int lid = wsElDofs((int)cell,(int)node,0);
          (this->val)(cell,node) = (lid >=0) ? (*pvec)[lid] : 0;
      }
    }
  }
}

// **********************************************************************

}
