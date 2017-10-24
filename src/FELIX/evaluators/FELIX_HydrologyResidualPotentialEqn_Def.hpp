//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokesCoupling>
HydrologyResidualPotentialEqn<EvalT, Traits, HasThicknessEqn, IsStokesCoupling>::
HydrologyResidualPotentialEqn (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl) :
  BF        (p.get<std::string> ("BF Name"), dl->node_qp_scalar),
  GradBF    (p.get<std::string> ("Gradient BF Name"), dl->node_qp_gradient),
  w_measure (p.get<std::string> ("Weighted Measure Name"), dl->qp_scalar),
  q         (p.get<std::string> ("Water Discharge Variable Name"), dl->qp_gradient),
  N         (p.get<std::string> ("Effective Pressure Variable Name"), dl->qp_scalar),
  m         (p.get<std::string> ("Melting Rate Variable Name"), dl->qp_scalar),
  h         (p.get<std::string> ("Water Thickness Variable Name"), dl->qp_scalar),
  omega     (p.get<std::string> ("Surface Water Input Variable Name"), dl->qp_scalar),
  u_b       (p.get<std::string> ("Sliding Velocity Variable Name"), dl->qp_scalar),
  residual  (p.get<std::string> ("Potential Eqn Residual Name"),dl->node_scalar)
{
  /*
   *  The (hydraulic) potential equation has the following (strong) form
   *
   *     -div(q) + A*h*N^3 + rho_combo*m = -omega + (h_r-h)*|u_b|/l_r
   *
   *  where q is the water discharge, h the water thickness, A the Glen's law
   *  flow factor, N the effective pressure, rho_combo=1/rho_water-1/rho_ice,
   *  m the melting rate of the ice (due to geothermal flow and sliding),
   *  omega the water source (water reaching the bed from the surface, through
   *  moulins), h_r/l_r typical height/length of bed bumps, and u_b is the
   *  sliding velocity of the ice
   */

  if (IsStokesCoupling)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(2);
    numQPs   = dl->qp_scalar->dimension(2);
    numDims  = dl->qp_gradient->dimension(3);

    sideSetName = p.get<std::string>("Side Set Name");

    metric = PHX::MDField<const MeshScalarT,Cell,Side,QuadPoint,Dim,Dim>(p.get<std::string>("Metric Name"),dl->qp_tensor);
    this->addDependentField(metric);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numNodes = dl->node_scalar->dimension(1);
    numQPs   = dl->qp_scalar->dimension(1);
    numDims  = dl->qp_gradient->dimension(2);
  }

  this->addEvaluatedField(residual);

  // Setting parameters
  Teuchos::ParameterList& hydrology_params = *p.get<Teuchos::ParameterList*>("FELIX Hydrology Parameters");
  Teuchos::ParameterList& physical_params  = *p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");

  double rho_i      = physical_params.get<double>("Ice Density", 910.0);
  double rho_w      = physical_params.get<double>("Water Density", 1028.0);
  bool melting_mass = hydrology_params.get<bool>("Use Melting In Conservation Of Mass", false);
  bool melting_cav  = hydrology_params.get<bool>("Use Melting In Thickness Equation", false);
  use_eff_cav       = (hydrology_params.get<bool>("Use Effective Cavities Height", true) ? 1.0 : 0.0);
  eta_i             = physical_params.get<double>("Ice Viscosity",-1.0);

  rho_combo = (melting_mass ? 1.0 : 0.0) / rho_w - (melting_cav ? 1.0 : 0.0) / rho_i;
  mu_w      = physical_params.get<double>("Water Viscosity");
  h_r       = hydrology_params.get<double>("Bed Bumps Height");
  l_r       = hydrology_params.get<double>("Bed Bumps Length");
  A         = hydrology_params.get<double>("Flow Factor Constant");

  mass_lumping = hydrology_params.isParameter("Mass Lumping") ? hydrology_params.get<bool>("Mass Lumping") : false;
  if (mass_lumping) {
    h_nodal = PHX::MDField<const hScalarT>(p.get<std::string> ("Water Thickness Variable Name"), dl->node_scalar);
    N_nodal = PHX::MDField<const ScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar);
    m_nodal = PHX::MDField<const ScalarT>(p.get<std::string> ("Melting Rate Variable Name"), dl->node_scalar);
    this->addDependentField(N_nodal);
    this->addDependentField(h_nodal);
    this->addDependentField(m_nodal);
  } else {
    this->addDependentField(m);
  }
  this->addDependentField(N);
  this->addDependentField(h);
  this->addDependentField(BF);
  this->addDependentField(GradBF);
  this->addDependentField(w_measure);
  this->addDependentField(q);
  this->addDependentField(omega);
  this->addDependentField(u_b);


  /*
   * Scalings, needed to account for different units: ice velocity
   * is in m/yr, the mesh is in km, and hydrology time unit is s.
   *
   * The residual has 5 terms (forget about signs), with the following
   * units (including the km^2 from dx):
   *
   *  1) \int rho_combo*m*v*dx          [m km^2 yr^-1]
   *  2) \int omega*v*dx                [mm km^2 day^-1]
   *  3) \int dot(q*grad(v))*dx         [1000 m^3 s^-1]
   *  4) \int A*h*N^3*v*dx              [1000 m km^2 yr^-1]
   *  5) \int (h_r-h)*|u|/l_r*v*dx      [m km^2 yr^-1]
   *
   * where q=k*h^3*gradPhi/mu_w, and v is the test function (non-dimensional).
   * We decide to uniform all terms to have units [m km^2 s^-1].
   * Where possible, we do this by rescaling some constants. Otherwise,
   * we simply introduce a new scaling factor
   *
   *  1) rho_combo*m                    (no scaling)
   *  2) scaling_omega*omega            scaling_omega = yr_to_day/1000
   *  3) scaling_q*dot(q,grad(v))       scaling_q     = 1e-3*yr_to_s
   *  4) A_mod*h*N^3                    A_mod         = A/1000
   *  5) (h_r-h)*|u|/l_r                (no scaling)
   *
   * where yr_to_s=365.25*24*3600 (the number of seconds in a year)
   */
  double yr_to_s = 365.25*24*3600;
  A               = A/1000;
  scaling_omega   = 365.25/1000;
  scaling_q       = 1e-3*yr_to_s;

  this->setName("HydrologyResidualPotentialEqn"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokesCoupling>
void HydrologyResidualPotentialEqn<EvalT, Traits, HasThicknessEqn, IsStokesCoupling>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(q,fm);
  this->utils.setFieldData(omega,fm);
  this->utils.setFieldData(u_b,fm);

  if (IsStokesCoupling)
    this->utils.setFieldData(metric,fm);

  if (mass_lumping) {
    this->utils.setFieldData(N_nodal,fm);
    this->utils.setFieldData(h_nodal,fm);
    this->utils.setFieldData(m_nodal,fm);
  } else {
    this->utils.setFieldData(m,fm);
  }
  this->utils.setFieldData(N,fm);
  this->utils.setFieldData(h,fm);

  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool HasThicknessEqn, bool IsStokesCoupling>
void HydrologyResidualPotentialEqn<EvalT, Traits, HasThicknessEqn, IsStokesCoupling>::
evaluateFields (typename Traits::EvalData workset)
{
  // Omega is in mm/d rather than m/s
  double scaling_omega = 0.001/(24*3600);

  if (IsStokesCoupling)
  {
    // Zero out, to avoid leaving stuff from previous workset!
    residual.deep_copy(ScalarT(0.));

    if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
      return;

    ScalarT res_qp, res_node;
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      for (int node=0; node < numNodes; ++node)
      {
        res_node = 0;
        for (int qp=0; qp < numQPs; ++qp)
        {
          res_qp = rho_combo*m(cell,side,qp) + scaling_omega*omega(cell,side,qp)
                 - (h_r -h(cell,side,qp))*u_b(cell,side,qp)/l_r
                 + h(cell,side,qp)*std::pow(A*N(cell,side,qp),3);

          res_qp *= BF(cell,side,node,qp);

          for (int idim=0; idim<numDims; ++idim)
          {
            for (int jdim=0; jdim<numDims; ++jdim)
            {
              res_qp += scaling_q*q(cell,side,qp,idim) * metric(cell,side,qp,idim,jdim) * GradBF(cell,side,node,qp,jdim);
            }
          }

          res_node += res_qp * w_measure(cell,side,qp);
        }
        residual (cell,side,node) += res_node;
      }
    }
  }
  else
  {
    if (eta_i>0)
    {
      ScalarT res_qp, res_node;
      for (int cell=0; cell < workset.numCells; ++cell)
      {
        for (int node=0; node < numNodes; ++node)
        {
          res_node = 0;
          for (int qp=0; qp < numQPs; ++qp)
          {
            res_qp = rho_combo*m(cell,qp) + scaling_omega*omega(cell,qp)
                   - (h_r - use_eff_cav*h(cell,qp))*u_b(cell,qp)/l_r
                   + h(cell,qp)*N(cell,qp)/eta_i;

            res_qp *= BF(cell,node,qp);

            for (int dim=0; dim<numDims; ++dim)
            {
              res_qp += scaling_q*q(cell,qp,dim) * GradBF(cell,node,qp,dim);
            }

            res_node += res_qp * w_measure(cell,qp);
          }

          residual (cell,node) = res_node;
        }
      }
    }
    else
    {
      ScalarT res_qp, res_node;
      for (int cell=0; cell < workset.numCells; ++cell)
      {
        for (int node=0; node < numNodes; ++node)
        {
          res_node = 0;
          for (int qp=0; qp < numQPs; ++qp)
          {
            res_qp = scaling_omega*omega(cell,qp)
                   - (h_r - use_eff_cav*h(cell,qp))*u_b(cell,qp)/l_r;
            if (!mass_lumping) {
              res_qp += rho_combo*m(cell,qp) + (2.0/9.0)*h(cell,qp)*A*std::pow(N(cell,qp),3);
            }

            res_qp *= BF(cell,node,qp);

            for (int dim=0; dim<numDims; ++dim)
            {
              res_qp += scaling_q*q(cell,qp,dim) * GradBF(cell,node,qp,dim);
            }

            res_node += res_qp * w_measure(cell,qp);
          }

          if (mass_lumping) {
            res_node += rho_combo*m_nodal(cell,node) + (2.0/9.0)*h_nodal(cell,node)*A*std::pow(N_nodal(cell,node),3);
          }

          residual (cell,node) = res_node;
        }
      }
    }
  }
}

} // Namespace FELIX
