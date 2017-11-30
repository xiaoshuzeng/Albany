//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_PROBLEM_HPP
#define FELIX_HYDROLOGY_PROBLEM_HPP 1

#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Dimension.hpp"
#include "PHAL_FieldFrobeniusNorm.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_Workset.hpp"

#include "FELIX_HydrologyBasalGravitationalWaterPotential.hpp"
#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_EffectivePressure.hpp"
#include "FELIX_FlowFactorA.hpp"
#include "FELIX_ParamEnum.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_SimpleOperation.hpp"
#include "FELIX_HydrologyDirichlet.hpp"
#include "FELIX_HydrologyWaterDischarge.hpp"
#include "FELIX_HydrologyWaterSource.hpp"
#include "FELIX_HydrologyWaterThickness.hpp"
#include "FELIX_HydrologyMeltingRate.hpp"
#include "FELIX_HydrologyResidualPotentialEqn.hpp"
#include "FELIX_HydrologyResidualThicknessEqn.hpp"

namespace FELIX
{

/*!
 * \brief  A 2D problem for the subglacial hydrology
 */
class Hydrology : public Albany::AbstractProblem
{
public:

  //! Default constructor
  Hydrology (const Teuchos::RCP<Teuchos::ParameterList>& topLevelPparams,
             const Teuchos::RCP<Teuchos::ParameterList>& problemPparams,
             const Teuchos::RCP<Teuchos::ParameterList>& discParams,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDimensions);

  //! Destructor
  virtual ~Hydrology();

  //! Return number of spatial dimensions
  virtual int spatialDimension () const
  {
      return numDim;
  }

  //! Get boolean telling code if SDBCs are utilized
  virtual bool useSDBCs() const {return false; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  virtual void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                             Albany::StateManager& stateMgr);

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  //! Main problem setup routine. Not directly called, but indirectly by buildEvaluators
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Boundary conditions evaluators
  void constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators   (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  bool has_h_equation;
  bool eliminate_h;
  bool unsteady;

  int numDim;
  std::string elementBlockName;

  //! Top level parameter list
  Teuchos::RCP<Teuchos::ParameterList> topLevelParams;
  //! Discretization parameter list
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  Teuchos::ArrayRCP<std::string> dof_names;
  Teuchos::ArrayRCP<std::string> dof_names_dot;
  Teuchos::ArrayRCP<std::string> resid_names;

  Teuchos::RCP<Albany::Layouts> dl;

  Teuchos::RCP<shards::CellTopology> cellType;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature;

  static constexpr char hydraulic_potential_name[]          = "hydraulic_potential";
  static constexpr char hydraulic_potential_gradient_name[] = "hydraulic_potential Gradient";
  static constexpr char water_thickness_name[]              = "water_thickness";
  static constexpr char water_thickness_dot_name[]          = "water_thickness_dot";

  static constexpr char effective_pressure_name[]           = "effective_pressure";
  static constexpr char ice_thickness_name[]                = "ice_thickness";
  static constexpr char surface_height_name[]               = "surface_height";
  static constexpr char beta_name[]                         = "beta";
  static constexpr char melting_rate_name[]                 = "melting_rate";
  static constexpr char surface_water_input_name[]          = "surface_water_input";
  static constexpr char surface_mass_balance_name[]         = "surface_mass_balance";
  static constexpr char geothermal_flux_name[]              = "geothermal_flux";
  static constexpr char water_discharge_name[]              = "water_discharge";
  static constexpr char sliding_velocity_name[]             = "sliding_velocity";
  static constexpr char basal_velocity_name[]               = "basal_velocity";
  static constexpr char basal_gravitational_water_potential_name[] = "basal_gravitational_water_potential";
};

// ===================================== IMPLEMENTATION ======================================= //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Hydrology::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                const Albany::MeshSpecsStruct& meshSpecs,
                                Albany::StateManager& stateMgr,
                                Albany::FieldManagerChoice fieldManagerChoice,
                                const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string state_name, field_name, param_name;

  // Getting the names of the distributed parameters (they won't have to be loaded as states)
  std::map<std::string,bool> is_dist_param;
  std::map<std::string,bool> is_dist_params_optimized_upon;
  std::map<std::string,bool> is_dist;
  std::map<std::string,std::string> dist_params_name_to_mesh_part;
  std::set<std::string> inputs_found;
  if (this->params->isSublist("Distributed Parameters"))
  {
    Teuchos::ParameterList& dist_params_list =  this->params->sublist("Distributed Parameters");
    Teuchos::ParameterList* param_list;
    int numParams = dist_params_list.get<int>("Number of Parameter Vectors",0);
    for (int p_index=0; p_index< numParams; ++p_index)
    {
      std::string parameter_sublist_name = Albany::strint("Distributed Parameter", p_index);
      if (dist_params_list.isSublist(parameter_sublist_name))
      {
        // The better way to specify dist params: with sublists
        param_list = &dist_params_list.sublist(parameter_sublist_name);
        param_name = param_list->get<std::string>("Name");
        dist_params_name_to_mesh_part[param_name] = param_list->get<std::string>("Mesh Part","");
      }
      else
      {
        // Legacy way to specify dist params: with parameter entries. Note: no mesh part can be specified.
        param_name = dist_params_list.get<std::string>(Albany::strint("Parameter", p_index));
        dist_params_name_to_mesh_part[param_name] = "";
      }
      is_dist_param[param_name] = true;
      is_dist_params_optimized_upon[param_name] = false;
    }
  }

  // Although they may not be specified in the mesh section or may be specified but only
  // as Input or only as Output, parameters that we are optimizing upon must always be
  // gathered/scattered. To enforce that, we first get their names.
  // NOTE: so far we implement this check only for ROL-type analysis
  if (topLevelParams->sublist("Piro").sublist("Analysis").isSublist("ROL"))
  {
    // Get the # of non-distributed parameters. For ROL there is no distinction, so
    // we need to know if, say, parameter 2 is distributed.
    auto& plist = this->params->sublist("Parameters");
    int num_p = plist.isParameter("Number") ? plist.get<int>("Number") : plist.get<int>("Number of Parameter Vectors",0);

    const auto& rol_params = topLevelParams->sublist("Piro").sublist("Analysis").sublist("ROL");
    int np = rol_params.isParameter("Number of Parameters") ? rol_params.get<int>("Number of Parameter") : 1;
    for (int ip=0; ip<np; ++ip) {
      // Get the idx of the ip-th parameter optimized upon.
      const std::string key = Albany::strint("Parameter Vector Index",ip);
      int idx = rol_params.isParameter(key) ? rol_params.get<int>(key) : ip;
      if (idx>num_p) {
        // Ok, we're optimizing upon a distributed parameter. Let's store its name
        idx = idx-num_p;
        const auto& dist_p_list = this->params->sublist("Distributed Parameters").sublist(Albany::strint("Distributed Parameter", idx) );
        const std::string& dist_p_name = dist_p_list.get<std::string>("Name");
        is_dist_params_optimized_upon[dist_p_name] = true;
      }
    }
  }

  // Now we can start register parameters
  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  std::string fieldType, fieldUsage, meshPart;
  bool nodal_state, vector_state;
  for (int ifield=0; ifield<num_fields; ++ifield)
  {
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(Albany::strint("Field", ifield));

    // Get current state specs
    state_name  = field_name = thisFieldList.get<std::string>("Field Name");
    fieldType  = thisFieldList.get<std::string>("Field Type");
    fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

    if (fieldUsage == "Unused")
      continue;

    bool inputField  = (fieldUsage == "Input")  || (fieldUsage == "Input-Output") || (is_dist_param[state_name] && is_dist_params_optimized_upon[state_name]);
    bool outputField = (fieldUsage == "Output") || (fieldUsage == "Input-Output") || (is_dist_param[state_name] && is_dist_params_optimized_upon[state_name]);

    inputs_found.insert(state_name);
    meshPart = is_dist_param[state_name] ? dist_params_name_to_mesh_part[state_name] : "";

    if(fieldType == "Elem Scalar") {
      entity = Albany::StateStruct::ElemData;
      p = stateMgr.registerStateVariable(state_name, dl->cell_scalar2, elementBlockName, true, &entity, meshPart);
      nodal_state = false;
      vector_state = false;
    }
    else if(fieldType == "Node Scalar") {
      entity = is_dist_param[state_name] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(state_name, dl->node_scalar, elementBlockName, true, &entity, meshPart);
      nodal_state = true;
      vector_state = false;
    }
    else if(fieldType == "Elem Vector") {
      entity = Albany::StateStruct::ElemData;
      p = stateMgr.registerStateVariable(state_name, dl->cell_vector, elementBlockName, true, &entity, meshPart);
      nodal_state = false;
      vector_state = true;
    }
    else if(fieldType == "Node Vector") {
      entity = is_dist_param[state_name] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(state_name, dl->node_vector, elementBlockName, true, &entity, meshPart);
      nodal_state = true;
      vector_state = true;
    }

    if (outputField)
    {
      if (is_dist_param[state_name])
      {
        // A parameter: scatter it
        ev = evalUtils.constructScatterScalarNodalParameter(state_name,field_name);
        fm0.template registerEvaluator<EvalT>(ev);

        // Only the residual FM and EvalT=PHAL::AlbanyTraits::Residual will actually evaluate anything
        if ( fieldManagerChoice==Albany::BUILD_RESID_FM && ev->evaluatedFields().size()>0) {
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      } else {
        // A 'regular' field output: save it.
        p->set<bool>("Nodal State", nodal_state);
        ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
        if (ev->evaluatedFields().size()>0)
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }

    if (inputField) {
      if (is_dist_param[state_name])
      {
        // A parameter: gather it
        ev = evalUtils.constructGatherScalarNodalParameter(state_name,field_name);
        fm0.template registerEvaluator<EvalT>(ev);
      } else {
        // A 'regular' field input: load it.
        p->set<std::string>("Field Name", field_name);
        ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }

  // If a distributed parameter that is optimized upon was not in the discretization list,
  // we throw an execption. The user should list the distributed parameter somewhere, and
  // set 'Field Origin' to 'Mesh'.
  for (auto it : is_dist_params_optimized_upon) {
    if (inputs_found.find(it.first)==inputs_found.end()) {
      // The user has not specified this distributed parameter in the discretization list.
      // That's ok, since we have all we need to register the state ourselve, and create
      // and register its gather/scatter evaluators

      // Get info
      entity = Albany::StateStruct::NodalDistParameter;
      meshPart = dist_params_name_to_mesh_part[it.first];

      // Register the state
      p = stateMgr.registerStateVariable(it.first, dl->node_scalar, elementBlockName, true, &entity, meshPart);

      // Gather evaluator
      ev = evalUtils.constructGatherScalarNodalParameter(it.first,it.first);
      fm0.template registerEvaluator<EvalT>(ev);

      // Scatter evaluator
      ev = evalUtils.constructScatterScalarNodalParameter(it.first,it.first);
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // -------------------- Interpolation and utilities ------------------------ //

  int offset_phi = 0;
  int offset_h = 1;

  // Gather solution field
  if (unsteady)
  {
    Teuchos::ArrayRCP<std::string> tmp;
    tmp.resize(1);
    tmp[0] = dof_names[0];

    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, dof_names, offset_phi);
    fm0.template registerEvaluator<EvalT> (ev);

    tmp[0] = dof_names[1];
    ev = evalUtils.constructGatherSolutionEvaluator (false, tmp, dof_names_dot, offset_h);
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, dof_names);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Compute basis functions
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  int offset = 0;
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Hydrology");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Hydraulic Potential
  ev = evalUtils.constructDOFInterpolationEvaluator(hydraulic_potential_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Effective Pressure
  ev = evalUtils.constructDOFInterpolationEvaluator(effective_pressure_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // In case we want to save Water Discharge
  ev = evalUtils.constructQuadPointsToCellInterpolationEvaluator(water_discharge_name,dl->qp_vector,dl->cell_vector);
  fm0.template registerEvaluator<EvalT>(ev);

  // Water Thickness
  if (has_h_equation && !eliminate_h)
  {
    ev = evalUtils.constructDOFInterpolationEvaluator(water_thickness_name);
    fm0.template registerEvaluator<EvalT> (ev);
    if (unsteady)
    {
      // Interpolate Water Thickness Time Derivative
      ev = evalUtils.constructDOFInterpolationEvaluator(water_thickness_dot_name);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }
  else if (!has_h_equation)
  {
    ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(water_thickness_name);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Hydraulic Potential Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(hydraulic_potential_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // Basal Velocity
  ev = evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator(basal_velocity_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // Surface Water Input
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(surface_water_input_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // Geothermal Flux
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(geothermal_flux_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // Lumped mass matrix diagonal
  ev = evalUtils.constructLumpedMassEvaluator("Lumped Mass","Row Sum");
  fm0.template registerEvaluator<EvalT> (ev);

  // --------------------------------- FELIX evaluators -------------------------------- //

  if (params->sublist("FELIX Hydrology").get<bool>("Use SMB To Approximate Water Input",false))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (inputs_found.find(surface_mass_balance_name)==inputs_found.end(), std::logic_error,
                                "Error! The field '" << surface_mass_balance_name << "' is required " <<
                                "(due to 'Use SMB To Approximate Water Input' being true), " <<
                                "but was not found in the discretization list.\n");

    //--- Compute Water Input From SMB
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Hydrology Water Input"));

    // Input
    p->set<std::string>("Surface Mass Balance Variable Name",surface_mass_balance_name);

    // Output
    p->set<std::string>("Surface Water Input Variable Name","surface_water_input");

    ev = Teuchos::rcp(new FELIX::HydrologyWaterSource<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // ------- Hydrology Basal Gravitational Potential -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Basal Gravitational Water Potential"));

  //Input
  p->set<std::string> ("Surface Height Variable Name",surface_height_name);
  p->set<std::string> ("Ice Thickness Variable Name",ice_thickness_name);
  p->set<bool> ("Is Stokes", false);

  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Basal Gravitational Water Potential Variable Name",basal_gravitational_water_potential_name);

  ev = Teuchos::rcp(new FELIX::BasalGravitationalWaterPotential<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Water Discharge -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Water Discharge"));

  p->set<std::string> ("Water Thickness Variable Name",water_thickness_name);
  p->set<std::string> ("Hydraulic Potential Gradient Variable Name",hydraulic_potential_gradient_name);
  p->set<std::string> ("Regularization Parameter Name","Regularization");

  p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Water Discharge Variable Name",water_discharge_name);

  if (has_h_equation)
    ev = Teuchos::rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl));
  else
    ev = Teuchos::rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));

  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  //Input
  p->set<std::string> ("Geothermal Heat Source Variable Name",geothermal_flux_name);
  p->set<std::string> ("Sliding Velocity Variable Name",sliding_velocity_name);
  p->set<std::string> ("Basal Friction Coefficient Variable Name",beta_name);
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);

  ev = Teuchos::rcp(new FELIX::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (params->sublist("FELIX Hydrology").get<bool>("Thickness Equation Nodal", false) ||
      params->sublist("FELIX Hydrology").get<bool>("Lump Mass In Potential Equation", false)) {
    // We need the melting rate in the nodes
    p->set<bool>("Nodal", true);

    ev = Teuchos::rcp(new FELIX::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,false>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --------- Flow Factor A --------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Flow Factor A"));

  // Input
  p->set<std::string>("Flow Rate Type",params->sublist("FELIX Basal Friction Coefficient").get<std::string>("Flow Rate Type","Uniform"));

  // Output
  p->set<std::string>("Flow Factor A Variable Name","Flow Factor A");

  ev = Teuchos::rcp(new FELIX::FlowFactorA<EvalT,PHAL::AlbanyTraits, false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Sliding Velocity -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name",basal_velocity_name);
  p->set<std::string>("Field Layout","Cell QuadPoint Vector");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name",sliding_velocity_name);

  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Effective pressure calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure"));

  // Input
  p->set<std::string>("Surface Height Variable Name",surface_height_name);
  p->set<std::string>("Ice Thickness Variable Name", ice_thickness_name);
  p->set<std::string>("Hydraulic Potential Variable Name", hydraulic_potential_name);
  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

  // Output
  p->set<std::string>("Effective Pressure Variable Name",effective_pressure_name);

  ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits, false, false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity Variable Name", sliding_velocity_name);
  p->set<std::string>("BF Variable Name", Albany::bf_name);
  p->set<std::string>("Effective Pressure Variable Name", effective_pressure_name);
  p->set<std::string>("Flow Factor A Variable Name", "Flow Factor A");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", beta_name);

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,true,false,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient nodal (for output in the mesh) ---//
  p->set<bool>("Nodal",true);
  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,true,false,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Sliding Velocity -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name",basal_velocity_name);
  p->set<std::string>("Field Layout","Cell Node Vector");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name",sliding_velocity_name);

  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Residual Potential Eqn-------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Potential Eqn"));

  //Input
  p->set<std::string> ("BF Name", Albany::bf_name);
  p->set<std::string> ("Gradient BF Name", Albany::grad_bf_name);
  p->set<std::string> ("Weighted Measure Name", Albany::weights_name);
  p->set<std::string> ("Water Discharge Variable Name", water_discharge_name);
  p->set<std::string> ("Effective Pressure Variable Name", effective_pressure_name);
  p->set<std::string> ("Water Thickness Variable Name", water_thickness_name);
  p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);
  p->set<std::string> ("Surface Water Input Variable Name",surface_water_input_name);
  p->set<std::string> ("Sliding Velocity Variable Name",sliding_velocity_name);
  p->set<std::string> ("Flow Factor A Variable Name","Flow Factor A");
  p->set<std::string> ("Basal Gravitational Water Potential Variable Name",basal_gravitational_water_potential_name);

  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology Parameters",&params->sublist("FELIX Hydrology"));

  //Output
  p->set<std::string> ("Potential Eqn Residual Name",resid_names[0]);

  if (has_h_equation)
    ev = Teuchos::rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,true,false,false>(*p,dl));
  else
    ev = Teuchos::rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,false,false,false>(*p,dl));

  fm0.template registerEvaluator<EvalT>(ev);

  if (has_h_equation)
  {
    if (eliminate_h) {
      // -------- Hydrology Water Thickness (QPs) ------- //
      p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Water Thickness"));

      //Input
      p->set<std::string> ("Water Thickness Variable Name",water_thickness_name);
      p->set<std::string> ("Effective Pressure Variable Name",effective_pressure_name);
      p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);
      p->set<std::string> ("Sliding Velocity Variable Name",sliding_velocity_name);
      p->set<std::string> ("Flow Factor A Variable Name","Flow Factor A");
      p->set<bool> ("Nodal", false);
      p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
      p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

      //Output
      p->set<std::string> ("Water Thickness Variable Name", water_thickness_name);

      ev = Teuchos::rcp(new FELIX::HydrologyWaterThickness<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

      // -------- Hydrology Water Thickness (nodes) ------- //
      p->set<bool> ("Nodal", true);
      ev = Teuchos::rcp(new FELIX::HydrologyWaterThickness<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

    } else {
      // ------- Hydrology Thickness Residual -------- //
      p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Thickness"));

      //Input
      p->set<std::string> ("BF Name", Albany::bf_name);
      p->set<std::string> ("Weighted Measure Name", Albany::weights_name);
      p->set<std::string> ("Water Thickness Variable Name",water_thickness_name);
      p->set<std::string> ("Water Thickness Dot Variable Name",water_thickness_dot_name);
      p->set<std::string> ("Effective Pressure Variable Name",effective_pressure_name);
      p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);
      p->set<std::string> ("Sliding Velocity Variable Name",sliding_velocity_name);
      p->set<std::string> ("Flow Factor A Variable Name","Flow Factor A");
      p->set<bool> ("Unsteady", unsteady);
      p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
      p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

      //Output
      p->set<std::string> ("Thickness Eqn Residual Name", resid_names[1]);

      ev = Teuchos::rcp(new FELIX::HydrologyResidualThicknessEqn<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // -------- Regularization from Homotopy Parameter h: reg = 10^(-10*h)
  p = Teuchos::rcp(new Teuchos::ParameterList("Simple Op"));

  //Input
  p->set<std::string> ("Input Field Name",ParamEnum::HomotopyParam_name);
  p->set<Teuchos::RCP<PHX::DataLayout>> ("Field Layout",dl->shared_param);
  p->set<double>("Tau",-10.0*log(10.0));

  //Output
  p->set<std::string> ("Output Field Name","Regularization");

  ev = Teuchos::rcp(new FELIX::SimpleOperationExp<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter: Uniform Flow Factor A ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Constant Flow Factor A"));

  param_name = ParamEnum::FlowFactorA_name;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::FlowFactorA>> ptr_A;
  ptr_A = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::FlowFactorA>(*p,dl));
  ptr_A->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Hydrology").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_A);

  //--- Shared Parameter for basal friction coefficient: lambda ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

  param_name = ParamEnum::Lambda_name;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>> ptr_lambda;
  ptr_lambda = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>(*p,dl));
  ptr_lambda->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_lambda);

  //--- Shared Parameter for basal friction coefficient: mu ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

  param_name = ParamEnum::Mu_name;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>> ptr_mu;
  ptr_mu = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>(*p,dl));
  ptr_mu->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_mu);

  //--- Shared Parameter for basal friction coefficient: power ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

  param_name = ParamEnum::Power_name;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>> ptr_power;
  ptr_power = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>(*p,dl));
  ptr_power->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_power);

  //--- Shared Parameter for Continuation:  ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  param_name = ParamEnum::HomotopyParam_name;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  // Note: homotopy param (h) is used to regularize. Hence, set default to 1.0, in case there's no continuation,
  //       so we regularize very little. Recall that if no nominal values are set in input files, setNominalValue picks
  //       the value passed as second input.
  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),1.0);
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  // ----------------------------------------------------- //

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Hydrology", dl->dummy);
    fm0.template requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));

    Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_PROBLEM_HPP
