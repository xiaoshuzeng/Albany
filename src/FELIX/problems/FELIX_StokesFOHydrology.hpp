//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP
#define FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP 1

#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_FieldFrobeniusNorm.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"

#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_EffectivePressure.hpp"
#include "FELIX_FlowFactorA.hpp"
#include "FELIX_HydrologyBasalGravitationalWaterPotential.hpp"
#include "FELIX_HydrologyMeltingRate.hpp"
#include "FELIX_HydrologyResidualPotentialEqn.hpp"
#include "FELIX_HydrologyResidualThicknessEqn.hpp"
#include "FELIX_HydrologyWaterDischarge.hpp"
#include "FELIX_ParamEnum.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_SimpleOperation.hpp"
#include "FELIX_StokesFOBasalResid.hpp"
#include "FELIX_StokesFOBodyForce.hpp"
#include "FELIX_StokesFOResid.hpp"
#include "FELIX_ViscosityFO.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

/*!
 * \brief The coupled problem StokesFO+Hydrology
 */
class StokesFOHydrology : public Albany::AbstractProblem
{
public:

  //! Default constructor
  StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<Teuchos::ParameterList>& discParams,
                     const Teuchos::RCP<ParamLib>& paramLib,
                     const int numDim_);

  //! Destructor
  ~StokesFOHydrology();

  //! Return number of spatial dimensions
  virtual int spatialDimension() const { return numDim; }

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

private:

  //! Private to prohibit copying
  StokesFOHydrology(const StokesFOHydrology&) = delete;

  //! Private to prohibit copying
  StokesFOHydrology& operator=(const StokesFOHydrology&) = delete;

public:

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> basalSideType;
  Teuchos::RCP<shards::CellTopology> surfaceSideType;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>  cellCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>  basalCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>  surfaceCubature;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> cellBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> basalSideBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> surfaceSideBasis;

  int numDim;
  int stokes_neq;
  int hydro_neq;

  bool has_h_equation;
  bool unsteady;

  Teuchos::RCP<Albany::Layouts> dl,dl_basal,dl_surface;

  //! Discretization parameter
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  std::string basalSideName;
  std::string surfaceSideName;

  std::string elementBlockName;
  std::string basalEBName;
  std::string surfaceEBName;

  Teuchos::ArrayRCP<std::string> stokes_dof_names;
  Teuchos::ArrayRCP<std::string> stokes_resid_names;

  Teuchos::ArrayRCP<std::string> hydro_dof_names;
  Teuchos::ArrayRCP<std::string> hydro_dof_names_dot;
  Teuchos::ArrayRCP<std::string> hydro_resid_names;

  static constexpr char ice_velocity_name[]        = "ice_velocity";
  static constexpr char hydraulic_potential_name[] = "hydraulic_potential";
  static constexpr char water_thickness_name[]     = "water_thickness";
  static constexpr char water_thickness_dot_name[] = "water_thickness_dot";
};

} // Namespace FELIX

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::StokesFOHydrology::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                               const Albany::MeshSpecsStruct& meshSpecs,
                                               Albany::StateManager& stateMgr,
                                               Albany::FieldManagerChoice fieldManagerChoice,
                                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  int offset=0;

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels = Teuchos::rcp(new std::map<std::string, int> ());

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName, param_name;

  // Getting the names of the distributed parameters (they won't have to be loaded as states)
  std::map<std::string,bool> is_dist_param;
  std::map<std::string,bool> is_extruded_param;
  std::map<std::string,bool> save_sensitivities;
  std::map<std::string,std::string> dist_params_name_to_mesh_part;
  // The following are used later to check that the needed fields (depending on the simulation options) are successfully loaded/gathered
  std::set<std::string> inputs_found;
  std::map<std::string,std::set<std::string>> ss_inputs_found;
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
        is_extruded_param[param_name] = param_list->get<bool>("Extruded",false);
        int extruded_param_level = 0;
        extruded_params_levels->insert(std::make_pair(param_name, extruded_param_level));
      }
      else
      {
        // Legacy way to specify dist params: with parameter entries. Note: no mesh part can be specified.
        param_name = dist_params_list.get<std::string>(Albany::strint("Parameter", p_index));
        dist_params_name_to_mesh_part[param_name] = "";
      }
      is_dist_param[param_name] = true;
    }
  }

  // Registering 3D states and building their load/save/gather evaluators
  if (discParams->isSublist("Required Fields Info"))
  {
    Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
    int num_fields = req_fields_info.get<int>("Number Of Fields",0);

    std::string fieldType, fieldUsage, meshPart;
    bool nodal_state;
    for (int ifield=0; ifield<num_fields; ++ifield)
    {
      Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

      // Get current state specs
      stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
      fieldType  = thisFieldList.get<std::string>("Field Type");
      fieldUsage = thisFieldList.get<std::string>("Field Usage", "Input");

      if (fieldUsage == "Unused")
        continue;

      bool inputField  = (fieldUsage == "Input")  || (fieldUsage == "Input-Output");
      bool outputField = (fieldUsage == "Output") || (fieldUsage == "Input-Output");

      inputs_found.insert(stateName);
      meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

      if(fieldType == "Elem Scalar") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity, meshPart);
        nodal_state = false;
      }
      else if(fieldType == "Node Scalar") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, meshPart);
        nodal_state = true;
      }
      else if(fieldType == "Elem Vector") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, meshPart);
        nodal_state = false;
      }
      else if(fieldType == "Node Vector") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, meshPart);
        nodal_state = true;
      }

      if (outputField)
      {
        if (is_dist_param[stateName])
        {
          // A parameter: scatter it
          ev = evalUtils.constructScatterScalarNodalParameter(stateName,fieldName);
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
        if (is_dist_param[stateName])
        {
          // A parameter: gather it
          ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
          fm0.template registerEvaluator<EvalT>(ev);
        } else {
          // A 'regular' field input: load it.
          p->set<std::string>("Field Name", fieldName);
          ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }
    }
  }

  // Registering 2D states and building their load/save/gather evaluators
  // Note: we MUST have 'Side Set Discretizations', since this is the StokesFOHydrology coupling and we MUST have a basal mesh...
  Teuchos::Array<std::string> ss_names = discParams->sublist("Side Set Discretizations",true).get<Teuchos::Array<std::string>>("Side Sets");
  for (int i=0; i<ss_names.size(); ++i)
  {
    const std::string& ss_name = ss_names[i];
    Teuchos::ParameterList& req_fields_info = discParams->sublist("Side Set Discretizations").sublist(ss_name,true).sublist("Required Fields Info");
    int num_fields = req_fields_info.get<int>("Number Of Fields",0);
    Teuchos::RCP<PHX::DataLayout> dl_temp;
    Teuchos::RCP<PHX::DataLayout> sns;
    std::string fieldType, fieldUsage, meshPart;
    bool nodal_state;
    int numLayers;

    const std::string& sideEBName = meshSpecs.sideSetMeshSpecs.at(ss_name)[0]->ebName;
    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (int ifield=0; ifield<num_fields; ++ifield)
    {
      Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

      // Get current state specs
      stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
      fieldType  = thisFieldList.get<std::string>("Field Type");
      fieldUsage = thisFieldList.get<std::string>("Field Usage", "Input");

      if (fieldUsage == "Unused")
        continue;

      bool inputField  = (fieldUsage == "Input")  || (fieldUsage == "Input-Output");
      bool outputField = (fieldUsage == "Output") || (fieldUsage == "Input-Output");

      ss_inputs_found[ss_name].insert(stateName);
      meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";
      meshPart = ""; // Distributed parameters are defined either on the whole volume mesh or on a whole side mesh. Either way, here we want "" as part (the whole mesh).

      numLayers = thisFieldList.isParameter("Number Of Layers") ? thisFieldList.get<int>("Number Of Layers") : -1;
      fieldType  = thisFieldList.get<std::string>("Field Type");

      if(fieldType == "Elem Scalar") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_scalar2, sideEBName, true, &entity, meshPart);
        nodal_state = false;
      }
      else if(fieldType == "Node Scalar") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_scalar, sideEBName, true, &entity, meshPart);
        nodal_state = true;
      }
      else if(fieldType == "Elem Vector") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_vector, sideEBName, true, &entity, meshPart);
        nodal_state = false;
      }
      else if(fieldType == "Node Vector") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_vector, sideEBName, true, &entity, meshPart);
        nodal_state = true;
      }
      else if(fieldType == "Elem Layered Scalar") {
        entity = Albany::StateStruct::ElemData;
        sns = ss_dl->cell_scalar2;
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,LayerDim>(sns->dimension(0),sns->dimension(1),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }
      else if(fieldType == "Node Layered Scalar") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        sns = ss_dl->node_scalar;
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }
      else if(fieldType == "Elem Layered Vector") {
        entity = Albany::StateStruct::ElemData;
        sns = ss_dl->cell_vector;
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Dim,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }
      else if(fieldType == "Node Layered Vector") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        sns = ss_dl->node_vector;
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,Dim,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),
                                                                               sns->dimension(3),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }

      if (fieldUsage == "Unused")
        continue;

      if (outputField)
      {
        if (is_dist_param[stateName])
        {
          // A parameter: scatter it
          if (is_extruded_param[stateName])
          {
            ev = evalUtils.constructScatterScalarExtruded2DNodalParameter(stateName,fieldName);
            fm0.template registerEvaluator<EvalT>(ev);
          }
          else
          {
            ev = evalUtils.constructScatterScalarNodalParameter(stateName,fieldName);
            fm0.template registerEvaluator<EvalT>(ev);
          }

          // Only the residual FM and EvalT=PHAL::AlbanyTraits::Residual will actually evaluate anything
          if ( fieldManagerChoice==Albany::BUILD_RESID_FM && ev->evaluatedFields().size()>0) {
            fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
          }
        } else {
          // A 'regular' field output: save it.
          p->set<bool>("Nodal State", nodal_state);
          p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
          ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,ss_dl));
          fm0.template registerEvaluator<EvalT>(ev);

          // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
          if (ev->evaluatedFields().size()>0)
            fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      }

      if (inputField) {
        if (is_dist_param[stateName])
        {
          // A parameter: gather it
          if (is_extruded_param[stateName])
          {
            ev = evalUtils.constructGatherScalarExtruded2DNodalParameter(stateName,fieldName);
            fm0.template registerEvaluator<EvalT>(ev);
          }
          else
          {
            ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
            fm0.template registerEvaluator<EvalT>(ev);
          }
        } else {
          // A 'regular' field input: load it.
          p->set<std::string>("Field Name", fieldName);
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }
    }
  }

  // ------------------- Interpolations and utilities ------------------ //

  int offsetStokes = 0;
  int offsetHydro  = stokes_neq;

  // Gather stokes solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, stokes_dof_names, offsetStokes);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather hydrology solution field
  if (has_h_equation && unsteady)
  {
    ev = evalUtils.constructGatherSolutionEvaluator (false, hydro_dof_names, hydro_dof_names_dot, offsetHydro);
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, hydro_dof_names, offsetHydro);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Scatter stokes residual
  ev = evalUtils.constructScatterResidualEvaluator(true, stokes_resid_names, offsetStokes, "Scatter Stokes");
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter hydrology residual(s)
  ev = evalUtils.constructScatterResidualEvaluator(false, hydro_resid_names, offsetHydro, "Scatter Hydrology");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate stokes solution field
  ev = evalUtils.constructDOFVecInterpolationEvaluator(stokes_dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate stokes solution gradient
  ev = evalUtils.constructDOFVecGradInterpolationEvaluator(stokes_dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate effective pressure
  ev = evalUtils.constructDOFInterpolationSideEvaluator("effective_pressure", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("surface_height");
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height gradient
  ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("surface_height");
  fm0.template registerEvaluator<EvalT> (ev);

  // If temperature is loaded as node-based field, then interpolate it as a cell-based field
  ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator("temperature", false);
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------- Special evaluators for basal side handling ----------------- //

  //---- Restrict vertex coordinates from cell-based to cell-side-based on basalside
  ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector",cellType,"Coord Vec " + basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict ice velocity from cell-based to cell-side-based on basal side
  ev = evalUtils.constructDOFCellToSideEvaluator(stokes_dof_names[0],basalSideName,"Node Vector",cellType, "basal_velocity");
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict hydraulic potential from cell-based to cell-side-based on basal side
  ev = evalUtils.constructDOFCellToSideEvaluator(hydro_dof_names[0],basalSideName,"Node Scalar", cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  if (has_h_equation)
  {
    //---- Restrict water thickness from cell-based to cell-side-based on basal side
    //TODO: need to write GatherSolutionSide evaluator
    ev = evalUtils.constructDOFCellToSideEvaluator(hydro_dof_names[1],basalSideName,"Node Scalar", cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    // Interpolate water thickness
    ev = evalUtils.constructDOFInterpolationSideEvaluator(hydro_dof_names[1], basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    if (unsteady)
    {
      // Interpolate water thickness time derivative
      ev = evalUtils.constructDOFInterpolationSideEvaluator(hydro_dof_names_dot[0], basalSideName);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }
  else
  {
    // If only potential equation is solved, the water_thickness MUST be loaded/gathered
    TEUCHOS_TEST_FOR_EXCEPTION (ss_inputs_found[basalSideName].find(water_thickness_name)==ss_inputs_found[basalSideName].end(),
                                std::logic_error, "Error! You did not specify the '" << water_thickness_name << "' requirement in the basal mesh.\n");

    // Interpolate water thickness (no need to restrict it to the side, cause we loaded it as a side state already)
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(water_thickness_name, basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  //---- Compute side basis functions
  ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict ice velocity from cell-based to cell-side-based and interpolate on quad points
  ev = evalUtils.constructDOFVecInterpolationSideEvaluator("basal_velocity", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate ice thickness on QP on side
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("ice_thickness", basalSideName);
  fm0.template registerEvaluator<EvalT>(ev);

  //---- Interpolate surface height on QP on side
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("surface_height", basalSideName);
  fm0.template registerEvaluator<EvalT>(ev);

  //---- Interpolate hydraulic potential gradient
  ev = evalUtils.constructDOFGradInterpolationSideEvaluator(hydraulic_potential_name, basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate surface water input
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("surface_water_input", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict flow factor A from cell-based to cell-side-based (may be needed by BasalFrictionCoefficient)
  ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("Flow Factor A",basalSideName,"Cell Scalar",cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate geothermal flux
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("geothermal_flux", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Extend hydrology potential equation residual from cell-side-based to cell-based
  // NOTE: if we had something like a ScatterSideSetResidual evaluator, this would not be needed
  ev = evalUtils.constructDOFSideToCellEvaluator(hydro_resid_names[0],basalSideName,"Node Scalar", cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Extend basal_gravitational_water_potential from cell-side-based to cell-based
  // NOTE: if we had something like a ScatterSideSetScalarNodalParameter evaluator, this would not be needed
  ev = evalUtils.constructDOFSideToCellEvaluator("basal_gravitational_water_potential",basalSideName,"Node Scalar", cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  if (has_h_equation)
  {
    //---- Extend hydrology thickness equation residual from cell-side-based to cell-based
    ev = evalUtils.constructDOFSideToCellEvaluator(hydro_resid_names[1],basalSideName,"Node Scalar", cellType);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // -------------------- Special evaluators for surface side handling ----------------- //

  if (surfaceSideName!="INVALID")
  {
    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",surfaceSideName,"Vertex Vector",cellType,"Coord Vec " + surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, surfaceSideBasis, surfaceCubature, surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side and interpolate on quad points
    ev = evalUtils.constructDOFCellToSideEvaluator(ice_velocity_name,surfaceSideName,"Node Vector", cellType,"surface_velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity (the solution) on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("surface_velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity rms on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity_RMS", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // -------------------------------- FELIX evaluators ------------------------- //

  // --- FO Stokes Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Resid"));

  //Input
  p->set<std::string>("Weighted BF Variable Name", "wBF");
  p->set<std::string>("Weighted Gradient BF Variable Name", "wGrad BF");
  p->set<std::string>("Velocity QP Variable Name", stokes_dof_names[0]);
  p->set<std::string>("Velocity Gradient QP Variable Name", stokes_dof_names[0] + " Gradient");
  p->set<std::string>("Body Force Variable Name", "Body Force");
  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
  p->set<std::string>("Coordinate Vector Name", "Coord Vec");
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));
  p->set<std::string>("Basal Residual Variable Name", "Basal Residual");
  p->set<bool>("Needs Basal Residual", true);

  //Output
  p->set<std::string>("Residual Variable Name", stokes_resid_names[0]);

  ev = Teuchos::rcp(new FELIX::StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Basal Stokes Residual --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Basal Resid"));

  //Input
  p->set<std::string>("BF Side Name", "BF "+basalSideName);
  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
  p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "beta");
  p->set<std::string>("Velocity Side QP Variable Name", "basal_velocity");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

  //Output
  p->set<std::string>("Basal Residual Variable Name", "Basal Residual");

  ev = Teuchos::rcp(new FELIX::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Sliding velocity calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name","basal_velocity");
  p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name","sliding_velocity");

  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Effective pressure calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure"));

  // Input
  p->set<std::string>("Ice Thickness Variable Name","ice_thickness");
  p->set<std::string>("Surface Height Variable Name","surface_height");
  p->set<std::string>("Hydraulic Potential Variable Name",hydro_dof_names[0]);
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

  // Output
  p->set<std::string>("Effective Pressure Variable Name","effective_pressure");

  ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  // --------- Flow Factor A --------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Flow Factor A"));

  // Input
  p->set<std::string>("Flow Rate Type",params->sublist("FELIX Basal Friction Coefficient").get<std::string>("Flow Rate Type","Uniform"));

  // Output
  p->set<std::string>("Flow Factor A Variable Name","Flow Factor A");

  ev = Teuchos::rcp(new FELIX::FlowFactorA<EvalT,PHAL::AlbanyTraits, false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity Variable Name", "sliding_velocity");
  p->set<std::string>("BF Variable Name", "BF "+basalSideName);
  p->set<std::string>("Effective Pressure Variable Name", "effective_pressure");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<std::string>("Flow Factor A Variable Name", "Flow Factor A");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", "beta");

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,true,true,false>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient at nodes (for output in the mesh) ---//
  p->set<bool>("Nodal",true);
  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,true,true,false>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter: Uniform Flow Factor A ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

  param_name = "Constant Flow Factor A";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::FlowFactorA>> ptr_A;
  ptr_A = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::FlowFactorA>(*p,dl));
  ptr_A->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_A);

  //--- Shared Parameter for basal friction coefficient: lambda ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

  param_name = "Bed Roughness";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>> ptr_lambda;
  ptr_lambda = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>(*p,dl));
  ptr_lambda->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_lambda);

  //--- Shared Parameter for basal friction coefficient: mu ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

  param_name = "Coulomb Friction Coefficient";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>> ptr_mu;
  ptr_mu = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>(*p,dl));
  ptr_mu->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_mu);

  //--- Shared Parameter for basal friction coefficient: power ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

  param_name = "Power Exponent";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>> ptr_power;
  ptr_power = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>(*p,dl));
  ptr_power->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_power);

  //--- Sliding velocity at nodes calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name","basal_velocity");
  p->set<std::string>("Field Layout","Cell Side Node Vector");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name","sliding_velocity");

  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX viscosity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string>("Velocity QP Variable Name", stokes_dof_names[0]);
  p->set<std::string>("Velocity Gradient QP Variable Name", stokes_dof_names[0] + " Gradient");
  p->set<std::string>("Temperature Variable Name", "temperature");
  p->set<std::string>("Flow Factor Variable Name", "flow_factor");
  p->set<std::string>("Homotopy Parameter Name", FELIX::ParamEnum::HomotopyParam_name);
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Viscosity"));

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

  ev = Teuchos::rcp(new FELIX::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT, typename EvalT::ParamScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

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


  //--- Body Force ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Body Force"));

  //Input
  p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");
  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string>("Surface Height Gradient Name", "surface_height Gradient");
  p->set<std::string>("Surface Height Name", "surface_height");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string>("Body Force Variable Name", "Body Force");

  ev = Teuchos::rcp(new FELIX::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // =================================== FELIX Hydrology ===================================== //

  // ------- Hydrology Basal Gravitational Potential -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Basal Gravitational Water Potential"));

  //Input
  p->set<std::string> ("Surface Height Variable Name","surface_height");
  p->set<std::string> ("Ice Thickness Variable Name","ice_thickness");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<bool> ("Is Stokes", true);

  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Basal Gravitational Water Potential Variable Name","basal_gravitational_water_potential");

  ev = Teuchos::rcp(new FELIX::BasalGravitationalWaterPotential<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Water Discharge -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Water Discharge"));

  //Input
  p->set<std::string> ("Water Thickness Variable Name","water_thickness");
  p->set<std::string> ("Hydraulic Potential Gradient Variable Name", hydro_dof_names[0] + " Gradient");
  p->set<std::string> ("Regularization Parameter Name","Regularization");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Water Discharge Variable Name","Water Discharge");

  if (has_h_equation)
    ev = Teuchos::rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
  else
    ev = Teuchos::rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  //Input
  p->set<std::string> ("Geothermal Heat Source Variable Name","geothermal_flux");
  p->set<std::string> ("Sliding Velocity Variable Name","sliding_velocity");
  p->set<std::string> ("Basal Friction Coefficient Variable Name","beta");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Melting Rate Variable Name","Melting Rate");

  ev = Teuchos::rcp(new FELIX::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  if (params->sublist("FELIX Hydrology").get<bool>("Thickness Equation Nodal", false) ||
      params->sublist("FELIX Hydrology").get<bool>("Lump Mass In Potential Equation", false)) {
    // We need the melting rate in the nodes
    p->set<bool>("Nodal", true);

    ev = Teuchos::rcp(new FELIX::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // ------- Hydrology Residual Potential Eqn-------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Potential Eqn"));

  //Input
  p->set<std::string> ("BF Name", "BF " + basalSideName);
  p->set<std::string> ("Metric Name", "Metric " + basalSideName);
  p->set<std::string> ("Gradient BF Name", "Grad BF " + basalSideName);
  p->set<std::string> ("Weighted Measure Name", "Weighted Measure " + basalSideName);
  p->set<std::string> ("Basal Gravitational Water Potential Variable Name","basal_gravitational_water_potential");
  p->set<std::string> ("Effective Pressure Variable Name", "effective_pressure");
  p->set<std::string> ("Flow Factor A Variable Name","Flow Factor A");
  p->set<std::string> ("Hydraulic Potential Variable Name",hydro_dof_names[0]);
  p->set<std::string> ("Melting Rate Variable Name","Melting Rate");
  p->set<std::string> ("Sliding Velocity Variable Name","sliding_velocity");
  p->set<std::string> ("Surface Water Input Variable Name","surface_water_input");
  p->set<std::string> ("Water Discharge Variable Name", "Water Discharge");
  p->set<std::string> ("Water Thickness Variable Name", water_thickness_name);

  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology Parameters",&params->sublist("FELIX Hydrology"));

  //Output
  p->set<std::string> ("Potential Eqn Residual Name",hydro_resid_names[0]);

  if (has_h_equation)
    ev = Teuchos::rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,true,true,false>(*p,dl_basal));
  else
    ev = Teuchos::rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,false,true,false>(*p,dl_basal));

  fm0.template registerEvaluator<EvalT>(ev);

  if (has_h_equation)
  {
    // ------- Hydrology Residual Thickness Eqn -------- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Thickness Eqn"));

    //Input
    p->set<std::string> ("BF Name", "BF " + basalSideName);
    p->set<std::string> ("Weighted Measure Name", "Weighted Measure " + basalSideName);
    p->set<std::string> ("Water Thickness Variable Name",water_thickness_name);
    p->set<std::string> ("Water Thickness Dot Variable Name",water_thickness_dot_name);
    p->set<std::string> ("Effective Pressure Variable Name","effective_pressure");
    p->set<std::string> ("Melting Rate Variable Name","Melting Rate");
    p->set<std::string> ("Sliding Velocity Variable Name","sliding_velocity");
    p->set<std::string> ("Flow Factor A Variable Name","Flow Factor A");
    p->set<std::string> ("Side Set Name", basalSideName);
    p->set<bool> ("Unsteady", unsteady);
    p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
    p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

    //Output
    p->set<std::string> ("Thickness Eqn Residual Name", hydro_resid_names[1]);

    ev = Teuchos::rcp(new FELIX::HydrologyResidualThicknessEqn<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of stokes residual
    PHX::Tag<typename EvalT::ScalarT> stokes_res_tag("Scatter Stokes", dl->dummy);
    fm0.requireField<EvalT>(stokes_res_tag);

    // Require scattering of hydrology residual
    PHX::Tag<typename EvalT::ScalarT> hydro_res_tag("Scatter Hydrology", dl->dummy);
    fm0.requireField<EvalT>(hydro_res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
    Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<std::string>("Surface Velocity Side QP Variable Name","surface_velocity");
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","observed_surface_velocity");
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","observed_surface_velocity_RMS");
    paramList->set<std::string>("BF Surface Name","BF " + surfaceSideName);
    paramList->set<std::string>("Weighted Measure Surface Name","Weighted Measure " + surfaceSideName);
    paramList->set<std::string>("Surface Side Name", surfaceSideName);
    // The regularization with the gradient of beta is available only for GIVEN_FIELD or GIVEN_CONSTANT, which do not apply here
    // We could simply not set the following names, but this way we should get some debugging help in case one mistakenly activates regularization
    paramList->set<std::string>("Basal Friction Coefficient Gradient Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
    paramList->set<std::string>("Basal Side Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
    paramList->set<std::string>("Inverse Metric Basal Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
    paramList->set<std::string>("Weighted Measure Basal Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

#endif // FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP
