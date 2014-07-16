//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

//Utility function to split a std::string by a delimiter, so far only used here
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

}


template<typename EvalT, typename Traits>
Aeras::ResponseL2Error<EvalT, Traits>::
ResponseL2Error(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl) :
  sphere_coord("Lat-Long", dl->qp_gradient),
  weighted_measure("Weights", dl->qp_scalar),
  flow_state_field("Flow State", dl->node_vector), 
  BF("BF",dl->node_qp_scalar)
{
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  // Get field type and corresponding layouts
  std::string fieldName = plist->get<std::string>("Field Name", "");

  // coordinate dimensions
  std::vector<PHX::DataLayout::size_type> coord_dims;
  dl->qp_vector->dimensions(coord_dims);
  numQPs = coord_dims[1]; //# quad points
  numDims = coord_dims[2]; //# spatial dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  flow_state_field.fieldTag().dataLayout().dimensions(dims);
  vecDim = dims[2]; //# dofs per node
  numNodes =  dims[1]; //# nodes per element

 
  // User-specified parameters
  std::string ebNameStr = plist->get<std::string>("Element Block Name","");
  if(ebNameStr.length() > 0) split(ebNameStr,',',ebNames);

  // add dependent fields
  this->addDependentField(sphere_coord);
  this->addDependentField(flow_state_field);
  this->addDependentField(weighted_measure);
  this->addDependentField(BF);
  this->setName(fieldName+" Aeras L2 Error"+PHX::TypeString<EvalT>::value);
  
  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = fieldName + " Local Response Aeras L2 Error";
  std::string global_response_name = fieldName + " Global Response Aeras L2 Error";
  int worksetSize = dl->qp_scalar->dimension(0);
  int responseSize = 1; 
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ResponseL2Error<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sphere_coord,fm);
  this->utils.setFieldData(flow_state_field,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(BF,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ResponseL2Error<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ResponseL2Error<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{   
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;

  if( ebNames.size() == 0 || 
      std::find(ebNames.begin(), ebNames.end(), workset.EBName) != ebNames.end() ) {

  Intrepid::FieldContainer<ScalarT> flow_state_field_qp(workset.numCells, numQPs, vecDim); //flow_state_field at quad points
  Intrepid::FieldContainer<ScalarT> flow_state_field_ref_qp(workset.numCells, numQPs, vecDim); //flow_state_field_ref (exact solution) at quad points
  Intrepid::FieldContainer<ScalarT> err_qp(workset.numCells, numQPs, vecDim); //error at quadrature points

  //The following reference solution / parameters are hard-coded for now to those of TC2 
  //TO DO: make this generic, with TC name read in from input file parameter list
  //TO DO: make it possible to evaluate exact solution that has time in it 
  const double myPi = Aeras::ShallowWaterConstants::self().pi;
  const double gravity = Aeras::ShallowWaterConstants::self().gravity;
  const double Omega = 2.0*myPi/(24.*3600.);
  const double a = Aeras::ShallowWaterConstants::self().earthRadius;
  const double u0 = 2.*myPi*a/(12*24*3600.);  // magnitude of wind
  const double h0g = 2.94e04;
  const double alpha = 0; /* must match value in ShallowWaterResidDef
                             don't know how to get data from input into this class and that one. */
  const double cosAlpha = std::cos(alpha);
  const double sinAlpha = std::sin(alpha);
  const double h0     = h0g/gravity;

  //Interpolate flow_state_field from nodes -> quadrature points.  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
        flow_state_field_qp(cell,qp,i) = 0.0;
        flow_state_field_qp(cell,qp,i) = flow_state_field(cell, 0, i) * BF(cell, 0, qp); 
        for (std::size_t node=1; node < numNodes; ++node) {
          flow_state_field_qp(cell,qp,i) += flow_state_field(cell,node,i)*BF(cell,node,qp); 
        }
       }
     }
    }

  //Set reference solution at quadrature points -- right now, hard-coded to TC2, shallow water equations
  static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      MeshScalarT lambda = sphere_coord(cell, qp, 0);//lambda 
      MeshScalarT theta = sphere_coord(cell, qp, 1); //theta
      if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0.0;
      else if (lambda < 0) lambda += 2*myPi;
      const MeshScalarT cosLambda = std::cos(lambda); //cos(lambda)
      const MeshScalarT sinLambda = std::sin(lambda); //sin(lambda)
      const MeshScalarT cosTheta = std::cos(theta); //cos(theta)
      const MeshScalarT sinTheta = std::sin(theta); //sin(theta)
      flow_state_field_ref_qp(cell,qp,0) =  h0 - 1.0/gravity * (a*Omega*u0 + u0*u0/2.0)*(-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha)*
         (-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha); //h
      flow_state_field_ref_qp(cell,qp,1) = u0*(cosTheta*cosAlpha + sinTheta*cosLambda*sinAlpha); //u
      flow_state_field_ref_qp(cell,qp,2) = -u0*(sinLambda*sinAlpha); //v
     }
   }

  //Calculate L2 error at all the quad points 
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t dim=0; dim < vecDim; ++dim) {
        err_qp(cell,qp,dim) = flow_state_field_qp(cell,qp,dim) - flow_state_field_ref_qp(cell,qp,dim); 
      }
      //debug print statements
      /*std::cout << "cell, qp: " << cell << ", " << qp << std::endl;
      std::cout << "error h: " << err_qp(cell,qp,0) << std::endl;   
      std::cout << "h calc, h ref: " << flow_state_field_qp(cell,qp,0) << ", " << flow_state_field_ref_qp(cell,qp,0) << std::endl; 
      std::cout << "error u: " << err_qp(cell,qp,1) << std::endl;  
      std::cout << "u calc, u ref: " << flow_state_field_qp(cell,qp,1) << ", " << flow_state_field_ref_qp(cell,qp,1) << std::endl; 
      std::cout << "error v: " << err_qp(cell,qp,2) << std::endl;  
      */
    }
  }
  
  //Calculate total L2 error
    ScalarT s = 0.0;
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t dim=0; dim < vecDim; ++dim) {
           //L^2 error squared w.r.t. flow_state_field_ref
           s += err_qp(cell,qp,dim)*err_qp(cell,qp,dim);  
           //s += flow_state_field_ref_qp(cell,qp,dim)*flow_state_field_ref_qp(cell,qp,dim); 
	   this->local_response(cell) += s;
	   this->global_response(0) += s;
        
        }
      }
    }
  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ResponseL2Error<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  // Add contributions across processors
  Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
    workset.serializerManager.template getValue<EvalT>();

  //Take square root of global response, which defines error 
  ScalarT err = this->global_response[0]; 
  this->global_response[0] = sqrt(err); 
  Teuchos::reduceAll(
    *workset.comm, *serializer, Teuchos::REDUCE_SUM,
    this->global_response.size(), &this->global_response[0], 
    &this->global_response[0]);

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Aeras::ResponseL2Error<EvalT,Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ResponseL2Error Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Field Type", "", "Type of field (scalar, vector, ...)");
  validPL->set<std::string>(
    "Element Block Name", "", 
    "Name of the element block to use as the integration domain");
  validPL->set<std::string>("Field Name", "", "Field to integrate");
  validPL->set<bool>("Positive Return Only",false);

  validPL->set< Teuchos::Array<int> >("Field Components", Teuchos::Array<int>(),
				      "Field components to scatter");

  return validPL;
}

// **********************************************************************

