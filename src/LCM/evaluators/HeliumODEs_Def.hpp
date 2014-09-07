//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <cmath>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Intrepid_MiniTensor.h>

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
HeliumODEs<EvalT, Traits>::
HeliumODEs(Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
      totalConcentration_(p.get < std::string > ("Total Concentration Name"),
          dl->qp_scalar),
      delta_time_(p.get < std::string > ("Delta Time Name"),
          dl->workset_scalar),
      diffusionCoefficient_(p.get < std::string > ("Diffusion Coefficient Name"),
                        dl->qp_scalar),
      HeConcentration_(p.get < std::string > ("Helium Concentration Name"),
              dl->qp_scalar),
      totalBubbleDensity_(p.get < std::string > ("Total Bubble Density Name"),
          dl->qp_scalar),
      bubbleVolumeFraction_(
          p.get < std::string > ("Bubble Volume Fraction Name"),
          dl->qp_scalar)
{
  // get the material parameter list
  Teuchos::ParameterList* mat_params =
      p.get<Teuchos::ParameterList*>("Material Parameters");

  avogadrosNum_ = mat_params->get<RealType>("Avogadro's Number");
  omega_ = mat_params->get<RealType>("Molar Volume");
  TDecayConstant_ = mat_params->get<RealType>("Tritium Decay Constant");
  HeRadius_ = mat_params->get<RealType>("Helium Radius");
  eta_ = mat_params->get<RealType>("Atoms Per Cluster");

  // add dependent fields
  this->addDependentField(totalConcentration_);
  this->addDependentField(diffusionCoefficient_);
  this->addDependentField(delta_time_);

  // add evaluated fields
  this->addEvaluatedField(HeConcentration_);
  this->addEvaluatedField(totalBubbleDensity_);
  this->addEvaluatedField(bubbleVolumeFraction_);

  this->setName(
      "Helium ODEs" + PHX::TypeString < EvalT > ::value);
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  num_pts_ = dims[1];
  num_dims_ = dims[2];

  totalConcentration_name_ = p.get<std::string>("Total Concentration Name")+"_old";
  HeConcentration_name_ = p.get<std::string>("Helium Concentration Name")+"_old";
  totalBubbleDensity_name_ = p.get<std::string>("Total Bubble Density Name")+"_old";
  bubbleVolumeFraction_name_ = p.get<std::string>("Bubble Volume Fraction Name")+"_old";

}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void HeliumODEs<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(totalConcentration_, fm);
  this->utils.setFieldData(delta_time_, fm);
  this->utils.setFieldData(diffusionCoefficient_, fm);
  this->utils.setFieldData(HeConcentration_, fm);
  this->utils.setFieldData(totalBubbleDensity_, fm);
  this->utils.setFieldData(bubbleVolumeFraction_, fm);
  
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void HeliumODEs<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

 // Declaring temporary variables for time integration at (cell,pt) following Schaldach & Wolfer
   ScalarT dt, dtExplicit;
   ScalarT N1old, Nbold, Sbold, N1new, Nbnew, Sbnew;
   ScalarT N1exp, Nbexp, Sbexp;
   ScalarT D, Gold, Gnew;
 // Declaring tangent, residual, norms, and increment for N-R
   Intrepid::Tensor<ScalarT> tangent(3);
   Intrepid::Vector<ScalarT> residual(3);
   ScalarT normResidual, normResidualGoal;
   Intrepid::Vector<ScalarT> increment(3);
	
 // constants for computations
  const double pi = acos(-1.0);  
  const double onethrd = 1.0/3.0;
  const double twothrd = 2.0/3.0;
  const double tolerance = 1.0e-12;
  const int explicitSubIncrements = 5;
// const int maxIterations = 20; //FIXME: Include a maximum number of iterations
  
  // state old
  Albany::MDArray totalConcentration_old = (*workset.stateArrayPtr)[totalConcentration_name_]; 
  Albany::MDArray HeConcentration_old = (*workset.stateArrayPtr)[HeConcentration_name_]; 
  Albany::MDArray totalBubbleDensity_old = (*workset.stateArrayPtr)[totalBubbleDensity_name_]; 
  Albany::MDArray bubbleVolumeFraction_old = (*workset.stateArrayPtr)[bubbleVolumeFraction_name_]; 
 
  // state new
  //   HeConcentration_ - He concentration at t + deltat
  //   totalBubbleDensity - total bubble density at t + delta t
  //   bubbleVolumeFraction - bubble volume fraction at t + delta t
  //   totalConcentration_ - total concentration of tritium at t + delta t
  
  // fields required for computation
  //   diffusionCoefficient_ - current diffusivity (varies with temperature)
  //   delta_time_ - time step 
  
  // input properties
  //   avogadrosNum_ - Avogadro's Number
  //   omega_ - molar volume
  //   TDecayConstant_ - radioactive decay constant for tritium
  //   HeRadius_ - radius of He atom
  //   eta_ - atoms per cluster (not variable)
  
  // time step
  dt = delta_time_(0);
 
  // loop over cells and points for implicit time integration
  
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
	  
	  for (std::size_t pt = 0; pt < num_pts_; ++pt) {
		  
		  // temporary variables
		  N1old = HeConcentration_old(cell,pt);
		  Nbold = totalBubbleDensity_old(cell,pt);
		  Sbold = bubbleVolumeFraction_old(cell,pt);
		  N1new = N1old;
		  Nbnew = Nbold;
		  Sbnew = Sbold;
		  D = diffusionCoefficient_(cell,pt);
		  
		  // determine if any tritium exists - note that concentration is in mol (not atoms)
		  // if no tritium exists, no need to solve the ODEs
		  if (totalConcentration_(cell,pt) > tolerance) {
			  
			  // source terms for helium bubble generation
			  Gold = avogadrosNum_*TDecayConstant_*totalConcentration_old(cell,pt);
			  Gnew = avogadrosNum_*TDecayConstant_*totalConcentration_(cell,pt);
			  
			  // check if old bubble density is small
			  // if small, use an explict guess to avoid issues with 1/Nbnew and 1/Sbnew in tangent
			  
			  if (Nbold < tolerance) {
				  
				  // explicit time integration for predictor
				  // Note that two or more steps are required to obtain a finite Nbnew if the
				  // totalConcentration_old is zero.
				  dtExplicit = dt/explicitSubIncrements;
				  N1exp = N1old;
				  Nbexp = Nbold;
				  Sbexp = Sbold;
				  
				  for (int subIncrement = 0; subIncrement < explicitSubIncrements; subIncrement++) {
					  N1new = N1exp + dtExplicit*(Gold - 32.*pi*HeRadius_*D*N1exp*N1exp - 
							  4.0*pi*D*N1exp*pow(3.0/4.0/pi,onethrd)*pow(Sbexp,onethrd)*
							  pow(Nbexp,twothrd));
					  Nbnew = Nbexp + dtExplicit*(16.0*pi*HeRadius_*D*N1exp*N1exp);
					  Sbnew = Sbexp + omega_/eta_*dtExplicit*(32.*pi*HeRadius_*D*N1exp*N1exp + 
							  4.0*pi*D*N1exp*pow(3.0/4.0/pi,onethrd)*pow(Sbexp,onethrd)*
							  pow(Nbexp,twothrd));
					  N1exp = N1new;
					  Nbexp = Nbnew;
					  Sbexp = Sbnew;
				  }   
			  }
			  
			  // calculate initial residual for a relative tolerance
			  residual(0) = N1new - N1old - dt*(Gnew - 32.*pi*HeRadius_*D*N1new*N1new -
					  4.0*pi*D*N1new*pow(3.0/4.0/pi,onethrd)*pow(Sbnew,onethrd)*pow(Nbnew,twothrd));
			  residual(1) = Nbnew - Nbold - dt*(16.0*pi*HeRadius_*D*N1new*N1new);
			  residual(2) = Sbnew - Sbold - omega_/eta_*dt*(32.*pi*HeRadius_*D*N1new*N1new + 
					  4.0*pi*D*N1new*pow(3.0/4.0/pi,onethrd)*pow(Sbnew,onethrd)*pow(Nbnew,twothrd));
		      normResidual = Intrepid::norm(residual);
		      normResidualGoal = tolerance*normResidual;
			  
		      // N-R loop for implicit time integration
		      while (normResidual > normResidualGoal) {
		    	  
		    	  // calculate tangent
		    	  tangent(0,0) = 1.0 + 2.0*dt*D*(32.0*N1new*pi*HeRadius_ + pow(6.0,onethrd)*
		    			  pow(Nbnew,twothrd)*pow(pi,twothrd)*pow(Sbnew,onethrd));
		    	  tangent(0,1) = 4.0*pow(2.0,onethrd)*dt*D*N1new*pow(pi,twothrd)*
		    			  pow(Sbnew,onethrd)/pow(3.0,twothrd)/pow(Nbnew,onethrd);
		    	  tangent(0,2) = 2.0*pow(2.0,onethrd)*dt*D*N1new*pow(Nbnew,twothrd)*
		    			  pow(pi/3.0,twothrd)/pow(Sbnew,twothrd);
		    	  tangent(1,0) = -32.0*dt*D*N1new*pi*HeRadius_; 
		    	  tangent(1,1) = 1.0;
		    	  tangent(1,2) = 0.0;
		    	  tangent(2,0) = -2.0*dt*D*omega_*(32*N1new*pi*HeRadius_ + pow(6.0,onethrd)*
		    			  pow(Nbnew,twothrd)*pow(pi,onethrd)*pow(Sbnew,onethrd))/eta_;
		    	  tangent(2,1) = -4.0*pow(2.0,onethrd)*pow(2,onethrd)*dt*D*N1new*omega_*
		    			  pow(pi,twothrd)*pow(Sbnew,onethrd)/pow(3,twothrd)/eta_/pow(Nbnew,onethrd);
		    	  tangent(2,2) = 1.0 - 2.0*pow(2.0,onethrd)*dt*D*N1new*pow(Nbnew,twothrd)*
		    			  omega_*pow(pi/3.0,twothrd)/eta_/pow(Sbnew,twothrd);
		    	  
		    	  // find increment
		    	  increment = -Intrepid::inverse(tangent)*residual;
		    	  
		    	  // update quantities
		    	  N1new = N1new + increment(0);
		    	  Nbnew = Nbnew + increment(1);
		    	  Sbnew = Sbnew + increment(2);
		    	  
		    	  // find new residual and norm
		    	  residual(0) = N1new - N1old - dt*(Gnew - 32.*pi*HeRadius_*D*N1new*N1new - 
		    			  4.0*pi*D*N1new*pow(3.0/4.0/pi,onethrd)*pow(Sbnew,onethrd)*pow(Nbnew,twothrd));
		    	  residual(1) = Nbnew - Nbold - dt*(16.0*pi*HeRadius_*D*N1new*N1new);
		    	  residual(2) = Sbnew - Sbold - omega_/eta_*dt*(32.*pi*HeRadius_*D*N1new*N1new + 
		    			  4.0*pi*D*N1new*pow(3.0/4.0/pi,onethrd)*pow(Sbnew,onethrd)*pow(Nbnew,twothrd));
		    	  normResidual = Intrepid::norm(residual);
		      } 
		  }
		  
		  // Update global fields
		  HeConcentration_(cell,pt) = N1new;
		  totalBubbleDensity_(cell,pt) = Nbnew;
		  bubbleVolumeFraction_(cell,pt) = Sbnew;
	  }
  }  
  
}
//------------------------------------------------------------------------------
}

