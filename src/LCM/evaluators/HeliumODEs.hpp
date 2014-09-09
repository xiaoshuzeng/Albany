//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(HeliumODEs_hpp)
#define HeliumODEs_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief
  ///
  /// This evaluator integrates coupled ODEs for the
  ///   1. He concentration
  ///   2. Total bubble density
  ///   3. Bubble volume fraction
  /// We employ implicit integration (backward Euler)
  ///
  template<typename EvalT, typename Traits>
  class HeliumODEs : public PHX::EvaluatorWithBaseImpl<Traits>,
                                       public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    HeliumODEs(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Input: totalConcentration - addition of lattice and trapped
    ///        concentration
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> totalConcentration_;

    ///
    /// Input: time step
    ///
    PHX::MDField<ScalarT,Dummy> delta_time_;

    ///
    /// Input: temperature dependent diffusion coefficient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> diffusionCoefficient_;

    ///
    /// Input: spatial dimension and number of integration points
    ///
    std::size_t num_dims_;
    std::size_t num_pts_;

    ///
    /// Output
    /// (1) He concentration
    /// (2) Total bubble density
    /// (3) Bubble volume fracture
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> HeConcentration_;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> totalBubbleDensity_;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> bubbleVolumeFraction_;

    ///
    /// Constants
    ///
    RealType avogadrosNum_, omega_, TDecayConstant_, HeRadius_, eta_;

    /// 
    /// Scalar names for obtaining state old
    ///

    std::string totalConcentration_name_;
    std::string HeConcentration_name_;
    std::string totalBubbleDensity_name_;
    std::string bubbleVolumeFraction_name_;

  };
}

#endif
