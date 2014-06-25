//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef HMCPROBLEM_HPP
#define HMCPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"


/*\begin{text}
This source has been annotated with latex comments.  Use the eqcc script to compile into a summary pdf.  The source is best viewed using folding in vim (i.e., \begin{verbatim} :g/\\begin{text}/foldc \end{verbatim})
\end{text}*/


namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class HMCProblem : public Albany::AbstractProblem {
  public:
  
    //! Default constructor
    HMCProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_);

    //! Destructor
    virtual ~HMCProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
			    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
			    ) const;

  private:

    //! Private to prohibit copying
    HMCProblem(const HMCProblem&);
    
    //! Private to prohibit copying
    HMCProblem& operator=(const HMCProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> 
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);


  protected:

    //! Boundary conditions on source term
    bool haveSource;
    int numDim;

    std::string matModel; 
    Teuchos::RCP<Albany::Layouts> dl;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState;
  };

}

#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "PHAL_Source.hpp"
#include "Strain.hpp"
#include "DefGrad.hpp"
#include "Stress.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"

#include "Time.hpp"
#include "CapExplicit.hpp"
#include "CapImplicit.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::HMCProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using PHAL::AlbanyTraits;

  // get the name of the current element block
   std::string elementBlockName = meshSpecs.ebName;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
     intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid::DefaultCubatureFactory<RealType> cubFactory;
   RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << std::endl;


   // Construct standard FEM evaluators with standard field names                              
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in Mechanics problems assume vecDim = numDim");
   Albany::HMCEvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   bool supportsTransient=true;

   // Define Field Names
   //   get number of additional scales
   const int nTotalScales = 2;
   const int nAddedScales = 1;


   Teuchos::ArrayRCP<std::string> dof_names(nscales);
     dof_names[0] = "Displacement";
     for(int i=0;i<nAddedScales;i++){
       stringstream dofname;
       dofname << "Microstrain " << i;
       dof_names[i] = dofname.str();
     }
   Teuchos::ArrayRCP<std::string> dof_names_dotdot(nscales);
   Teuchos::ArrayRCP<std::string> resid_names(nscales);
   if (supportsTransient)
     for(int i=0;i<nTotalScales;i++){
       dof_names_dotdot[i] = dof_names[i]+"_dotdot";
       resid_names[i] = dof_names[i]+" Residual";
     }

// 1.1 Gather Solution (displacements and accelerations)
/*\begin{text} 
   New evaluator:  Gather solution data from solver data structures to grid based structures.  Note that accelerations are added as an evaluated field if appropriate.\\
   \textbf{Dependent Fields:} \\
     None. \\
  \textbf{Evaluated Fields:} \\
  \begin{tabular}{l l l l}
     $u_{iI}$ & Nodal displacements & ("Variable Name", "Displacement") & dims(cell,nNodes,vecDim) \\
     $a_{iI}$ & Nodal accelerations & ("Variable Name", "Displacement\_dotdot") & dims(cell,nNodes,vecDim)
  \end{tabular} \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_GatherSolution\_Def.hpp
\end{text}*/
   if (supportsTransient) fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_withAcceleration(true, dof_names, Teuchos::null, dof_names_dotdot));
   else fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));

// 1.2  Gather Coordinates
/*\begin{text}
   New evaluator: Gather coordinate data from solver data structures to grid based structures.
   \textbf{Dependent Fields:} \\
     None. \\
 
  \textbf{Evaluated Fields:} \\
   \begin{tabular}{l l l l}
     $x_{iI}$ & Nodal coordinates & ("Coordinate Vector Name", "Coord Vec") & dims(cell,nNodes,vecDim) \\
   \end{tabular} \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_GatherCoordinateVector\_Def.hpp
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

// 2.1  Compute gradient matrix and weighted basis function values in current coordinates
/*\begin{text} 
    Register new evaluator.\\
    \textbf{Dependent Fields:}\\
    \begin{tabular}{l l l l}
    $x_{iI}$ & Nodal coordinates & ("Cordinate Vector Name", "Coord Vec") & dims(cell,nNodes,vecDim) \\
    \end{tabular} \\

    \textbf{Evaluated Fields:} \\
    \begin{tabular}{l l l l}
    $det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
    Weighted measure & ("Weights Name", "Weights") & dims(cell,nQPs)\\
    %
    $det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right)$ &
    Jacobian determinant & ("Jacobian Det Name", Jacobian Det") & dims(cell,nQPs) \\
    %
    $N_I(\mathbf{x}_p)$ &
    Basis function values & ("BF Name", "BF") & dims(cell,nNodes,nQPs)\\
    %
    $N_I(\mathbf{x}_p)\ det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
    Weighted ... & ("Weighted BF Name", "wBF") & dims(cell,nNode,nQPs)\\
    %
    $\frac{\partial N_I (x_p)}{\partial \xi_k} J^{-1}_{kj}$ &
    Gradient matrix wrt physical frame & ("Gradient BF Name", "Gradient BF") & dims(cell,nNodes,nQPs,spcDim)\\
    %
    $\frac{\partial N_I (x_p)}{\partial \xi_k} J^{-1}_{kj} det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
    Weighted ... & ("Weighted Gradient BF Name", "Weighted Gradient BF") & dims(cell,nNodes,nQPs,spcDim)\\
    \end{tabular} \\

    For implementation see: \\ 
      problems/Albany\_EvaluatorUtils\_Def.hpp \\
      evaluators/PHAL\_ComputeBasisFunctions\_Def.hpp
 \end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

// 3.1  Project displacements to Gauss points
/*\begin{text} 
   New evaluator:
  \begin{align*}
     u_i(\xi_p)&=N_I(\xi_p) u_{iI}\\
     (c,p,i)&=(c,I,p)*(c,I,i)
  \end{align*}
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $u_{iI}$ & Nodal Displacements & ("Variable Name", "Displacements") & dims(cell,nNodes,vecDim) \\
     $N_I(\xi_p)$ & Basis Functions & ("BF Name", "BF") & dims(cell,nNodes,nQPs) \\
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
  \begin{tabular}{l l l l}
     $u_i(\xi_p)$ & Displacements at quadrature points & ("Variable Name", "Displacements") & dims(cell,nQPs,vecDim)
  \end{tabular} \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_DOFVecInterpolation\_Def.hpp
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

// 3.2  Project accelerations to Gauss points
/*\begin{text} 
   \newpage
   New evaluator:
  \begin{align*}
     a_i(\xi_p)&=N_I(\xi_p) a_{iI}\\
     (c,p,i)&=(c,I,p)*(c,I,i)
  \end{align*}
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $a_{iI}$ & Nodal Acceleration & ("Variable Name", "Displacement\_dotdot") & dims(cell,nNodes,vecDim)\\
     $N_I(\xi_p)$ & Basis Functions & ("BF Name", "BF") & dims(cell,nNodes,nQPs) \\
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
  \begin{tabular}{l l l l}
     $a_i(\xi_p)$ & Acceleration at quadrature points & ("Variable Name", "Dsplacement\_dotdot") & dims(cell,nQPs,vecDim)
  \end{tabular} \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_DOFVecInterpolation\_Def.hpp
\end{text}*/
   if(supportsTransient) fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));

// 3.3  Project nodal coordinates to Gauss points
/*\begin{text}
   New evaluator: Compute Gauss point locations from nodal locations.
  \begin{align*}
     x_{pi} &= N_{I}(\xi_p) x_{iI}\\
   (c,p,i) &= (c,I,p)*(c,I,i)
  \end{align*}
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $x_{iI}$ & Nodal coordinates & ("Coordinate Vector Name", "Coord Vec") & dims(cell,nNodes,vecDim) \\
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
   \begin{tabular}{l l l l}
     $x_{pi}$ & Gauss point coordinates & ("Coordinate Vector Name", "Coord Vec") & dims(cell,nQPs,vecDim) \\
   \end{tabular} \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_MapToPhysicalFrame\_Def.hpp
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

// 3.4  Compute displacement gradient
/*\begin{text} 
   New evaluator:
  \begin{align*}
     \left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p} &= \partial_j N_{I}(\xi_p) u_{iI}\\
   (c,p,i,j) &= (c,I,p,j)*(c,I,i)
  \end{align*}
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $u_{iI}$ & Nodal Displacement & ("Variable Name", "Displacement") & dims(cell,nNodes,vecDim) \\
     $B_I(\xi_p)$ & Gradient of Basis Functions & ("Gradient BF Name", "Grad BF") & dims(cell,nNodes,nQPs,vecDim) \\
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
  \begin{tabular}{l l l l}
     $\left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p}$ & Gradient of node vector & ("Gradient Variable Name", "Displacement Gradient") & dims(cell,nQPs,vecDim,spcDim)
  \end{tabular} \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_DOFVecGradInterpolation\_Def.hpp
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));
 
  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

// 4.1  Compute strain
/*\begin{text} 
   New evaluator:
  \begin{align*}
     \epsilon^p_{ij} &=  
               \frac{1}{2}\left(\left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p}
                               +\left.\frac{\partial u_j}{\partial x_i}\right|_{\xi_p} \right)\\
   (c,p,i,j) &= ((c,p,i,j)+(c,p,j,i)/2.0)
  \end{align*}
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $\left.\frac{\partial u_i}{\partial x_j}\right|_{\xi_p}$ & Gradient of node vector & ("Gradient Variable Name", "Displacement Gradient") & dims(cell,nQPs,vecDim,spcDim)
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
  \begin{tabular}{l l l l}
     $\epsilon^p_{ij}$ & Infinitesimal strain & ("Strain Name", "Strain") & dims(cell,nQPs,vecDim,spcDim) \\
  \end{tabular} \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_DOFVecGradInterpolation\_Def.hpp
\end{text}*/
  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList("Strain"));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Output
    p->set<std::string>("Strain Name", "Strain");

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    if(matModel == "CapExplicit" || matModel == "GursonSD" || matModel == "CapImplicit"){
      p = stateMgr.registerStateVariable("Strain", dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0, true);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    	fm0.template registerEvaluator<EvalT>(ev);
    }
  }

// 5.1 Compute stress
/*\begin{text} 
   New evaluator:
  \begin{align*}
   \sigma^p_{ij} &= \lambda \epsilon^p_{kk} \delta_{ij}+2\mu \epsilon^p_{ij}\\
   \lambda&=\frac{\nu E}{(1+\nu)(1-2\nu)}\\
   \mu&=\frac{E}{2(1+\nu)}
  \end{align*}
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $\epsilon^p_{ij}$ & Infinitesimal strain & ("Strain Name", "Strain") & dims(cell,nQPs,vecDim,spcDim) \\
     $E$ & Elastic modulus & ("Elastic Modulus Name", "Elastic Modulus") & dims(cell,nQPs)\\
     $\nu$ & Poisson's ratio & ("Poissons Ratio Name", "Poissons Ratio") & dims(cell,nQPs)\\
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
  \begin{tabular}{l l l l}
     $\sigma^p_{ij}$ & Stress & ("Stress Name", "Stress") & dims(cell,nQPs,vecDim,spcDim) \\
  \end{tabular} \\

  For implementation see: \\ 
    LCM/evaluators/Stress\_Def.hpp \\
\end{text}*/
  { // Linear elasticity stress
      RCP<ParameterList> p = rcp(new ParameterList("Stress"));

      //Input
      p->set<std::string>("Strain Name", "Strain");
      p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

      p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
      p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

      p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");  // dl->qp_scalar also

      //Output
      p->set<std::string>("Stress Name", "Stress"); //dl->qp_tensor also

      ev = rcp(new LCM::Stress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      p = stateMgr.registerStateVariable("Stress",dl->qp_tensor, dl->dummy, elementBlockName, "scalar", 0.0);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
  }

// 6.1 Compute residual (stress divegence + inertia term)
/*\begin{text} 
   New evaluator:
  \begin{align*}
    f_{iI} = \sum_p 
    \frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_k} J^{-1}_{kj} det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p \sigma^p_{ij}
   + \sum_p N_I(\mathbf{x}_p)\ det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p a^p_i
  \end{align*}
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $\sigma^p_{ij}$ & Stress & ("Stress Name", "Stress") & dims(cell,nQPs,vecDim,spcDim) \\
     $\frac{\partial N_I (\mathbf{x}_p)}{\partial \xi_k} J^{-1}_{kj} det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
     Weighted GradBF & ("Weighted Gradient BF Name", "Weighted Gradient BF") & dims(cell,nNodes,nQPs,spcDim)\\
     $a^p_i$ & Acceleration at quadrature points & ("Variable Name", "Dsplacement\_dotdot") & dims(cell,nQPs,vecDim)\\
     $N_I(\mathbf{x}_p)\ det\left(\frac{\partial x_{ip}}{\partial \xi_j}\right) \omega_p$ &
     Weighted BF & ("Weighted BF Name", "wBF") & dims(cell,nNode,nQPs)\\
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
  \begin{tabular}{l l l l}
     $f_{iI}(x_iI)$ & Residual & ("Residual Name", "Residual") & dims(cell,nNodes,spcDim)\\
  \end{tabular} \\

  For implementation see: \\ 
    LCM/evaluators/ElasticityResid\_Def.hpp \\
\end{text}*/
  { // Displacement Resid
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    //Input
    p->set<std::string>("Stress Name", "Stress");
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    // \todo Is the required?
    p->set<std::string>("DefGrad Name", "Deformation Gradient"); //dl->qp_tensor also

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    // extra input for time dependent term
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
    p->set<std::string>("Time Dependent Variable Name", "Displacement_dotdot");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// X.X  Scatter nodal forces
/*\begin{text}
   New evaluator: Scatter the nodal forces (i.e., "Displacement Residual") from the grid based structures to the solver data structures.
   \textbf{Dependent Fields:} \\
   \begin{tabular}{l l l l}
     $u_{iI}$ & Displacement residual & ("Residual Name", "Displacement Residual") & dims(cell,nNodes,vecDim) \\
   \end{tabular} \\
 
  \textbf{Evaluated Fields:} \\
     None. \\

  For implementation see: \\ 
    problems/Albany\_EvaluatorUtils\_Def.hpp \\
    evaluators/PHAL\_ScatterResidual\_Def.hpp
\end{text}*/
   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(true, resid_names));

  { // Time
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Elastic Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Elastic Modulus");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Elastic Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::ElasticModulus<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Poissons Ratio 
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Poissons Ratio");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set< RCP<DataLayout> >("Node Data Layout", dl->node_scalar);
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Poissons Ratio");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::PoissonsRatio<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

// ?
  if (haveSource) { // Source
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       "Error!  Sources not implemented in HMC yet!");

    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, stateMgr);
  }

  return Teuchos::null;
}

#endif // ALBANY_ELASTICITYPROBLEM_HPP
