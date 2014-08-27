//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_OptimizationProblem.hpp"
#include "Albany_AbstractDiscretization.hpp"

/******************************************************************************/
ATO::OptimizationProblem::
OptimizationProblem( const Teuchos::RCP<Teuchos::ParameterList>& _params,
                     const Teuchos::RCP<ParamLib>& _paramLib,
                     const int _numDim) :
Albany::AbstractProblem(_params, _paramLib, _numDim) {}
/******************************************************************************/


/******************************************************************************/
void
ATO::OptimizationProblem::
ComputeVolume(double& v)
/******************************************************************************/
{
  v = 0.0;

  //JR:  hardwired for element topo

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
    wsElNodeEqID = disc->getWsElNodeEqID();
  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();


  int numWorksets = wsElNodeEqID.size();

  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];

    int numCells = wsElNodeEqID[ws].size();
    int numQPs = cubatures[physIndex]->getNumPoints();
    
    for(int cell=0; cell<numCells; cell++)
      for(int qp=0; qp<numQPs; qp++)
        v += weighted_measure[ws](cell,qp);
  }
}
/******************************************************************************/
void
ATO::OptimizationProblem::
ComputeVolume(double* p, double& v, double* dvdp)
/******************************************************************************/
{
  v = 0.0;

  //JR:  hardwired for element topo

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
    wsElNodeEqID = disc->getWsElNodeEqID();
  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();


  int numWorksets = wsElNodeEqID.size();


  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];

    int numCells = wsElNodeEqID[ws].size();
    int numQPs = cubatures[physIndex]->getNumPoints();
    
    int wsOffset = 0;
    for(int cell=0; cell<numCells; cell++){
      double elVol = 0.0;
      for(int qp=0; qp<numQPs; qp++)
        elVol += weighted_measure[ws](cell,qp);
      v += elVol*p[wsOffset+cell];
    }
    wsOffset += numCells;
  }

  if( dvdp != NULL ){
    for(int ws=0; ws<numWorksets; ws++){
  
      int physIndex = wsPhysIndex[ws];
  
      int numCells = wsElNodeEqID[ws].size();
      int numQPs = wsElNodeEqID[ws][0][0].size();
      
      int wsOffset = 0;
      for(int cell=0; cell<numCells; cell++){
        double elVol = 0.0;
        for(int qp=0; qp<numQPs; qp++)
          elVol += weighted_measure[ws](cell,qp);
        dvdp[wsOffset+cell] = elVol;
      }
      wsOffset += numCells;
    }
  }

}
/******************************************************************************/
void
ATO::OptimizationProblem::
setupTopOpt( Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  _meshSpecs,
             Albany::StateManager& _stateMgr)
/******************************************************************************/
{
  meshSpecs=_meshSpecs; 
  stateMgr=&_stateMgr;
}


/******************************************************************************/
void
ATO::OptimizationProblem::InitTopOpt()
/******************************************************************************/
{

  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
    wsElNodeEqID = disc->getWsElNodeEqID();


  int numPhysSets = meshSpecs.size();
  int numWorksets = wsElNodeEqID.size();

  cellTypes.resize(numPhysSets);
  cubatures.resize(numPhysSets);

  // build cubature for each physics set (elements with same type)
  refPoints.resize(numPhysSets);
  refWeights.resize(numPhysSets);
  for(int i=0; i<numPhysSets; i++){
    cellTypes[i] = Teuchos::rcp(new shards::CellTopology (&meshSpecs[i]->ctd));
    Intrepid::DefaultCubatureFactory<double> cubFactory;
    cubatures[i] = cubFactory.create(*(cellTypes[i]), meshSpecs[i]->cubatureDegree);

    int numDims = cubatures[i]->getDimension();
    int numQPs = cubatures[i]->getNumPoints();

    refPoints[i].resize(numQPs, numDims);
    refWeights[i].resize(numQPs);
    cubatures[i]->getCubature(refPoints[i],refWeights[i]);
  }

  Intrepid::FieldContainer<double> jacobian;
  Intrepid::FieldContainer<double> jacobian_det;
  Intrepid::FieldContainer<double> coordCon;

  weighted_measure.resize(numWorksets);
  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];
    int numCells  = wsElNodeEqID[ws].size();
    int numNodes  = wsElNodeEqID[ws][0].size();
    int numDims   = wsElNodeEqID[ws][0][0].size();
    int numQPs    = cubatures[physIndex]->getNumPoints();

    coordCon.resize(numCells, numNodes, numDims);
    jacobian.resize(numCells,numQPs,numDims,numDims);
    jacobian_det.resize(numCells,numQPs);
    weighted_measure[ws].resize(numCells,numQPs);

    for(int cell=0; cell<numCells; cell++)
      for(int node=0; node<numNodes; node++)
        for(int dim=0; dim<numDims; dim++)
          coordCon(cell,node,dim) = coords[ws][cell][node][dim];

    Intrepid::CellTools<double>::setJacobian(jacobian, refPoints[physIndex], 
                                             coordCon, *(cellTypes[physIndex]));
    Intrepid::CellTools<double>::setJacobianDet(jacobian_det, jacobian);
    Intrepid::FunctionSpaceTools::computeCellMeasure<double>
     (weighted_measure[ws], jacobian_det, refWeights[physIndex]);


  }
}


