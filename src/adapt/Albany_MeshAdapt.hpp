//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_MESHADAPT_HPP
#define ALBANY_MESHADAPT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractAdapter.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"


namespace Albany {

class MeshAdapt : public AbstractAdapter {
public:

   MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_);
   //! Destructor
    ~MeshAdapt();

    //! Check adaptation criteria to determine if the mesh needs adapting
    virtual bool queryAdaptationCriteria();

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    virtual bool adaptMesh();

    //! Transfer solution between meshes.
    virtual void solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution);

   //! Each adapter must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;

private:

   // Disallow copy and assignment
   MeshAdapt(const MeshAdapt &);
   MeshAdapt &operator=(const MeshAdapt &);

   int numDim;
   int remeshFileIndex;

};

}

#endif //ALBANY_MESHADAPT_HPP
