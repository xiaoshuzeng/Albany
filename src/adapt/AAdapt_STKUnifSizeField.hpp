//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_STKUNIFSIZEFIELD_HPP
#define AADAPT_STKUNIFSIZEFIELD_HPP

#ifdef HAZ_PERCEPT
#include <stk_percept/PerceptMesh.hpp>
#include <stk_percept/function/ElementOp.hpp>
#endif

namespace AAdapt {

class STKUnifRefineField
#ifdef HAZ_PERCEPT
: public stk::percept::ElementOp
#endif
{

  public:

#ifdef HAZ_PERCEPT
  STKUnifRefineField(stk::percept::PerceptMesh& eMesh) : m_eMesh(eMesh) {
  }
#endif

    virtual bool operator()(const stk::mesh::Entity element,
                            stk::mesh::FieldBase* field,  const stk::mesh::BulkData& bulkData);
    virtual void init_elementOp() {}
    virtual void fini_elementOp() {}

  private:
#ifdef HAZ_PERCEPT
    stk::percept::PerceptMesh& m_eMesh;
#endif
};

class STKUnifUnrefineField
#ifdef HAZ_PERCEPT
: public stk::percept::ElementOp
#endif
{

  public:

#ifdef HAZ_PERCEPT
    STKUnifUnrefineField(stk::percept::PerceptMesh& eMesh) : m_eMesh(eMesh) {
    }
#endif

    virtual bool operator()(const stk::mesh::Entity element,
                            stk::mesh::FieldBase* field,  const stk::mesh::BulkData& bulkData);
    virtual void init_elementOp() {}
    virtual void fini_elementOp() {}

  private:
#ifdef HAZ_PERCEPT
    stk::percept::PerceptMesh& m_eMesh;
#endif
};

}

#endif

