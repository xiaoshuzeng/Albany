//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_BCUtils.hpp"
#include "Albany_BCUtils_Def.hpp"

// Define macro for explicit template instantiation
#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS_DIRICHLET(name) \
  template class name<Albany::DirichletTraits>;
#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS_NEUMANN(name) \
  template class name<Albany::NeumannTraits>;
#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS_SIDE_EQ_DIRICHLET(name) \
  template class name<Albany::SideEqDirichletTraits>;

#define BCUTILS_INSTANTIATE_TEMPLATE_CLASS(name)              \
  BCUTILS_INSTANTIATE_TEMPLATE_CLASS_DIRICHLET(name)          \
  BCUTILS_INSTANTIATE_TEMPLATE_CLASS_NEUMANN(name)            \
  BCUTILS_INSTANTIATE_TEMPLATE_CLASS_SIDE_EQ_DIRICHLET(name)

BCUTILS_INSTANTIATE_TEMPLATE_CLASS(Albany::BCUtils)
