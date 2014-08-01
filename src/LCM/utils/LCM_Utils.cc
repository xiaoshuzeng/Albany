//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "LCM_Utils.h"

namespace LCM {

Teuchos::RCP<MaterialDatabase>
createMaterialDatabase(std::string const & filename,
    Teuchos::RCP<Epetra_Comm const> const & comm)
{
  return Teuchos::rcp(new MaterialDatabase(filename, comm));
}

} // namespace LCM
