//
// $Id: FloatingPoint.h,v 1.1 2008/07/14 17:50:46 lxmota Exp $
//
// $Log: FloatingPoint.h,v $
// Revision 1.1  2008/07/14 17:50:46  lxmota
// Initial sources.
//
//

#if !defined(LCM_Utils_h)
#define LCM_Utils_h

#include <algorithm>

#include "QCAD_MaterialDatabase.hpp"
#include "Teuchos_RCP.hpp"

namespace LCM {

typedef QCAD::MaterialDatabase MaterialDatabase;

template <typename Container, typename T>
bool contains(Container const & c, T const & t)
{
  return std::find(c.begin(), c.end(), t) != c.end();
}

Teuchos::RCP<MaterialDatabase>
createMaterialDatabase(
    std::string const & filename,
    Teuchos::RCP<Epetra_Comm const> const & comm);

} // namespace LCM

#endif // LCM_Utils_h
