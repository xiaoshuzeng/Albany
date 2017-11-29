#include "Albany_BCTraits.hpp"

namespace Albany
{

// ================== Dirichlet ================== //

// Initialize statics
const std::string Albany::DirichletTraits::bcParamsPl = "Dirichlet BCs";


std::string
Albany::DirichletTraits::constructBCName(
    const std::string& ns, const std::string& dof) {
  std::stringstream ss;
  ss << "DBC on NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::DirichletTraits::constructStrongDBCName(
    const std::string& ns, const std::string& dof) {
  std::stringstream ss;
  ss << "SDBC on NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::DirichletTraits::constructBCNameField(
    const std::string& ns, const std::string& dof) {
  std::stringstream ss;
  ss << "DBC on NS " << ns << " for DOF " << dof << " prescribe Field";

  return ss.str();
}

std::string
Albany::DirichletTraits::constructStrongDBCNameField(
    const std::string& ns, const std::string& dof) {
  std::stringstream ss;
  ss << "SDBC on NS " << ns << " for DOF " << dof << " prescribe Field";

  return ss.str();
}

std::string
Albany::DirichletTraits::constructTimeDepBCName(
    const std::string& ns, const std::string& dof) {
  std::stringstream ss;
  ss << "Time Dependent " << Albany::DirichletTraits::constructBCName(ns, dof);
  return ss.str();
}

std::string
Albany::DirichletTraits::constructTimeDepStrongDBCName(
    const std::string& ns, const std::string& dof) {
  std::stringstream ss;
  ss << "Time Dependent "
     << Albany::DirichletTraits::constructStrongDBCName(ns, dof);
  return ss.str();
}

std::string
Albany::DirichletTraits::constructPressureDepBCName(
    const std::string& ns, const std::string& dof) {
  std::stringstream ss;
  ss << "Pressure Dependent "
     << Albany::DirichletTraits::constructBCName(ns, dof);
  return ss.str();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::DirichletTraits::getValidBCParameters(
    const std::vector<std::string>& nodeSetIDs,
    const std::vector<std::string>& bcNames) {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));

  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      std::string ss =
          Albany::DirichletTraits::constructBCName(nodeSetIDs[i], bcNames[j]);
      std::string tt = Albany::DirichletTraits::constructTimeDepBCName(
          nodeSetIDs[i], bcNames[j]);
      std::string ts = Albany::DirichletTraits::constructTimeDepStrongDBCName(
          nodeSetIDs[i], bcNames[j]);
      std::string pp = Albany::DirichletTraits::constructPressureDepBCName(
          nodeSetIDs[i], bcNames[j]);
      std::string st = Albany::DirichletTraits::constructStrongDBCName(
          nodeSetIDs[i], bcNames[j]);
      validPL->set<double>(
          ss, 0.0, "Value of BC corresponding to nodeSetID and dofName");
      validPL->set<double>(
          st, 0.0, "Value of SDBC corresponding to nodeSetID and dofName");
      validPL->sublist(
          tt, false, "SubList of BC corresponding to nodeSetID and dofName");
      validPL->sublist(
          ts, false, "SubList of SDBC corresponding to nodeSetID and dofName");
      validPL->sublist(
          pp, false, "SubList of BC corresponding to nodeSetID and dofName");
      ss = Albany::DirichletTraits::constructBCNameField(
          nodeSetIDs[i], bcNames[j]);
      st = Albany::DirichletTraits::constructStrongDBCNameField(
          nodeSetIDs[i], bcNames[j]);
      validPL->set<std::string>(
          ss, "dirichlet field", "Field used to prescribe Dirichlet BCs");
      validPL->set<std::string>(
          st, "dirichlet field", "Field used to prescribe Strong DBCs");
    }
  }

  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    std::string ss =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "K");
    std::string tt =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "twist");
    std::string ww =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "Schwarz");
    std::string sw =
        Albany::DirichletTraits::constructStrongDBCName(nodeSetIDs[i], "StrongSchwarz");
    std::string uu =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "CoordFunc");
    std::string pd =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "lsfit");
    validPL->sublist(ss, false, "");
    validPL->sublist(tt, false, "");
    validPL->sublist(ww, false, "");
    validPL->sublist(sw, false, "");
    validPL->sublist(uu, false, "");
    validPL->sublist(pd, false, "");
  }

  return validPL;
}

// ====================== Neumann ========================= //

// Initialize statics
const std::string Albany::NeumannTraits::bcParamsPl = "Neumann BCs";

std::string
Albany::NeumannTraits::constructBCName(
    const std::string& ns, const std::string& dof,
    const std::string& condition) {
  std::stringstream ss;
  ss << "NBC on SS " << ns << " for DOF " << dof << " set " << condition;
  return ss.str();
}

std::string
Albany::NeumannTraits::constructTimeDepBCName(
    const std::string& ns, const std::string& dof,
    const std::string& condition) {
  std::stringstream ss;
  ss << "Time Dependent "
     << Albany::NeumannTraits::constructBCName(ns, dof, condition);
  return ss.str();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NeumannTraits::getValidBCParameters(
    const std::vector<std::string>& sideSetIDs,
    const std::vector<std::string>& bcNames,
    const std::vector<std::string>& conditions) {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      Teuchos::rcp(new Teuchos::ParameterList("Valid Neumann BC List"));
  ;

  for (std::size_t i = 0; i < sideSetIDs.size();
       i++) {  // loop over all side sets in the mesh
    for (std::size_t j = 0; j < bcNames.size();
         j++) {  // loop over all possible types of condition
      for (std::size_t k = 0; k < conditions.size();
           k++) {  // loop over all possible types of condition

        std::string ss = Albany::NeumannTraits::constructBCName(
            sideSetIDs[i], bcNames[j], conditions[k]);
        std::string tt = Albany::NeumannTraits::constructTimeDepBCName(
            sideSetIDs[i], bcNames[j], conditions[k]);

        /*
         if(numDim == 2)
         validPL->set<Teuchos::Array<double>>(ss, Teuchos::tuple<double>(0.0,
         0.0),
         "Value of BC corresponding to sideSetID and boundary condition");
         else
         validPL->set<Teuchos::Array<double>>(ss, Teuchos::tuple<double>(0.0,
         0.0, 0.0),
         "Value of BC corresponding to sideSetID and boundary condition");
         */
        Teuchos::Array<double> defaultData;
        validPL->set<Teuchos::Array<double>>(
            ss, defaultData,
            "Value of BC corresponding to sideSetID and boundary condition");

        validPL->sublist(
            tt, false,
            "SubList of BC corresponding to sideSetID and boundary condition");
      }
    }
  }

  validPL->set<std::string>("BetaXY", "Constant", "Function Type for Basal BC");
  validPL->set<int>("Cubature Degree", 3, "Cubature Degree for Neumann BC");
  validPL->set<double>("L", 1, "Length Scale for ISMIP-HOM Tests");
  return validPL;
}

// =============== Side Eqn Dirichlet ================== //

// Initialize statics
const std::string Albany::SideEqDirichletTraits::bcParamsPl = "Side Eqn Dirichlet BCs";

std::string
Albany::SideEqDirichletTraits::constructBCName(
    const std::string& ss, const std::string& ns, const std::string& dof) {
  std::stringstream sstr;
  sstr << "DBC on SS " << ss << ", NS " << ns << " for DOF " << dof;

  return sstr.str();
}

std::string
Albany::SideEqDirichletTraits::constructBCNameField(
    const std::string& ss, const std::string& ns, const std::string& dof) {
  std::stringstream sstr;
  sstr << "DBC on SS " << ss << ", NS " << ns << " for DOF " << dof << " prescribe Field";

  return sstr.str();
}


std::string
Albany::SideEqDirichletTraits::constructBCNameOffSideSet(
    const std::string& ss, const std::string& dof) {
  std::stringstream sstr;
  sstr << "DBC off SS " << ss << " for DOF " << dof;

  return sstr.str();
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::SideEqDirichletTraits::getValidBCParameters(
    const std::vector<std::string>& sideSetNames,
    const std::map<std::string,std::vector<std::string>>& sideNodeSetNames,
    const std::vector<std::string>& dofNames) {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));

  std::string bc;
  for (const auto& dof : dofNames) {
    for (const auto& sideSet : sideSetNames) {
      for (const auto& nodeSet: sideNodeSetNames.at(sideSet)) {
        bc = Albany::SideEqDirichletTraits::constructBCName(sideSet,nodeSet,dof);
        validPL->set<double>(bc, 0.0, "Value of BC corresponding to sideSet's nodeSet and dofName");

        bc = Albany::SideEqDirichletTraits::constructBCNameField(sideSet,nodeSet,dof);
        validPL->set<std::string>(bc, "dirichlet field", "Field used to prescribe Dirichlet BCs on a sideSet's nodeSet for given field");
      }
      bc = Albany::SideEqDirichletTraits::constructBCNameOffSideSet(sideSet,dof);
      validPL->set<double>(bc, 0.0, "Value of BC to prescribe off the given sideset (use multiple "
                                     "entries for multiple side sets) for given field");
    }
  }

  return validPL;
}

} // namespace Albany
