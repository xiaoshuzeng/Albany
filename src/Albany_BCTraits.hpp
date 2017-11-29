#ifndef ALBANY_BC_TRAITS_HPP
#define ALBANY_BC_TRAITS_HPP

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_FactoryTraits.hpp"

#include <string>

//! Traits classes used for BCUtils

namespace Albany
{

//! Dirichlet BC Traits
struct DirichletTraits {
  enum {
    type = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet
  };
  enum {
    typeTd = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc
  };
  enum {
    typeTs = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_timedep_sdbc
  };
  enum {
    typeKf = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_kfield_bc
  };
  enum {
    typeEq =
        PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_eq_concentration_bc
  };
  enum {
    typeTo = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_torsion_bc
  };
  enum {
    typeSt = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_strong_dbc
  };
  enum {
    typeSw = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_schwarz_bc
  };
  enum {
    typeSsw =
        PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_strong_schwarz_bc
  };
  enum {
    typePd =
        PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_pd_neigh_fit_bc
  };
  enum {
    typeDa = PHAL::DirichletFactoryTraits<
        PHAL::AlbanyTraits>::id_dirichlet_aggregator
  };
  enum {
    typeFb = PHAL::DirichletFactoryTraits<
        PHAL::AlbanyTraits>::id_dirichlet_coordinate_function
  };
  enum {
    typeF = PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_field
  };

  static const std::string bcParamsPl;

  typedef PHAL::DirichletFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<const Teuchos::ParameterList>
  getValidBCParameters(
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& bcNames);

  static std::string
  constructBCName(const std::string& ns, const std::string& dof);

  static std::string
  constructStrongDBCName(const std::string& ns, const std::string& dof);

  static std::string
  constructBCNameField(const std::string& ns, const std::string& dof);

  static std::string
  constructStrongDBCNameField(const std::string& ns, const std::string& dof);

  static std::string
  constructTimeDepBCName(const std::string& ns, const std::string& dof);

  static std::string
  constructTimeDepStrongDBCName(const std::string& ns, const std::string& dof);

  static std::string
  constructPressureDepBCName(const std::string& ns, const std::string& dof);
};

//! Neumann BC Traits
struct NeumannTraits {
  enum { type = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann };
  enum {
    typeNa =
        PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_neumann_aggregator
  };
  enum {
    typeGCV =
        PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_coord_vector
  };
  enum {
    typeGS = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_gather_solution
  };
  enum {
    typeTd = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_timedep_bc
  };
  enum {
    typeSF = PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits>::id_load_stateField
  };
  enum {
    typeSNP = PHAL::NeumannFactoryTraits<
        PHAL::AlbanyTraits>::id_GatherScalarNodalParameter
  };

  static const std::string bcParamsPl;

  typedef PHAL::NeumannFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<const Teuchos::ParameterList>
  getValidBCParameters(
      const std::vector<std::string>& sideSetIDs,
      const std::vector<std::string>& bcNames,
      const std::vector<std::string>& conditions);

  static std::string
  constructBCName(
      const std::string& ns, const std::string& dof,
      const std::string& condition);

  static std::string
  constructTimeDepBCName(
      const std::string& ns, const std::string& dof,
      const std::string& condition);
};

//! Dirichlet BC Traits for equations defined on side sets
struct SideEqDirichletTraits {
  enum {
    type = PHAL::SideEqDirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet
  };
  enum {
    typeDa = PHAL::SideEqDirichletFactoryTraits<
        PHAL::AlbanyTraits>::id_dirichlet_aggregator
  };
  enum {
    typeF = PHAL::SideEqDirichletFactoryTraits<PHAL::AlbanyTraits>::id_dirichlet_field
  };
  enum {
    typeOS = PHAL::SideEqDirichletFactoryTraits<
        PHAL::AlbanyTraits>::id_dirichlet_off_sideset
  };

  static const std::string bcParamsPl;

  typedef PHAL::SideEqDirichletFactoryTraits<PHAL::AlbanyTraits> factory_type;

  static Teuchos::RCP<const Teuchos::ParameterList>
  getValidBCParameters(
      const std::vector<std::string>& sideSetsNames,
      const std::map<std::string,std::vector<std::string>>& nodeSetsNames,
      const std::vector<std::string>& dofsNames);

  static std::string
  constructBCName(const std::string& ss, const std::string& ns, const std::string& dof);

  static std::string
  constructBCNameField(const std::string& ss, const std::string& ns, const std::string& dof);

  static std::string
  constructBCNameOffSideSet(const std::string& ss, const std::string& dof);
};

} // namespace Albany

#endif // ALBANY_BC_TRAITS_HPP
