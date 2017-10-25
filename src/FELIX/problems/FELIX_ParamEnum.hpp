#ifndef FELIX_PARAM_ENUM_HPP
#define FELIX_PARAM_ENUM_HPP

#include <string>

namespace FELIX
{

enum class FelixParamEnum
{
  Alpha        = 0,
  Lambda       = 1,
  Mu           = 2,
  Power        = 3,
  Homotopy     = 4,
  FlowFactorA  = 5
};

namespace ParamEnum
{
  static const std::string Alpha_name         = "Hydraulic-Over-Hydrostatic Potential Ratio";
  static const std::string Lambda_name        = "Bed Roughness";
  static const std::string Mu_name            = "Coulomb Friction Coefficient";
  static const std::string Power_name         = "Power Exponent";
  static const std::string HomotopyParam_name = "Homotopy Parameter";
  static const std::string FlowFactorA_name   = "Constant Flow Factor A";
} // ParamEnum

} // Namespace FELIX

#endif // FELIX_PARAM_ENUM_HPP
