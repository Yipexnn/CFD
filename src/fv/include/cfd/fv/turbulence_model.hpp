#pragma once

#include <cfd/core/types.hpp>
#include <string>
#include <utility>
#include <vector>

namespace cfd::fv {

using core::Scalar;

class FlowSolver;

// Eddy-viscosity turbulence model coupled to FlowSolver.
class TurbulenceModel {
public:
    TurbulenceModel() = default;
    TurbulenceModel(const TurbulenceModel&) = delete;
    TurbulenceModel& operator=(const TurbulenceModel&) = delete;
    TurbulenceModel(TurbulenceModel&&) = delete;
    TurbulenceModel& operator=(TurbulenceModel&&) = delete;
    virtual ~TurbulenceModel() = default;

    // Solves the model equations for the current flow state and updates nu_t.
    // Returns the initial normalised residuals of the two transport equations.
    virtual std::pair<Scalar, Scalar> correct(const FlowSolver& flow, Scalar dt) = 0;

    // Turbulent kinematic viscosity in cells.
    [[nodiscard]] virtual const std::vector<Scalar>& nut() const = 0;
    // Turbulent kinematic viscosity on boundary faces (indexed by f - n_internal_faces).
    [[nodiscard]] virtual const std::vector<Scalar>& boundary_nut() const = 0;
};

} // namespace cfd::fv
