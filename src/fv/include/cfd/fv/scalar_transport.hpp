#pragma once

#include <cfd/fv/fv_geometry.hpp>
#include <cfd/fv/ldu_matrix.hpp>
#include <vector>

namespace cfd::fv {

// FixedGradient: value = outward normal gradient d(phi)/dn, i.e. the diffusive flux into
// the domain per unit area is gamma * value.
struct ScalarBoundary {
    enum class Kind { FixedValue, ZeroGradient, FixedGradient };
    Kind kind = Kind::ZeroGradient;
    Scalar value = 0.0;
};

// Steady or transient convection-diffusion equation for a cell scalar phi:
//   ddt(phi) + div(F phi) - div(Gamma grad phi) = su + sp * phi    (sp <= 0 recommended)
// Sources su/sp are per unit volume. The convection term is discretised in bounded form
// (div(F) * phi subtracted), with upwind or linear-upwind (deferred correction).
struct ScalarEquation {
    std::vector<Scalar> gamma_face;       // diffusivity per face (all faces)
    std::vector<Scalar> su;               // explicit source per cell
    std::vector<Scalar> sp;               // implicit source coefficient per cell
    std::vector<ScalarBoundary> boundary; // per boundary face (f - n_internal_faces)
    std::vector<char> fixed_cell;         // optional: 1 = cell value fixed
    std::vector<Scalar> fixed_value;      // values for fixed cells
    Scalar relaxation = 1.0;              // implicit under-relaxation (steady)
    Scalar dt = 0.0;                      // > 0: implicit Euler time term with old_value
    const std::vector<Scalar>* old_value = nullptr;
    bool linear_upwind = true;
};

// Boundary-face values of a cell scalar for the given boundary conditions.
[[nodiscard]] std::vector<Scalar> boundary_values(const FvGeometry& geo,
                                                  const std::vector<Scalar>& field,
                                                  const std::vector<ScalarBoundary>& bc);

// Gauss-linear gradient of a cell scalar.
[[nodiscard]] std::vector<Vec3> scalar_gradient(const FvGeometry& geo,
                                                const std::vector<Scalar>& field,
                                                const std::vector<ScalarBoundary>& bc);

// Assembles and solves the equation in place; returns the linear solve result.
LinearSolveResult solve_scalar(const FvGeometry& geo, const std::vector<Scalar>& face_flux,
                               const ScalarEquation& eq, std::vector<Scalar>& field,
                               const LinearSolverSettings& settings);

} // namespace cfd::fv
