#include <algorithm>
#include <cfd/fv/scalar_transport.hpp>
#include <cfd/gpu/krylov.hpp>
#include <cmath>
#include <stdexcept>

namespace cfd::fv {

std::vector<Scalar> boundary_values(const FvGeometry& geo, const std::vector<Scalar>& field,
                                    const std::vector<ScalarBoundary>& bc) {
    const Mesh& m = geo.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
    std::vector<Scalar> values(nb);
    for (std::size_t b = 0; b < nb; ++b) {
        const auto p = static_cast<std::size_t>(m.owner()[nif + b]);
        switch (bc[b].kind) {
        case ScalarBoundary::Kind::FixedValue:
            values[b] = bc[b].value;
            break;
        case ScalarBoundary::Kind::ZeroGradient:
            values[b] = field[p];
            break;
        case ScalarBoundary::Kind::FixedGradient: {
            const Scalar dn = std::abs(
                (m.face_centres()[nif + b] - m.cell_centres()[p]).dot(geo.unit_normals()[nif + b]));
            values[b] = field[p] + bc[b].value * dn;
            break;
        }
        case ScalarBoundary::Kind::Mixed: {
            const Scalar dn = std::abs(
                (m.face_centres()[nif + b] - m.cell_centres()[p]).dot(geo.unit_normals()[nif + b]));
            const Scalar rd = bc[b].coefficient * dn;
            values[b] = (field[p] + rd * bc[b].value) / (1.0 + rd);
            break;
        }
        }
    }
    return values;
}

std::vector<Vec3> scalar_gradient(const FvGeometry& geo, const std::vector<Scalar>& field,
                                  const std::vector<ScalarBoundary>& bc, GradientScheme scheme) {
    if (scheme == GradientScheme::LeastSquares) {
        return geo.least_squares_gradient(field, boundary_values(geo, field, bc));
    }
    const Mesh& m = geo.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo.weights();
    const auto& sf = m.face_area_vectors();
    const std::vector<Scalar> bv = boundary_values(geo, field, bc);
    std::vector<Vec3> grad(nc, Vec3::Zero());
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const Vec3 flux = sf[f] * (w[f] * field[p] + (1.0 - w[f]) * field[n]);
        grad[p] += flux;
        grad[n] -= flux;
    }
    for (std::size_t f = nif; f < nf; ++f) {
        grad[static_cast<std::size_t>(m.owner()[f])] += sf[f] * bv[f - nif];
    }
    for (std::size_t c = 0; c < nc; ++c) {
        grad[c] /= m.cell_volumes()[c];
    }
    geo.exchange(grad);
    return grad;
}

LinearSolveResult solve_scalar(const FvGeometry& geo, const std::vector<Scalar>& face_flux,
                               const ScalarEquation& eq, std::vector<Scalar>& field,
                               const LinearSolverSettings& settings) {
    const Mesh& m = geo.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& own = m.owner();
    const auto& nei = m.neighbour();
    const auto& w = geo.weights();
    const auto& dc = geo.delta_coeffs();
    const auto& kv = geo.corrections();
    const auto& cf = m.face_centres();
    const auto& cc = m.cell_centres();
    const auto& vol = m.cell_volumes();

    if (eq.gamma_face.size() != nf || eq.su.size() != nc || eq.sp.size() != nc ||
        eq.boundary.size() != nf - nif || field.size() != nc) {
        throw std::invalid_argument("solve_scalar: size mismatch");
    }

    const std::vector<Vec3> grad = scalar_gradient(geo, field, eq.boundary, eq.gradient);
    LduMatrix a(m);
    a.pattern = &geo.csr_pattern();
    a.halo = geo.halo();
    std::vector<Scalar> divergence(nc, 0.0);

    // Implicit internal-face coefficients: on the accelerator when GPU solvers are selected.
    const bool device =
        settings.type == LinearSolverType::GpuCg || settings.type == LinearSolverType::GpuBiCgStab;
    if (device) {
        gpu::assemble_convection_diffusion(static_cast<int>(nc), static_cast<int>(nif), own.data(),
                                           nei.data(), face_flux.data(), eq.gamma_face.data(),
                                           dc.data(), 1.0, a.upper.data(), a.lower.data(),
                                           a.diag.data(), divergence.data());
    }
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const auto n = static_cast<std::size_t>(nei[f]);
        const Scalar flux = face_flux[f];
        if (!device) {
            const Scalar diff = eq.gamma_face[f] * dc[f];
            a.upper[f] = std::min(flux, 0.0) - diff;
            a.lower[f] = -std::max(flux, 0.0) - diff;
            a.diag[p] += std::max(flux, 0.0) + diff;
            a.diag[n] += -std::min(flux, 0.0) + diff;
            divergence[p] += flux;
            divergence[n] -= flux;
        }

        Scalar explicit_flux = 0.0;
        if (eq.linear_upwind) {
            const std::size_t up = flux >= 0.0 ? p : n;
            explicit_flux -= flux * grad[up].dot(cf[f] - cc[up]);
        }
        explicit_flux += eq.gamma_face[f] * kv[f].dot(w[f] * grad[p] + (1.0 - w[f]) * grad[n]);
        a.source[p] += explicit_flux;
        a.source[n] -= explicit_flux;
    }

    for (std::size_t f = nif; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const Scalar flux = face_flux[f];
        const ScalarBoundary& b = eq.boundary[f - nif];
        divergence[p] += flux;
        if (b.kind == ScalarBoundary::Kind::FixedValue) {
            const Scalar diff = eq.gamma_face[f] * dc[f];
            a.diag[p] += diff + std::max(flux, 0.0);
            a.source[p] += (diff - std::min(flux, 0.0)) * b.value;
        } else if (b.kind == ScalarBoundary::Kind::Mixed) {
            // gamma (phi_b - phi_P) / d = h (value - phi_b): flux coefficient gamma r / (1 + r d)
            const Scalar dn = std::abs((cf[f] - cc[p]).dot(geo.unit_normals()[f]));
            const Scalar rd = b.coefficient * dn;
            const Scalar coef = eq.gamma_face[f] * dc[f] * rd / (1.0 + rd);
            a.diag[p] += coef + flux;
            a.source[p] += coef * b.value;
        } else {
            a.diag[p] += flux;
            if (b.kind == ScalarBoundary::Kind::FixedGradient) {
                const Scalar dn = std::abs((cf[f] - cc[p]).dot(geo.unit_normals()[f]));
                a.source[p] +=
                    eq.gamma_face[f] * b.value * geo.face_areas()[f] - flux * b.value * dn;
            }
        }
    }

    for (std::size_t c = 0; c < nc; ++c) {
        a.diag[c] -= divergence[c]; // bounded convection form
        a.diag[c] -= std::min(eq.sp[c], 0.0) * vol[c];
        a.source[c] += eq.su[c] * vol[c] + std::max(eq.sp[c], 0.0) * field[c] * vol[c];
        if (eq.dt > 0.0) {
            const Scalar ddt = vol[c] / eq.dt * (eq.capacity != nullptr ? (*eq.capacity)[c] : 1.0);
            a.diag[c] += ddt;
            a.source[c] += ddt * (eq.old_value != nullptr ? (*eq.old_value)[c] : field[c]);
        }
        if (eq.relaxation < 1.0) {
            const Scalar d1 = a.diag[c] / eq.relaxation;
            a.source[c] += (d1 - a.diag[c]) * field[c];
            a.diag[c] = d1;
        }
    }

    if (!eq.fixed_cell.empty()) {
        for (std::size_t f = 0; f < nif; ++f) {
            if (eq.fixed_cell[static_cast<std::size_t>(own[f])] != 0) {
                a.upper[f] = 0.0;
            }
            if (eq.fixed_cell[static_cast<std::size_t>(nei[f])] != 0) {
                a.lower[f] = 0.0;
            }
        }
        for (std::size_t c = 0; c < nc; ++c) {
            if (eq.fixed_cell[c] != 0) {
                a.diag[c] = 1.0;
                a.source[c] = eq.fixed_value[c];
                field[c] = eq.fixed_value[c];
            }
        }
    }

    return a.solve(field, settings);
}

} // namespace cfd::fv
