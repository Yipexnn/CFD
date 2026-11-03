#include <algorithm>
#include <cfd/fv/scalar_transport.hpp>
#include <stdexcept>

namespace cfd::fv {

std::vector<Scalar> boundary_values(const FvGeometry& geo, const std::vector<Scalar>& field,
                                    const std::vector<ScalarBoundary>& bc) {
    const Mesh& m = geo.mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
    std::vector<Scalar> values(nb);
    for (std::size_t b = 0; b < nb; ++b) {
        values[b] = bc[b].kind == ScalarBoundary::Kind::FixedValue
                        ? bc[b].value
                        : field[static_cast<std::size_t>(m.owner()[nif + b])];
    }
    return values;
}

std::vector<Vec3> scalar_gradient(const FvGeometry& geo, const std::vector<Scalar>& field,
                                  const std::vector<ScalarBoundary>& bc) {
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

    const std::vector<Vec3> grad = scalar_gradient(geo, field, eq.boundary);
    LduMatrix a(m);
    a.pattern = &geo.csr_pattern();
    std::vector<Scalar> divergence(nc, 0.0);

    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(own[f]);
        const auto n = static_cast<std::size_t>(nei[f]);
        const Scalar flux = face_flux[f];
        const Scalar diff = eq.gamma_face[f] * dc[f];
        a.upper[f] = std::min(flux, 0.0) - diff;
        a.lower[f] = -std::max(flux, 0.0) - diff;
        a.diag[p] += std::max(flux, 0.0) + diff;
        a.diag[n] += -std::min(flux, 0.0) + diff;
        divergence[p] += flux;
        divergence[n] -= flux;

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
        } else {
            a.diag[p] += flux;
        }
    }

    for (std::size_t c = 0; c < nc; ++c) {
        a.diag[c] -= divergence[c]; // bounded convection form
        a.diag[c] -= std::min(eq.sp[c], 0.0) * vol[c];
        a.source[c] += eq.su[c] * vol[c] + std::max(eq.sp[c], 0.0) * field[c] * vol[c];
        if (eq.dt > 0.0) {
            const Scalar ddt = vol[c] / eq.dt;
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
