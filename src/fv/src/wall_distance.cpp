#include <algorithm>
#include <cfd/fv/scalar_transport.hpp>
#include <cfd/fv/wall_distance.hpp>
#include <cmath>
#include <limits>

namespace cfd::fv {

std::vector<Scalar> wall_distance(const FvGeometry& geo, const std::vector<char>& is_wall_face) {
    const Mesh& m = geo.mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());

    if (std::none_of(is_wall_face.begin(), is_wall_face.end(), [](char c) { return c != 0; })) {
        return std::vector<Scalar>(nc, std::numeric_limits<Scalar>::max() / 1e10);
    }

    ScalarEquation eq;
    eq.gamma_face.assign(nf, 1.0);
    eq.su.assign(nc, 1.0);
    eq.sp.assign(nc, 0.0);
    eq.boundary.resize(nf - nif);
    for (std::size_t b = 0; b < nf - nif; ++b) {
        if (is_wall_face[b] != 0) {
            eq.boundary[b] = {ScalarBoundary::Kind::FixedValue, 0.0};
        }
    }
    eq.linear_upwind = false;

    std::vector<Scalar> phi(nc, 0.0);
    const std::vector<Scalar> no_flux(nf, 0.0);
    const LinearSolverSettings settings{LinearSolverType::BiCgStab, 1e-12, 1e-8, 5000};
    for (int corr = 0; corr < 3; ++corr) { // non-orthogonal correction is explicit
        (void)solve_scalar(geo, no_flux, eq, phi, settings);
    }
    const std::vector<Vec3> grad = scalar_gradient(geo, phi, eq.boundary);

    std::vector<Scalar> y(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        const Scalar g = grad[c].norm();
        y[c] = std::sqrt(std::max(g * g + 2.0 * phi[c], 0.0)) - g;
    }

    // Exact normal distance for wall-adjacent cells.
    std::vector<Scalar> near(nc, std::numeric_limits<Scalar>::max());
    for (std::size_t f = nif; f < nf; ++f) {
        if (is_wall_face[f - nif] != 0) {
            const auto p = static_cast<std::size_t>(m.owner()[f]);
            const Scalar d =
                std::abs((m.cell_centres()[p] - m.face_centres()[f]).dot(geo.unit_normals()[f]));
            near[p] = std::min(near[p], d);
        }
    }
    for (std::size_t c = 0; c < nc; ++c) {
        if (near[c] < std::numeric_limits<Scalar>::max()) {
            y[c] = near[c];
        }
        y[c] = std::max(y[c], 1e-15);
    }
    return y;
}

} // namespace cfd::fv
