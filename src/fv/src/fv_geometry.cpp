#include <algorithm>
#include <cfd/fv/fv_geometry.hpp>
#include <cmath>
#include <stdexcept>

namespace cfd::fv {

namespace {

// Limits Sf.d from below to keep the over-relaxed coefficient bounded on very
// non-orthogonal faces (|cos| >= 0.05, i.e. non-orthogonality < ~87 deg).
constexpr Scalar k_min_cos = 0.05;

} // namespace

FvGeometry::FvGeometry(const Mesh& mesh, const parallel::Halo* halo)
    : mesh_(&mesh), halo_(halo), pattern_(build_csr_pattern(mesh)) {
    const auto nf = static_cast<std::size_t>(mesh.n_faces());
    const auto nif = static_cast<std::size_t>(mesh.n_internal_faces());
    const auto& owner = mesh.owner();
    const auto& neighbour = mesh.neighbour();
    const auto& cc = mesh.cell_centres();
    const auto& cf = mesh.face_centres();
    const auto& sf = mesh.face_area_vectors();

    weights_.assign(nif, 0.5);
    delta_coeffs_.resize(nf);
    corrections_.resize(nf);
    unit_normals_.resize(nf);
    face_areas_.resize(nf);

    for (std::size_t f = 0; f < nf; ++f) {
        const Vec3& s = sf[f];
        const Scalar area = s.norm();
        if (area <= 0.0) {
            throw std::runtime_error("FvGeometry: zero-area face");
        }
        face_areas_[f] = area;
        unit_normals_[f] = s / area;

        const Vec3& cp = cc[static_cast<std::size_t>(owner[f])];
        const Vec3 d =
            f < nif ? Vec3(cc[static_cast<std::size_t>(neighbour[f])] - cp) : Vec3(cf[f] - cp);
        const Scalar s_dot_d = std::max(s.dot(d), k_min_cos * area * d.norm());
        if (s_dot_d <= 0.0) {
            throw std::runtime_error("FvGeometry: degenerate cell-to-face distance");
        }
        delta_coeffs_[f] = area * area / s_dot_d;
        corrections_[f] = s - d * delta_coeffs_[f];

        if (f < nif) {
            const Vec3& cn = cc[static_cast<std::size_t>(neighbour[f])];
            const Scalar dp = s.dot(cf[f] - cp);
            const Scalar dn = s.dot(cn - cf[f]);
            const Scalar sum = dp + dn;
            weights_[f] = sum > 0.0 ? std::clamp(dn / sum, 0.0, 1.0) : 0.5;
        }
    }

    boundary_patch_.assign(nf - nif, -1);
    const auto& patches = mesh.patches();
    for (std::size_t p = 0; p < patches.size(); ++p) {
        for (Label i = 0; i < patches[p].n_faces; ++i) {
            const auto f = static_cast<std::size_t>(patches[p].start_face + i);
            boundary_patch_[f - nif] = static_cast<Label>(p);
        }
    }
}

} // namespace cfd::fv
