#include <algorithm>
#include <cfd/fv/fv_geometry.hpp>
#include <cfd/gpu/fv_device.hpp>
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

void FvGeometry::build_least_squares() const {
    const Mesh& m = *mesh_;
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& cc = m.cell_centres();
    const auto& cf = m.face_centres();
    std::vector<Eigen::Matrix3d> mat(nc, Eigen::Matrix3d::Zero());
    for (std::size_t f = 0; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const Vec3 d = f < nif ? Vec3(cc[static_cast<std::size_t>(m.neighbour()[f])] - cc[p])
                               : Vec3(cf[f] - cc[p]);
        const Eigen::Matrix3d dd = d * d.transpose() / d.squaredNorm();
        mat[p] += dd;
        if (f < nif) {
            mat[static_cast<std::size_t>(m.neighbour()[f])] += dd;
        }
    }
    lsq_inverse_.resize(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        lsq_inverse_[c] = mat[c].inverse();
    }
}

std::vector<Vec3>
FvGeometry::least_squares_gradient(const std::vector<Scalar>& cell_values,
                                   const std::vector<Scalar>& boundary_values) const {
    if (lsq_inverse_.empty()) {
        build_least_squares();
    }
    const Mesh& m = *mesh_;
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& cc = m.cell_centres();
    const auto& cf = m.face_centres();
    std::vector<Vec3> rhs(nc, Vec3::Zero());
    for (std::size_t f = 0; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        if (f < nif) {
            const auto n = static_cast<std::size_t>(m.neighbour()[f]);
            const Vec3 d = cc[n] - cc[p];
            const Vec3 g = d / d.squaredNorm() * (cell_values[n] - cell_values[p]);
            rhs[p] += g;
            rhs[n] += g; // (-d)(phi_P - phi_N) = d (phi_N - phi_P)
        } else {
            const Vec3 d = cf[f] - cc[p];
            rhs[p] += d / d.squaredNorm() * (boundary_values[f - nif] - cell_values[p]);
        }
    }
    std::vector<Vec3> grad(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        grad[c] = lsq_inverse_[c] * rhs[c];
    }
    exchange(grad);
    return grad;
}

std::vector<Vec3> FvGeometry::gauss_gradient(const std::vector<Scalar>& cell_values,
                                             const std::vector<Scalar>& boundary_values) const {
    const Mesh& m = *mesh_;
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = weights_;
    const auto& sf = m.face_area_vectors();
    std::vector<Vec3> grad(nc, Vec3::Zero());
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const Vec3 flux = sf[f] * (w[f] * cell_values[p] + (1.0 - w[f]) * cell_values[n]);
        grad[p] += flux;
        grad[n] -= flux;
    }
    for (std::size_t f = nif; f < nf; ++f) {
        grad[static_cast<std::size_t>(m.owner()[f])] += sf[f] * boundary_values[f - nif];
    }
    for (std::size_t c = 0; c < nc; ++c) {
        grad[c] /= m.cell_volumes()[c];
    }
    exchange(grad);
    return grad;
}

std::vector<Vec3> FvGeometry::gradient(const std::vector<Scalar>& cell_values,
                                       const std::vector<Scalar>& boundary_values,
                                       GradientScheme scheme, bool device) const {
    const bool lsq = scheme == GradientScheme::LeastSquares;
    if (device && device_capable()) {
        static_assert(sizeof(Vec3) == 3 * sizeof(Scalar));
        std::vector<Vec3> grad(cell_values.size());
        this->device().gradient(cell_values.data(), boundary_values.data(), lsq,
                                reinterpret_cast<Scalar*>(grad.data()));
        return grad;
    }
    return lsq ? least_squares_gradient(cell_values, boundary_values)
               : gauss_gradient(cell_values, boundary_values);
}

gpu::DeviceFv& FvGeometry::device() const {
    if (halo_ != nullptr) {
        throw std::logic_error("FvGeometry::device: not available for decomposed meshes");
    }
    if (!device_) {
        if (lsq_inverse_.empty()) {
            build_least_squares();
        }
        const Mesh& m = *mesh_;
        const auto nc = static_cast<std::size_t>(m.n_cells());
        const auto nf = static_cast<std::size_t>(m.n_faces());
        const auto nif = static_cast<std::size_t>(m.n_internal_faces());
        const auto& cc = m.cell_centres();
        const auto& cf = m.face_centres();
        std::vector<Scalar> lsq_d(3 * nf);
        for (std::size_t f = 0; f < nf; ++f) {
            const auto p = static_cast<std::size_t>(m.owner()[f]);
            const Vec3 d = f < nif ? Vec3(cc[static_cast<std::size_t>(m.neighbour()[f])] - cc[p])
                                   : Vec3(cf[f] - cc[p]);
            const Vec3 q = d / d.squaredNorm();
            for (int i = 0; i < 3; ++i) {
                lsq_d[3 * f + static_cast<std::size_t>(i)] = q[i];
            }
        }
        std::vector<Scalar> lsq(9 * nc);
        for (std::size_t c = 0; c < nc; ++c) {
            std::copy(lsq_inverse_[c].data(), lsq_inverse_[c].data() + 9, lsq.begin() + 9 * c);
        }
        auto vec = [](const std::vector<Vec3>& v) {
            static_assert(sizeof(Vec3) == 3 * sizeof(Scalar));
            return reinterpret_cast<const Scalar*>(v.data());
        };
        gpu::FvMeshView view;
        view.n_cells = m.n_cells();
        view.n_internal_faces = m.n_internal_faces();
        view.n_faces = m.n_faces();
        view.owner = m.owner().data();
        view.neighbour = m.neighbour().data();
        view.cell_offsets = m.cell_face_offsets().data();
        view.cell_faces = m.cell_faces().data();
        view.weight = weights_.data();
        view.sf = vec(m.face_area_vectors());
        view.face_centre = vec(cf);
        view.cell_centre = vec(cc);
        view.volume = m.cell_volumes().data();
        view.correction = vec(corrections_);
        view.lsq_d = lsq_d.data();
        view.lsq_inverse = lsq.data();
        device_ = std::make_shared<gpu::DeviceFv>(view);
    }
    return *device_;
}

} // namespace cfd::fv
