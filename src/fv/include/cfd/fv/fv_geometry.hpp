#pragma once

#include <Eigen/Dense>
#include <cfd/fv/ldu_matrix.hpp>
#include <cfd/mesh/mesh.hpp>
#include <vector>

namespace cfd::fv {

using core::Label;
using core::Scalar;
using core::Vec3;
using mesh::Mesh;

// Precomputed finite-volume face factors.
//
// For every face (internal and boundary) with d = C_N - C_P (boundary: C_f - C_P):
//   delta_coeff = |Sf|^2 / (Sf . d)        (over-relaxed implicit diffusion coefficient)
//   correction  = Sf - d * delta_coeff     (explicit non-orthogonal correction vector)
// For internal faces, weight is the owner-side linear interpolation weight:
//   phi_f = weight * phi_P + (1 - weight) * phi_N
class FvGeometry {
public:
    // halo (optional, must outlive the geometry): the mesh is one part of a decomposed mesh.
    explicit FvGeometry(const Mesh& mesh, const parallel::Halo* halo = nullptr);

    [[nodiscard]] const Mesh& mesh() const { return *mesh_; }
    [[nodiscard]] const parallel::Halo* halo() const { return halo_; }
    [[nodiscard]] const parallel::Comm& comm() const {
        return halo_ != nullptr ? *halo_->comm : parallel::Comm::self();
    }
    // Number of owned cells (all cells without a halo).
    [[nodiscard]] Label n_owned() const {
        return halo_ != nullptr ? halo_->n_owned : mesh_->n_cells();
    }
    // True if a boundary face (global face index) belongs to an owned cell; sums over
    // boundary faces must skip the others to avoid double counting.
    [[nodiscard]] bool owned_face(std::size_t face) const {
        return mesh_->owner()[face] < n_owned();
    }
    // Updates the ghost cells (no-op without a halo).
    template <typename T> void exchange(std::vector<T>& values) const {
        if (halo_ != nullptr) {
            halo_->exchange(values);
        }
    }
    // Is the boundary face part of the halo patch? (indexed by f - n_internal_faces)
    [[nodiscard]] bool halo_face(std::size_t boundary_face) const {
        return halo_ != nullptr && boundary_patch_[boundary_face] == halo_->halo_patch;
    }
    [[nodiscard]] const std::vector<Scalar>& weights() const { return weights_; }
    [[nodiscard]] const std::vector<Scalar>& delta_coeffs() const { return delta_coeffs_; }
    [[nodiscard]] const std::vector<Vec3>& corrections() const { return corrections_; }
    // Weighted least-squares gradient (weights 1/|d|^2 over the face neighbours and boundary
    // face centres): grad_P = M_P^-1 sum w d (phi_N - phi_P). Second-order on any cell shape
    // (Green-Gauss with linear face interpolation is not on skewed tetrahedra).
    // boundary_values: per boundary face (f - n_internal_faces).
    [[nodiscard]] std::vector<Vec3>
    least_squares_gradient(const std::vector<Scalar>& cell_values,
                           const std::vector<Scalar>& boundary_values) const;
    [[nodiscard]] const std::vector<Vec3>& unit_normals() const { return unit_normals_; }
    [[nodiscard]] const std::vector<Scalar>& face_areas() const { return face_areas_; }

    // Index of the patch owning each boundary face (indexed by f - n_internal_faces).
    [[nodiscard]] const std::vector<Label>& boundary_patch() const { return boundary_patch_; }

    // CSR pattern of the cell matrix (for fast LduMatrix -> CSR conversion).
    [[nodiscard]] const CsrPattern& csr_pattern() const { return pattern_; }

private:
    void build_least_squares() const;
    mutable std::vector<Eigen::Matrix3d> lsq_inverse_;
    const Mesh* mesh_;
    const parallel::Halo* halo_ = nullptr;
    std::vector<Scalar> weights_;
    std::vector<Scalar> delta_coeffs_;
    std::vector<Vec3> corrections_;
    std::vector<Vec3> unit_normals_;
    std::vector<Scalar> face_areas_;
    std::vector<Label> boundary_patch_;
    CsrPattern pattern_;
};

} // namespace cfd::fv
