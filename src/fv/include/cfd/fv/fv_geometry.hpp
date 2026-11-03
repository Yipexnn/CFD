#pragma once

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
    explicit FvGeometry(const Mesh& mesh);

    [[nodiscard]] const Mesh& mesh() const { return *mesh_; }
    [[nodiscard]] const std::vector<Scalar>& weights() const { return weights_; }
    [[nodiscard]] const std::vector<Scalar>& delta_coeffs() const { return delta_coeffs_; }
    [[nodiscard]] const std::vector<Vec3>& corrections() const { return corrections_; }
    [[nodiscard]] const std::vector<Vec3>& unit_normals() const { return unit_normals_; }
    [[nodiscard]] const std::vector<Scalar>& face_areas() const { return face_areas_; }

    // Index of the patch owning each boundary face (indexed by f - n_internal_faces).
    [[nodiscard]] const std::vector<Label>& boundary_patch() const { return boundary_patch_; }

    // CSR pattern of the cell matrix (for fast LduMatrix -> CSR conversion).
    [[nodiscard]] const CsrPattern& csr_pattern() const { return pattern_; }

private:
    const Mesh* mesh_;
    std::vector<Scalar> weights_;
    std::vector<Scalar> delta_coeffs_;
    std::vector<Vec3> corrections_;
    std::vector<Vec3> unit_normals_;
    std::vector<Scalar> face_areas_;
    std::vector<Label> boundary_patch_;
    CsrPattern pattern_;
};

} // namespace cfd::fv
