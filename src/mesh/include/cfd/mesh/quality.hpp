#pragma once

#include <cfd/mesh/mesh.hpp>
#include <string>

namespace cfd::mesh {

// Summary of mesh quality metrics. Face metrics are computed over internal faces only;
// they remain 0 when the mesh has no internal faces.
struct MeshQuality {
    Label n_cells = 0;
    Scalar min_volume = 0.0;
    Scalar max_volume = 0.0;
    Scalar total_volume = 0.0;
    Scalar max_non_orthogonality = 0.0; // degrees
    Scalar avg_non_orthogonality = 0.0; // degrees
    Scalar max_skewness = 0.0;
    Scalar max_aspect_ratio = 1.0;
    Label n_negative_volumes = 0;

    [[nodiscard]] bool ok() const {
        return n_negative_volumes == 0 && max_non_orthogonality < 70.0;
    }
};

[[nodiscard]] MeshQuality compute_quality(const Mesh& mesh);

// Multi-line, aligned report ending with "Status: OK" or "Status: FAILED".
[[nodiscard]] std::string to_string(const MeshQuality& q);

} // namespace cfd::mesh
