#include <algorithm>
#include <cfd/mesh/quality.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>

namespace cfd::mesh {

namespace {

constexpr Scalar k_tiny = 1e-300;
// Relative tolerance below which the skewness intersection is ill-conditioned (d . Sf ~ 0).
constexpr Scalar k_skew_rel_tol = 1e-12;

[[nodiscard]] std::string format_line(const std::string& label, const std::string& value) {
    std::ostringstream os;
    os << std::left << std::setw(24) << label << value << '\n';
    return os.str();
}

[[nodiscard]] std::string format_scalar(Scalar v) {
    std::ostringstream os;
    os << std::setprecision(6) << v;
    return os.str();
}

} // namespace

MeshQuality compute_quality(const Mesh& mesh) {
    MeshQuality q;
    q.n_cells = mesh.n_cells();

    // Volumes.
    const auto& volumes = mesh.cell_volumes();
    if (!volumes.empty()) {
        q.min_volume = volumes.front();
        q.max_volume = volumes.front();
    }
    for (const Scalar v : volumes) {
        q.min_volume = std::min(q.min_volume, v);
        q.max_volume = std::max(q.max_volume, v);
        q.total_volume += v;
        if (v <= 0.0) {
            ++q.n_negative_volumes;
        }
    }

    // Internal-face metrics: non-orthogonality and skewness.
    const auto& owner = mesh.owner();
    const auto& neighbour = mesh.neighbour();
    const auto& cell_c = mesh.cell_centres();
    const auto& face_c = mesh.face_centres();
    const auto& face_sf = mesh.face_area_vectors();
    const auto n_internal = static_cast<std::size_t>(mesh.n_internal_faces());

    constexpr Scalar rad_to_deg = 180.0 / std::numbers::pi;
    Scalar sum_non_orth = 0.0;
    std::size_t n_non_orth = 0;

    for (std::size_t f = 0; f < n_internal; ++f) {
        const auto p = static_cast<std::size_t>(owner[f]);
        const auto n = static_cast<std::size_t>(neighbour[f]);
        const Vec3 d = cell_c[n] - cell_c[p];
        const Vec3& sf = face_sf[f];
        const Scalar d_norm = d.norm();
        const Scalar sf_norm = sf.norm();
        const Scalar d_dot_sf = d.dot(sf);

        if (d_norm > k_tiny && sf_norm > k_tiny) {
            const Scalar cosine = std::clamp(d_dot_sf / (d_norm * sf_norm), -1.0, 1.0);
            const Scalar angle = std::acos(cosine) * rad_to_deg;
            q.max_non_orthogonality = std::max(q.max_non_orthogonality, angle);
            sum_non_orth += angle;
            ++n_non_orth;
        }

        // Skewness: distance from face centre to the intersection of d with the face plane,
        // normalised by |d|. Skipped when d . Sf is ~0 (d nearly parallel to the face).
        if (d_norm > k_tiny && std::abs(d_dot_sf) > k_skew_rel_tol * d_norm * sf_norm) {
            const Vec3 cf_minus_cp = face_c[f] - cell_c[p];
            const Scalar t = cf_minus_cp.dot(sf) / d_dot_sf;
            const Vec3 fi = cell_c[p] + d * t;
            const Scalar skew = (face_c[f] - fi).norm() / d_norm;
            q.max_skewness = std::max(q.max_skewness, skew);
        }
    }

    if (n_non_orth > 0) {
        q.avg_non_orthogonality = sum_non_orth / static_cast<Scalar>(n_non_orth);
    }

    // Per-cell aspect ratio from the axis-aligned bounding box of all cell points.
    const auto& cell_offsets = mesh.cell_face_offsets();
    const auto& cell_faces = mesh.cell_faces();
    const auto& points = mesh.points();
    const auto n_cells = static_cast<std::size_t>(mesh.n_cells());

    for (std::size_t c = 0; c < n_cells; ++c) {
        const auto k_begin = static_cast<std::size_t>(cell_offsets[c]);
        const auto k_end = static_cast<std::size_t>(cell_offsets[c + 1]);

        Vec3 lo = Vec3::Constant(std::numeric_limits<Scalar>::max());
        Vec3 hi = Vec3::Constant(std::numeric_limits<Scalar>::lowest());
        bool has_points = false;

        for (std::size_t k = k_begin; k < k_end; ++k) {
            const auto f = cell_faces[k];
            for (const Label pt : mesh.face(f)) {
                const Vec3& x = points[static_cast<std::size_t>(pt)];
                lo = lo.cwiseMin(x);
                hi = hi.cwiseMax(x);
                has_points = true;
            }
        }
        if (!has_points) {
            continue;
        }

        const Vec3 extent = hi - lo;
        const Scalar ext_max = extent.maxCoeff();
        const Scalar ext_min = extent.minCoeff();
        if (ext_min <= k_tiny) {
            continue;
        }
        q.max_aspect_ratio = std::max(q.max_aspect_ratio, ext_max / ext_min);
    }

    return q;
}

std::string to_string(const MeshQuality& q) {
    std::ostringstream os;
    os << format_line("Cells:", std::to_string(q.n_cells));
    os << format_line("Min volume:", format_scalar(q.min_volume));
    os << format_line("Max volume:", format_scalar(q.max_volume));
    os << format_line("Total volume:", format_scalar(q.total_volume));
    os << format_line("Max non-orthogonality:", format_scalar(q.max_non_orthogonality) + " deg");
    os << format_line("Avg non-orthogonality:", format_scalar(q.avg_non_orthogonality) + " deg");
    os << format_line("Max skewness:", format_scalar(q.max_skewness));
    os << format_line("Max aspect ratio:", format_scalar(q.max_aspect_ratio));
    os << format_line("Negative volumes:", std::to_string(q.n_negative_volumes));
    os << "Status: " << (q.ok() ? "OK" : "FAILED") << '\n';
    return os.str();
}

} // namespace cfd::mesh
