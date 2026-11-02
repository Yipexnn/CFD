#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <type_traits>

namespace cfd::core {

// Floating-point scalar used for all field values and geometry.
using Scalar = double;

// Signed 32-bit index type for mesh entities (cells, faces, points).
// Signed so that loops remain valid under MSVC OpenMP, which requires signed indices.
using Label = std::int32_t;

// Fixed-size 3D vector of Scalar, used for positions, velocities and face normals.
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

static_assert(sizeof(Scalar) == 8, "cfd::core::Scalar must be 64-bit");
static_assert(std::is_floating_point_v<Scalar>, "cfd::core::Scalar must be a floating-point type");
static_assert(sizeof(Label) == 4, "cfd::core::Label must be 32-bit");
static_assert(std::is_signed_v<Label>,
              "cfd::core::Label must be signed (required for MSVC OpenMP loops)");
static_assert(Vec3::RowsAtCompileTime == 3, "cfd::core::Vec3 must have exactly 3 rows");

} // namespace cfd::core
