#pragma once

#include <cfd/mesh/mesh.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace cfd::mesh {

struct CellScalarData {
    std::string name;
    const std::vector<Scalar>* values = nullptr;
};

struct CellVectorData {
    std::string name;
    const std::vector<Vec3>* values = nullptr;
};

// Writes an ASCII VTK XML UnstructuredGrid (.vtu) with every cell as VTK_POLYHEDRON (type 42).
// Throws std::runtime_error if the file cannot be written or a data array has the wrong size.
void write_vtu(const std::filesystem::path& file, const Mesh& mesh,
               const std::vector<CellScalarData>& scalars = {},
               const std::vector<CellVectorData>& vectors = {});

// Writes a parallel .pvtu index for the given piece files (paths relative to the .pvtu) with
// the cell data arrays of write_vtu (scalars, then 3-component vectors).
void write_pvtu(const std::filesystem::path& file, const std::vector<std::string>& pieces,
                const std::vector<std::string>& scalar_names,
                const std::vector<std::string>& vector_names);

} // namespace cfd::mesh
