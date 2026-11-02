#include <algorithm>
#include <cfd/mesh/vtu_writer.hpp>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cfd::mesh {

namespace {

constexpr int kVtkPolyhedron = 42;
constexpr std::size_t kValuesPerLine = 8;

std::size_t idx(Label i) {
    return static_cast<std::size_t>(i);
}

std::string xml_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        switch (c) {
        case '&':
            out += "&amp;";
            break;
        case '<':
            out += "&lt;";
            break;
        case '>':
            out += "&gt;";
            break;
        case '"':
            out += "&quot;";
            break;
        case '\'':
            out += "&apos;";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

template <typename T> void write_values(std::ofstream& os, const std::vector<T>& values) {
    os << "          ";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << (i % kValuesPerLine == 0 ? "\n          " : " ");
        }
        os << values[i];
    }
    os << '\n';
}

} // namespace

void write_vtu(const std::filesystem::path& file, const Mesh& mesh,
               const std::vector<CellScalarData>& scalars,
               const std::vector<CellVectorData>& vectors) {
    const Label n_cells = mesh.n_cells();
    const auto n_cells_sz = idx(n_cells);

    for (const auto& s : scalars) {
        if (s.values == nullptr || s.values->size() != n_cells_sz) {
            throw std::runtime_error("write_vtu: scalar '" + s.name +
                                     "' is missing or does not have one value per cell");
        }
    }
    for (const auto& v : vectors) {
        if (v.values == nullptr || v.values->size() != n_cells_sz) {
            throw std::runtime_error("write_vtu: vector '" + v.name +
                                     "' is missing or does not have one value per cell");
        }
    }

    const auto& points = mesh.points();
    const auto& cell_face_offsets = mesh.cell_face_offsets();
    const auto& cell_faces = mesh.cell_faces();
    const auto& owner = mesh.owner();

    std::vector<double> point_coords;
    point_coords.reserve(points.size() * 3);
    for (const auto& p : points) {
        point_coords.push_back(p(0));
        point_coords.push_back(p(1));
        point_coords.push_back(p(2));
    }

    std::vector<std::int64_t> connectivity;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> types;
    std::vector<std::int64_t> faces;
    std::vector<std::int64_t> face_offsets;
    offsets.reserve(n_cells_sz);
    types.reserve(n_cells_sz);
    face_offsets.reserve(n_cells_sz);

    for (Label c = 0; c < n_cells; ++c) {
        const Label first = cell_face_offsets[idx(c)];
        const Label last = cell_face_offsets[idx(c) + 1];

        std::vector<Label> cell_points;
        faces.push_back(static_cast<std::int64_t>(last - first));
        for (Label k = first; k < last; ++k) {
            const Label f = cell_faces[idx(k)];
            const auto face_pts = mesh.face(f);
            const bool is_owner = owner[idx(f)] == c;

            faces.push_back(static_cast<std::int64_t>(face_pts.size()));
            if (is_owner) {
                for (const Label p : face_pts) {
                    faces.push_back(static_cast<std::int64_t>(p));
                }
            } else {
                for (auto it = face_pts.rbegin(); it != face_pts.rend(); ++it) {
                    faces.push_back(static_cast<std::int64_t>(*it));
                }
            }
            cell_points.insert(cell_points.end(), face_pts.begin(), face_pts.end());
        }

        std::sort(cell_points.begin(), cell_points.end());
        cell_points.erase(std::unique(cell_points.begin(), cell_points.end()), cell_points.end());
        for (const Label p : cell_points) {
            connectivity.push_back(static_cast<std::int64_t>(p));
        }

        offsets.push_back(static_cast<std::int64_t>(connectivity.size()));
        types.push_back(kVtkPolyhedron);
        face_offsets.push_back(static_cast<std::int64_t>(faces.size()));
    }

    std::ofstream os(file, std::ios::out | std::ios::trunc);
    if (!os) {
        throw std::runtime_error("write_vtu: cannot open '" + file.string() + "' for writing");
    }
    os << std::setprecision(std::numeric_limits<double>::max_digits10);

    os << "<?xml version=\"1.0\"?>\n";
    os << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
          "header_type=\"UInt64\">\n";
    os << "  <UnstructuredGrid>\n";
    os << "    <Piece NumberOfPoints=\"" << points.size() << "\" NumberOfCells=\"" << n_cells_sz
       << "\">\n";

    os << "      <Points>\n";
    os << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    write_values(os, point_coords);
    os << "        </DataArray>\n";
    os << "      </Points>\n";

    os << "      <Cells>\n";
    os << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    write_values(os, connectivity);
    os << "        </DataArray>\n";
    os << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    write_values(os, offsets);
    os << "        </DataArray>\n";
    os << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    write_values(os, types);
    os << "        </DataArray>\n";
    os << "        <DataArray type=\"Int64\" Name=\"faces\" format=\"ascii\">\n";
    write_values(os, faces);
    os << "        </DataArray>\n";
    os << "        <DataArray type=\"Int64\" Name=\"faceoffsets\" format=\"ascii\">\n";
    write_values(os, face_offsets);
    os << "        </DataArray>\n";
    os << "      </Cells>\n";

    os << "      <CellData>\n";
    for (const auto& s : scalars) {
        os << "        <DataArray type=\"Float64\" Name=\"" << xml_escape(s.name)
           << "\" format=\"ascii\">\n";
        write_values(os, *s.values);
        os << "        </DataArray>\n";
    }
    for (const auto& v : vectors) {
        std::vector<double> flat;
        flat.reserve(v.values->size() * 3);
        for (const auto& value : *v.values) {
            flat.push_back(value(0));
            flat.push_back(value(1));
            flat.push_back(value(2));
        }
        os << "        <DataArray type=\"Float64\" Name=\"" << xml_escape(v.name)
           << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        write_values(os, flat);
        os << "        </DataArray>\n";
    }
    os << "      </CellData>\n";

    os << "    </Piece>\n";
    os << "  </UnstructuredGrid>\n";
    os << "</VTKFile>\n";

    os.flush();
    if (!os) {
        throw std::runtime_error("write_vtu: failed while writing '" + file.string() + "'");
    }
}

} // namespace cfd::mesh
