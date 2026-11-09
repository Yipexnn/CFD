#pragma once

#include <QWidget>
#include <array>
#include <cfd/case/case_config.hpp>
#include <cfd/mesh/mesh.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLineEdit;
class QSpinBox;
class QStackedWidget;
class QTableWidget;

namespace cfd::gui {

// Tabbed editor for a cases::CaseConfig (mesh, physics, boundaries, initial fields,
// solver, output). Fields not shown in the editor (name, base directory, restart) are
// kept from the last set_config() call.
class CaseEditor : public QWidget {
    Q_OBJECT
public:
    // A mesh patch as seen by the boundary table.
    struct PatchInfo {
        std::string name;
        mesh::PatchType type = mesh::PatchType::Patch;
    };

    explicit CaseEditor(QWidget* parent = nullptr);

    // Fills all widgets from cfg. Does not emit changed() or mesh_changed().
    void set_config(const cases::CaseConfig& cfg);
    // Current widget values merged into a copy of the stored config.
    [[nodiscard]] cases::CaseConfig config() const;

    // Rebuilds the boundary table with one row per patch. Values of patches that already
    // have a row or an entry in the stored config are kept; others get a default from
    // their patch type. Does not emit changed() or mesh_changed().
    void set_patches(const std::vector<PatchInfo>& patches);
    // The six box patches with the currently selected box patch types.
    [[nodiscard]] std::vector<PatchInfo> box_patches() const;

signals:
    // Emitted on any user edit.
    void changed();
    // Emitted when a mesh-tab input changes (source, box values, Gmsh path, box patch types).
    void mesh_changed();

private:
    [[nodiscard]] QWidget* build_mesh_tab();
    [[nodiscard]] QWidget* build_physics_tab();
    [[nodiscard]] QWidget* build_boundaries_tab();
    [[nodiscard]] QWidget* build_initial_tab();
    [[nodiscard]] QWidget* build_solver_tab();
    [[nodiscard]] QWidget* build_output_tab();

    void rebuild_boundary_table();
    [[nodiscard]] fv::BoundaryCondition read_boundary_row(int row) const;
    [[nodiscard]] std::map<std::string, fv::BoundaryCondition> boundaries_from_table() const;
    // Thermal condition of a table row (nullopt: default).
    [[nodiscard]] std::optional<fv::ThermalBoundary> read_thermal_row(int row) const;
    [[nodiscard]] std::map<std::string, fv::ThermalBoundary> thermal_from_table() const;

    void notify(bool mesh);
    void watch(QDoubleSpinBox* widget, bool mesh);
    void watch(QSpinBox* widget, bool mesh);
    void watch(QComboBox* widget, bool mesh);
    void watch(QLineEdit* widget, bool mesh);
    void watch(QCheckBox* widget, bool mesh);

    [[nodiscard]] QWidget* build_geometry_page(QWidget* parent);
    void add_body_row(const geometry::Body& body, const cases::SolidConfig* solid);
    void set_geometry(const geometry::GeometrySpec& spec);
    // Reads the geometry page; solid bodies are returned as solids (conjugate heat transfer).
    void read_geometry(geometry::GeometrySpec& spec, std::vector<cases::SolidConfig>& solids) const;

    // Mesh tab.
    QComboBox* source_combo_ = nullptr;
    QStackedWidget* mesh_stack_ = nullptr;
    std::array<QSpinBox*, 3> cells_{};
    std::array<QDoubleSpinBox*, 3> min_{};
    std::array<QDoubleSpinBox*, 3> max_{};
    std::array<QDoubleSpinBox*, 3> grading_{};
    QTableWidget* box_table_ = nullptr;
    std::array<QComboBox*, 6> box_type_combos_{};
    QLineEdit* gmsh_path_ = nullptr;
    // Geometry page (Gmsh auto meshing).
    std::array<QDoubleSpinBox*, 3> geo_min_{};
    std::array<QDoubleSpinBox*, 3> geo_max_{};
    QCheckBox* geo_two_d_ = nullptr;
    QCheckBox* geo_internal_ = nullptr;
    QCheckBox* geo_quads_ = nullptr;
    QLineEdit* geo_size_ = nullptr;
    QLineEdit* geo_size_near_ = nullptr;
    QLineEdit* geo_refine_ = nullptr;
    QSpinBox* geo_bl_layers_ = nullptr;
    QLineEdit* geo_bl_first_ = nullptr;
    QLineEdit* geo_bl_ratio_ = nullptr;
    std::array<QLineEdit*, 6> geo_patch_names_{};
    QTableWidget* bodies_table_ = nullptr;

    // Physics tab.
    QDoubleSpinBox* nu_ = nullptr;
    std::array<QDoubleSpinBox*, 3> body_force_{};
    QComboBox* turbulence_ = nullptr;
    QCheckBox* energy_ = nullptr;
    QDoubleSpinBox* prandtl_ = nullptr;
    QDoubleSpinBox* prandtl_t_ = nullptr;
    QLineEdit* beta_ = nullptr;
    QLineEdit* t_ref_ = nullptr;
    QLineEdit* nu_law_ = nullptr;
    QLineEdit* alpha_law_ = nullptr;
    QCheckBox* radiation_ = nullptr;
    QLineEdit* absorption_ = nullptr;
    QDoubleSpinBox* emissivity_ = nullptr;
    QCheckBox* compressible_ = nullptr;
    QDoubleSpinBox* gamma_ = nullptr;
    QLineEdit* r_gas_ = nullptr;
    QLineEdit* mu_ = nullptr;
    QLineEdit* cfl_ = nullptr;
    QCheckBox* sutherland_ = nullptr;
    QCheckBox* implicit_ = nullptr;
    QSpinBox* krylov_ = nullptr;
    QComboBox* limiter_ = nullptr;
    std::array<QDoubleSpinBox*, 3> gravity_{};

    // Boundaries tab.
    QTableWidget* boundary_table_ = nullptr;

    // Initial tab.
    std::array<QDoubleSpinBox*, 3> init_velocity_{};
    QLineEdit* init_k_ = nullptr;
    QLineEdit* init_omega_ = nullptr;

    // Solver tab.
    QComboBox* mode_ = nullptr;
    QSpinBox* max_iterations_ = nullptr;
    QLineEdit* tolerance_ = nullptr;
    QLineEdit* dt_ = nullptr;
    QLineEdit* end_time_ = nullptr;
    QDoubleSpinBox* alpha_u_ = nullptr;
    QDoubleSpinBox* alpha_p_ = nullptr;
    QComboBox* convection_ = nullptr;
    QComboBox* time_scheme_ = nullptr;
    QComboBox* device_ = nullptr;
    QComboBox* gradient_ = nullptr;
    QSpinBox* non_orth_correctors_ = nullptr;
    QSpinBox* piso_correctors_ = nullptr;

    // Output tab.
    QLineEdit* output_directory_ = nullptr;
    QSpinBox* write_interval_ = nullptr;
    QSpinBox* log_interval_ = nullptr;
    QSpinBox* checkpoint_interval_ = nullptr;

    cases::CaseConfig config_;
    std::vector<PatchInfo> patches_;
    bool updating_ = false; // true while widgets are set programmatically
};

} // namespace cfd::gui
