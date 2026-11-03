#include "case_editor.hpp"

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDoubleValidator>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QString>
#include <QStringList>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <cstddef>
#include <filesystem>
#include <utility>

namespace cfd::gui {

namespace {

constexpr std::array<const char*, 5> kBoxTypeNames{"wall", "inlet", "outlet", "symmetry", "patch"};
constexpr std::array<mesh::PatchType, 5> kBoxTypes{
    mesh::PatchType::Wall, mesh::PatchType::Inlet, mesh::PatchType::Outlet,
    mesh::PatchType::Symmetry, mesh::PatchType::Patch};

constexpr std::array<const char*, 5> kBoundaryTypeNames{"velocity_inlet", "wall", "pressure_outlet",
                                                        "symmetry", "empty"};
constexpr std::array<fv::FlowBoundary, 5> kBoundaryKinds{
    fv::FlowBoundary::VelocityInlet, fv::FlowBoundary::Wall, fv::FlowBoundary::PressureOutlet,
    fv::FlowBoundary::Symmetry, fv::FlowBoundary::Empty};

constexpr std::array<const char*, 3> kConvectionNames{"upwind", "linear_upwind",
                                                      "limited_linear_upwind"};
constexpr std::array<fv::ConvectionScheme, 3> kConvections{
    fv::ConvectionScheme::Upwind, fv::ConvectionScheme::LinearUpwind,
    fv::ConvectionScheme::LimitedLinearUpwind};

// Restores the updating flag on scope exit (nesting-safe).
class UpdateScope {
public:
    explicit UpdateScope(bool& flag) : flag_(flag), previous_(flag) { flag_ = true; }
    ~UpdateScope() { flag_ = previous_; }
    UpdateScope(const UpdateScope&) = delete;
    UpdateScope& operator=(const UpdateScope&) = delete;

private:
    bool& flag_;
    bool previous_;
};

template <typename T, std::size_t N> int index_of(const std::array<T, N>& values, const T& value) {
    for (std::size_t i = 0; i < N; ++i) {
        if (values[i] == value) {
            return static_cast<int>(i);
        }
    }
    return 0;
}

std::size_t current_index(const QComboBox* combo) {
    const int i = combo->currentIndex();
    return i < 0 ? 0U : static_cast<std::size_t>(i);
}

template <std::size_t N> void add_items(QComboBox* combo, const std::array<const char*, N>& names) {
    for (const char* name : names) {
        combo->addItem(QString::fromLatin1(name));
    }
}

QString to_qstring(const std::filesystem::path& p) {
    return QString::fromStdWString(p.generic_wstring());
}

std::filesystem::path to_path(const QString& text) {
    return std::filesystem::path(text.toStdWString());
}

// Parses a line edit as a number; falls back to the stored value while the text is not
// a valid number (e.g. mid-edit "1e-").
double parse_or(const QLineEdit* edit, double fallback) {
    bool ok = false;
    const double value = QLocale::c().toDouble(edit->text(), &ok);
    return ok ? value : fallback;
}

QLineEdit* make_number_edit(QWidget* parent, double lo, double hi, double value) {
    auto* edit = new QLineEdit(parent);
    auto* validator = new QDoubleValidator(lo, hi, 300, edit);
    validator->setNotation(QDoubleValidator::ScientificNotation);
    validator->setLocale(QLocale::c());
    edit->setValidator(validator);
    edit->setText(QString::number(value, 'g', 17));
    return edit;
}

QDoubleSpinBox* make_double(QWidget* parent, double lo, double hi, int decimals, double value) {
    auto* spin = new QDoubleSpinBox(parent);
    spin->setDecimals(decimals);
    spin->setRange(lo, hi);
    spin->setValue(value);
    return spin;
}

QSpinBox* make_int(QWidget* parent, int lo, int hi, int value) {
    auto* spin = new QSpinBox(parent);
    spin->setRange(lo, hi);
    spin->setValue(value);
    return spin;
}

// Three double spin boxes in a row. Returns the row container.
QWidget* make_vec_row(QWidget* parent, std::array<QDoubleSpinBox*, 3>& out, double lo, double hi,
                      int decimals) {
    auto* row = new QWidget(parent);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    for (QDoubleSpinBox*& spin : out) {
        spin = make_double(row, lo, hi, decimals, 0.0);
        layout->addWidget(spin);
    }
    return row;
}

void set_vec(const std::array<QDoubleSpinBox*, 3>& widgets, const core::Vec3& v) {
    widgets[0]->setValue(v.x());
    widgets[1]->setValue(v.y());
    widgets[2]->setValue(v.z());
}

core::Vec3 get_vec(const std::array<QDoubleSpinBox*, 3>& widgets) {
    return core::Vec3(widgets[0]->value(), widgets[1]->value(), widgets[2]->value());
}

mesh::PatchType box_type_at(const QComboBox* combo) {
    return kBoxTypes[current_index(combo)];
}

// Default boundary condition for a mesh patch type.
fv::BoundaryCondition default_boundary(mesh::PatchType type) {
    fv::BoundaryCondition bc;
    switch (type) {
    case mesh::PatchType::Inlet:
        bc.kind = fv::FlowBoundary::VelocityInlet;
        break;
    case mesh::PatchType::Outlet:
        bc.kind = fv::FlowBoundary::PressureOutlet;
        break;
    case mesh::PatchType::Symmetry:
        bc.kind = fv::FlowBoundary::Symmetry;
        break;
    case mesh::PatchType::Wall:
    case mesh::PatchType::Patch:
        bc.kind = fv::FlowBoundary::Wall;
        break;
    }
    return bc;
}

double table_double(const QTableWidget* table, int row, int col, double fallback) {
    const auto* spin = qobject_cast<QDoubleSpinBox*>(table->cellWidget(row, col));
    return spin ? spin->value() : fallback;
}

} // namespace

CaseEditor::CaseEditor(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    auto* tabs = new QTabWidget(this);
    root->addWidget(tabs);

    tabs->addTab(build_mesh_tab(), QStringLiteral("Mesh"));
    tabs->addTab(build_physics_tab(), QStringLiteral("Physics"));
    tabs->addTab(build_boundaries_tab(), QStringLiteral("Boundaries"));
    tabs->addTab(build_initial_tab(), QStringLiteral("Initial"));
    tabs->addTab(build_solver_tab(), QStringLiteral("Solver"));
    tabs->addTab(build_output_tab(), QStringLiteral("Output"));

    set_config(cases::CaseConfig{});
}

QWidget* CaseEditor::build_mesh_tab() {
    auto* page = new QWidget(this);
    auto* layout = new QVBoxLayout(page);

    source_combo_ = new QComboBox(page);
    source_combo_->addItem(QStringLiteral("Box"));
    source_combo_->addItem(QStringLiteral("Gmsh file"));
    auto* form = new QFormLayout;
    form->addRow(QStringLiteral("Source"), source_combo_);
    layout->addLayout(form);

    mesh_stack_ = new QStackedWidget(page);
    layout->addWidget(mesh_stack_);
    connect(source_combo_, QOverload<int>::of(&QComboBox::currentIndexChanged), mesh_stack_,
            &QStackedWidget::setCurrentIndex);
    watch(source_combo_, true);

    // Box page.
    auto* box_page = new QWidget(mesh_stack_);
    auto* box_layout = new QVBoxLayout(box_page);
    auto* grid = new QGridLayout;
    box_layout->addLayout(grid);

    const char* const axes[3] = {"x", "y", "z"};
    for (std::size_t i = 0; i < 3; ++i) {
        grid->addWidget(new QLabel(QString::fromLatin1(axes[i]), box_page), 0,
                        static_cast<int>(i) + 1);
    }

    grid->addWidget(new QLabel(QStringLiteral("Cells"), box_page), 1, 0);
    grid->addWidget(new QLabel(QStringLiteral("Min"), box_page), 2, 0);
    grid->addWidget(new QLabel(QStringLiteral("Max"), box_page), 3, 0);
    grid->addWidget(new QLabel(QStringLiteral("Grading"), box_page), 4, 0);

    for (std::size_t i = 0; i < 3; ++i) {
        const int col = static_cast<int>(i) + 1;
        cells_[i] = make_int(box_page, 1, 100000, 1);
        min_[i] = make_double(box_page, -1e9, 1e9, 6, 0.0);
        max_[i] = make_double(box_page, -1e9, 1e9, 6, 1.0);
        grading_[i] = make_double(box_page, 1e-6, 1e6, 6, 1.0);
        grid->addWidget(cells_[i], 1, col);
        grid->addWidget(min_[i], 2, col);
        grid->addWidget(max_[i], 3, col);
        grid->addWidget(grading_[i], 4, col);
        watch(cells_[i], true);
        watch(min_[i], true);
        watch(max_[i], true);
        watch(grading_[i], true);
    }

    auto* patch_group = new QGroupBox(QStringLiteral("Box patch types"), box_page);
    auto* patch_layout = new QVBoxLayout(patch_group);
    box_table_ = new QTableWidget(6, 1, patch_group);
    box_table_->setHorizontalHeaderLabels(QStringList{QStringLiteral("type")});
    box_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    for (std::size_t i = 0; i < box_type_combos_.size(); ++i) {
        auto* combo = new QComboBox(box_table_);
        add_items(combo, kBoxTypeNames);
        box_type_combos_[i] = combo;
        box_table_->setCellWidget(static_cast<int>(i), 0, combo);
        watch(combo, true);
    }
    patch_layout->addWidget(box_table_);
    box_layout->addWidget(patch_group);
    mesh_stack_->addWidget(box_page);

    // Gmsh page.
    auto* gmsh_page = new QWidget(mesh_stack_);
    auto* gmsh_layout = new QVBoxLayout(gmsh_page);
    auto* file_row = new QHBoxLayout;
    gmsh_path_ = new QLineEdit(gmsh_page);
    auto* browse = new QPushButton(QStringLiteral("Browse..."), gmsh_page);
    file_row->addWidget(gmsh_path_);
    file_row->addWidget(browse);
    gmsh_layout->addLayout(file_row);
    gmsh_layout->addStretch();
    mesh_stack_->addWidget(gmsh_page);
    watch(gmsh_path_, true);

    connect(browse, &QPushButton::clicked, this, [this] {
        const QString file =
            QFileDialog::getOpenFileName(this, QStringLiteral("Select Gmsh mesh"),
                                         gmsh_path_->text(), QStringLiteral("Gmsh mesh (*.msh)"));
        if (!file.isEmpty()) {
            gmsh_path_->setText(file);
        }
    });

    layout->addStretch();
    return page;
}

QWidget* CaseEditor::build_physics_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);

    nu_ = make_double(page, 1e-10, 1e3, 10, 1e-3);
    form->addRow(QStringLiteral("Kinematic viscosity nu [m^2/s]"), nu_);
    watch(nu_, false);

    form->addRow(QStringLiteral("Body force [m/s^2]"),
                 make_vec_row(page, body_force_, -1e9, 1e9, 6));
    for (QDoubleSpinBox* spin : body_force_) {
        watch(spin, false);
    }

    turbulence_ = new QComboBox(page);
    turbulence_->addItem(QStringLiteral("Laminar"));
    turbulence_->addItem(QStringLiteral("k-omega SST"));
    form->addRow(QStringLiteral("Turbulence"), turbulence_);
    watch(turbulence_, false);
    return page;
}

QWidget* CaseEditor::build_boundaries_tab() {
    auto* page = new QWidget(this);
    auto* layout = new QVBoxLayout(page);
    boundary_table_ = new QTableWidget(0, 8, page);
    boundary_table_->setHorizontalHeaderLabels(
        QStringList{QStringLiteral("Patch"), QStringLiteral("Type"), QStringLiteral("Ux"),
                    QStringLiteral("Uy"), QStringLiteral("Uz"), QStringLiteral("p"),
                    QStringLiteral("Turb. intensity"), QStringLiteral("Visc. ratio")});
    boundary_table_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    boundary_table_->setSelectionMode(QAbstractItemView::NoSelection);
    layout->addWidget(boundary_table_);
    return page;
}

QWidget* CaseEditor::build_initial_tab() {
    auto* page = new QWidget(this);
    auto* layout = new QVBoxLayout(page);
    auto* form = new QFormLayout;
    layout->addLayout(form);

    form->addRow(QStringLiteral("Velocity [m/s]"),
                 make_vec_row(page, init_velocity_, -1e6, 1e6, 6));
    for (QDoubleSpinBox* spin : init_velocity_) {
        watch(spin, false);
    }

    init_k_ = make_number_edit(page, -1e300, 1e300, -1.0);
    init_omega_ = make_number_edit(page, -1e300, 1e300, -1.0);
    form->addRow(QStringLiteral("k [m^2/s^2]"), init_k_);
    form->addRow(QStringLiteral("omega [1/s]"), init_omega_);
    watch(init_k_, false);
    watch(init_omega_, false);

    auto* hint = new QLabel(
        QStringLiteral("Values <= 0 are derived automatically by the turbulence model."), page);
    hint->setWordWrap(true);
    layout->addWidget(hint);
    layout->addStretch();
    return page;
}

QWidget* CaseEditor::build_solver_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);

    mode_ = new QComboBox(page);
    mode_->addItem(QStringLiteral("Steady"));
    mode_->addItem(QStringLiteral("Transient"));
    form->addRow(QStringLiteral("Mode"), mode_);
    watch(mode_, false);

    max_iterations_ = make_int(page, 1, 10000000, 1000);
    form->addRow(QStringLiteral("Max iterations"), max_iterations_);
    watch(max_iterations_, false);

    tolerance_ = make_number_edit(page, 0.0, 1e300, 1e-6);
    dt_ = make_number_edit(page, 0.0, 1e300, 1e-3);
    end_time_ = make_number_edit(page, 0.0, 1e300, 1.0);
    form->addRow(QStringLiteral("Tolerance"), tolerance_);
    form->addRow(QStringLiteral("dt [s]"), dt_);
    form->addRow(QStringLiteral("End time [s]"), end_time_);
    watch(tolerance_, false);
    watch(dt_, false);
    watch(end_time_, false);

    alpha_u_ = make_double(page, 0.01, 1.0, 3, 0.7);
    alpha_p_ = make_double(page, 0.01, 1.0, 3, 0.3);
    form->addRow(QStringLiteral("alpha_u"), alpha_u_);
    form->addRow(QStringLiteral("alpha_p"), alpha_p_);
    watch(alpha_u_, false);
    watch(alpha_p_, false);

    convection_ = new QComboBox(page);
    add_items(convection_, kConvectionNames);
    form->addRow(QStringLiteral("Convection"), convection_);
    watch(convection_, false);

    non_orth_correctors_ = make_int(page, 0, 10, 0);
    piso_correctors_ = make_int(page, 1, 10, 2);
    form->addRow(QStringLiteral("Non-orthogonal correctors"), non_orth_correctors_);
    form->addRow(QStringLiteral("PISO correctors"), piso_correctors_);
    watch(non_orth_correctors_, false);
    watch(piso_correctors_, false);
    return page;
}

QWidget* CaseEditor::build_output_tab() {
    auto* page = new QWidget(this);
    auto* form = new QFormLayout(page);

    output_directory_ = new QLineEdit(page);
    form->addRow(QStringLiteral("Directory"), output_directory_);
    watch(output_directory_, false);

    write_interval_ = make_int(page, 0, 10000000, 0);
    log_interval_ = make_int(page, 0, 10000000, 10);
    checkpoint_interval_ = make_int(page, 0, 10000000, 0);
    form->addRow(QStringLiteral("Write interval"), write_interval_);
    form->addRow(QStringLiteral("Log interval"), log_interval_);
    form->addRow(QStringLiteral("Checkpoint interval"), checkpoint_interval_);
    watch(write_interval_, false);
    watch(log_interval_, false);
    watch(checkpoint_interval_, false);
    return page;
}

void CaseEditor::set_config(const cases::CaseConfig& cfg) {
    const UpdateScope scope(updating_);
    config_ = cfg;
    const mesh::BoxSpec& box = config_.mesh.box;

    // Mesh.
    source_combo_->setCurrentIndex(config_.mesh.source == cases::MeshSource::Gmsh ? 1 : 0);
    cells_[0]->setValue(box.nx);
    cells_[1]->setValue(box.ny);
    cells_[2]->setValue(box.nz);
    set_vec(min_, box.min);
    set_vec(max_, box.max);
    set_vec(grading_, box.grading);

    QStringList labels;
    for (const std::string& name : box.patch_names) {
        labels << QString::fromStdString(name);
    }
    box_table_->setVerticalHeaderLabels(labels);
    for (std::size_t i = 0; i < box_type_combos_.size(); ++i) {
        box_type_combos_[i]->setCurrentIndex(index_of(kBoxTypes, box.patch_types[i]));
    }
    gmsh_path_->setText(to_qstring(config_.mesh.file));

    // Physics.
    const fv::FlowSettings& flow = config_.solver.flow;
    nu_->setValue(flow.nu);
    set_vec(body_force_, flow.body_force);
    turbulence_->setCurrentIndex(config_.turbulence == cases::TurbulenceKind::KOmegaSst ? 1 : 0);

    // Boundaries.
    rebuild_boundary_table();

    // Initial.
    set_vec(init_velocity_, config_.initial.velocity);
    init_k_->setText(QString::number(config_.initial.k, 'g', 17));
    init_omega_->setText(QString::number(config_.initial.omega, 'g', 17));

    // Solver.
    const cases::SolverConfig& solver = config_.solver;
    mode_->setCurrentIndex(solver.mode == cases::SolverMode::Transient ? 1 : 0);
    max_iterations_->setValue(solver.max_iterations);
    tolerance_->setText(QString::number(solver.tolerance, 'g', 17));
    dt_->setText(QString::number(solver.dt, 'g', 17));
    end_time_->setText(QString::number(solver.end_time, 'g', 17));
    alpha_u_->setValue(flow.alpha_u);
    alpha_p_->setValue(flow.alpha_p);
    convection_->setCurrentIndex(index_of(kConvections, flow.convection));
    non_orth_correctors_->setValue(flow.n_non_orth_correctors);
    piso_correctors_->setValue(flow.n_piso_correctors);

    // Output.
    const cases::OutputConfig& output = config_.output;
    output_directory_->setText(to_qstring(output.directory));
    write_interval_->setValue(output.write_interval);
    log_interval_->setValue(output.log_interval);
    checkpoint_interval_->setValue(output.checkpoint_interval);
}

cases::CaseConfig CaseEditor::config() const {
    cases::CaseConfig cfg = config_;

    // Mesh.
    cfg.mesh.source =
        source_combo_->currentIndex() == 1 ? cases::MeshSource::Gmsh : cases::MeshSource::Box;
    cfg.mesh.file = to_path(gmsh_path_->text());
    mesh::BoxSpec& box = cfg.mesh.box;
    box.nx = cells_[0]->value();
    box.ny = cells_[1]->value();
    box.nz = cells_[2]->value();
    box.min = get_vec(min_);
    box.max = get_vec(max_);
    box.grading = get_vec(grading_);
    for (std::size_t i = 0; i < box_type_combos_.size(); ++i) {
        box.patch_types[i] = box_type_at(box_type_combos_[i]);
    }

    // Physics.
    cfg.turbulence = turbulence_->currentIndex() == 1 ? cases::TurbulenceKind::KOmegaSst
                                                      : cases::TurbulenceKind::Laminar;
    fv::FlowSettings& flow = cfg.solver.flow;
    flow.nu = nu_->value();
    flow.body_force = get_vec(body_force_);

    // Boundaries.
    cfg.boundaries = patches_.empty() ? config_.boundaries : boundaries_from_table();

    // Initial.
    cfg.initial.velocity = get_vec(init_velocity_);
    cfg.initial.k = parse_or(init_k_, config_.initial.k);
    cfg.initial.omega = parse_or(init_omega_, config_.initial.omega);

    // Solver.
    cases::SolverConfig& solver = cfg.solver;
    solver.mode =
        mode_->currentIndex() == 1 ? cases::SolverMode::Transient : cases::SolverMode::Steady;
    solver.max_iterations = max_iterations_->value();
    solver.tolerance = parse_or(tolerance_, config_.solver.tolerance);
    solver.dt = parse_or(dt_, config_.solver.dt);
    solver.end_time = parse_or(end_time_, config_.solver.end_time);
    flow.alpha_u = alpha_u_->value();
    flow.alpha_p = alpha_p_->value();
    flow.convection = kConvections[current_index(convection_)];
    flow.n_non_orth_correctors = non_orth_correctors_->value();
    flow.n_piso_correctors = piso_correctors_->value();

    // Output.
    cases::OutputConfig& output = cfg.output;
    output.directory = to_path(output_directory_->text());
    output.write_interval = write_interval_->value();
    output.log_interval = log_interval_->value();
    output.checkpoint_interval = checkpoint_interval_->value();

    return cfg;
}

void CaseEditor::set_patches(const std::vector<PatchInfo>& patches) {
    // Keep the values currently in the table before the rows are rebuilt. Rows that only
    // show the default for their (possibly outdated) patch type are not stored, so they
    // pick up the default of the new patch type instead.
    for (std::size_t row = 0; row < patches_.size(); ++row) {
        const fv::BoundaryCondition bc = read_boundary_row(static_cast<int>(row));
        const fv::BoundaryCondition def = default_boundary(patches_[row].type);
        const bool is_default =
            bc.kind == def.kind && (bc.velocity - def.velocity).norm() < 1e-14 &&
            bc.pressure == def.pressure && bc.turbulence_intensity == def.turbulence_intensity &&
            bc.viscosity_ratio == def.viscosity_ratio;
        const bool explicit_entry = config_.boundaries.count(patches_[row].name) > 0;
        if (explicit_entry || !is_default) {
            config_.boundaries[patches_[row].name] = bc;
        }
    }
    patches_ = patches;
    rebuild_boundary_table();
}

std::vector<CaseEditor::PatchInfo> CaseEditor::box_patches() const {
    std::vector<PatchInfo> patches;
    patches.reserve(box_type_combos_.size());
    for (std::size_t i = 0; i < box_type_combos_.size(); ++i) {
        patches.push_back(
            PatchInfo{config_.mesh.box.patch_names[i], box_type_at(box_type_combos_[i])});
    }
    return patches;
}

void CaseEditor::rebuild_boundary_table() {
    const UpdateScope scope(updating_);
    boundary_table_->setRowCount(0);
    boundary_table_->setRowCount(static_cast<int>(patches_.size()));

    for (std::size_t row = 0; row < patches_.size(); ++row) {
        const int r = static_cast<int>(row);
        const PatchInfo& patch = patches_[row];

        auto* name = new QTableWidgetItem(QString::fromStdString(patch.name));
        name->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        boundary_table_->setItem(r, 0, name);

        const auto found = config_.boundaries.find(patch.name);
        const fv::BoundaryCondition bc =
            found != config_.boundaries.end() ? found->second : default_boundary(patch.type);

        auto* type = new QComboBox(boundary_table_);
        add_items(type, kBoundaryTypeNames);
        type->setCurrentIndex(index_of(kBoundaryKinds, bc.kind));
        watch(type, false);
        boundary_table_->setCellWidget(r, 1, type);

        const std::array<double, 3> velocity{bc.velocity.x(), bc.velocity.y(), bc.velocity.z()};
        for (std::size_t k = 0; k < velocity.size(); ++k) {
            QDoubleSpinBox* spin = make_double(boundary_table_, -1e6, 1e6, 6, velocity[k]);
            watch(spin, false);
            boundary_table_->setCellWidget(r, static_cast<int>(k) + 2, spin);
        }
        QDoubleSpinBox* pressure = make_double(boundary_table_, -1e9, 1e9, 6, bc.pressure);
        QDoubleSpinBox* intensity =
            make_double(boundary_table_, 0.0, 1e3, 6, bc.turbulence_intensity);
        QDoubleSpinBox* viscosity = make_double(boundary_table_, 0.0, 1e9, 6, bc.viscosity_ratio);
        watch(pressure, false);
        watch(intensity, false);
        watch(viscosity, false);
        boundary_table_->setCellWidget(r, 5, pressure);
        boundary_table_->setCellWidget(r, 6, intensity);
        boundary_table_->setCellWidget(r, 7, viscosity);
    }
}

fv::BoundaryCondition CaseEditor::read_boundary_row(int row) const {
    const std::size_t index = static_cast<std::size_t>(row);
    fv::BoundaryCondition bc = default_boundary(patches_[index].type);
    if (const auto* type = qobject_cast<QComboBox*>(boundary_table_->cellWidget(row, 1))) {
        bc.kind = kBoundaryKinds[current_index(type)];
    }
    bc.velocity = core::Vec3(table_double(boundary_table_, row, 2, bc.velocity.x()),
                             table_double(boundary_table_, row, 3, bc.velocity.y()),
                             table_double(boundary_table_, row, 4, bc.velocity.z()));
    bc.pressure = table_double(boundary_table_, row, 5, bc.pressure);
    bc.turbulence_intensity = table_double(boundary_table_, row, 6, bc.turbulence_intensity);
    bc.viscosity_ratio = table_double(boundary_table_, row, 7, bc.viscosity_ratio);
    return bc;
}

std::map<std::string, fv::BoundaryCondition> CaseEditor::boundaries_from_table() const {
    std::map<std::string, fv::BoundaryCondition> boundaries;
    for (std::size_t row = 0; row < patches_.size(); ++row) {
        boundaries[patches_[row].name] = read_boundary_row(static_cast<int>(row));
    }
    return boundaries;
}

void CaseEditor::notify(bool mesh) {
    if (updating_) {
        return;
    }
    if (mesh) {
        emit mesh_changed();
    }
    emit changed();
}

void CaseEditor::watch(QDoubleSpinBox* widget, bool mesh) {
    connect(widget, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this, mesh](double) { notify(mesh); });
}

void CaseEditor::watch(QSpinBox* widget, bool mesh) {
    connect(widget, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this, mesh](int) { notify(mesh); });
}

void CaseEditor::watch(QComboBox* widget, bool mesh) {
    connect(widget, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this, mesh](int) { notify(mesh); });
}

void CaseEditor::watch(QLineEdit* widget, bool mesh) {
    connect(widget, &QLineEdit::textChanged, this, [this, mesh](const QString&) { notify(mesh); });
}

} // namespace cfd::gui
