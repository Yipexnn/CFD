#include "main_window.hpp"

#include "case_editor.hpp"
#include "post_panel.hpp"
#include "render_view.hpp"
#include "residual_plot.hpp"
#include "solver_worker.hpp"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QDir>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QKeySequence>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>
#include <cfd/core/version.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/gmsh_reader.hpp>
#include <exception>

namespace cfd::gui {

namespace {

cases::CaseConfig default_case() {
    cases::CaseConfig c;
    c.name = "cavity";
    c.mesh.box.nx = 32;
    c.mesh.box.ny = 32;
    c.mesh.box.nz = 1;
    c.mesh.box.max = Vec3(1.0, 1.0, 1.0 / 32.0);
    c.mesh.box.patch_types[4] = mesh::PatchType::Symmetry;
    c.mesh.box.patch_types[5] = mesh::PatchType::Symmetry;
    c.solver.flow.nu = 0.01;
    c.solver.max_iterations = 3000;
    c.solver.tolerance = 1e-6;
    c.boundaries["ymax"] = {fv::FlowBoundary::Wall, Vec3(1.0, 0.0, 0.0), 0.0};
    c.output.directory = "results/cavity";
    return c;
}

std::vector<CaseEditor::PatchInfo> patches_of(const mesh::Mesh& m) {
    std::vector<CaseEditor::PatchInfo> out;
    for (const auto& p : m.patches()) {
        out.push_back({p.name, p.type});
    }
    return out;
}

} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    resize(1400, 900);
    view_ = new RenderView(this);
    setCentralWidget(view_);

    build_docks();
    build_actions();

    status_ = new QLabel("Ready", this);
    statusBar()->addPermanentWidget(status_, 1);

    worker_ = new SolverWorker();
    worker_->moveToThread(&worker_thread_);
    connect(&worker_thread_, &QThread::finished, worker_, &QObject::deleteLater);
    connect(worker_, &SolverWorker::started, this,
            [this](int cells) { log(QString("Running case with %1 cells").arg(cells)); });
    connect(worker_, &SolverWorker::progress, this, &MainWindow::on_progress);
    connect(worker_, &SolverWorker::snapshot_ready, this, &MainWindow::on_snapshot);
    connect(worker_, &SolverWorker::finished, this, &MainWindow::on_finished);
    worker_thread_.start();

    connect(post_, &PostPanel::settings_changed, this,
            [this](const DisplaySettings& s) { set_display(s); });
    connect(editor_, &CaseEditor::changed, this, [this] { update_title(); });

    new_case();
}

MainWindow::~MainWindow() {
    if (worker_ != nullptr) {
        worker_->stop();
    }
    worker_thread_.quit();
    worker_thread_.wait();
}

void MainWindow::build_docks() {
    editor_ = new CaseEditor(this);
    auto* editor_dock = new QDockWidget("Case", this);
    editor_dock->setObjectName("CaseDock");
    editor_dock->setWidget(editor_);
    addDockWidget(Qt::LeftDockWidgetArea, editor_dock);

    post_ = new PostPanel(this);
    auto* post_dock = new QDockWidget("Post-processing", this);
    post_dock->setObjectName("PostDock");
    post_dock->setWidget(post_);
    addDockWidget(Qt::RightDockWidgetArea, post_dock);

    plot_ = new ResidualPlot(this);
    auto* plot_dock = new QDockWidget("Residuals", this);
    plot_dock->setObjectName("ResidualDock");
    plot_dock->setWidget(plot_);
    addDockWidget(Qt::BottomDockWidgetArea, plot_dock);

    log_ = new QPlainTextEdit(this);
    log_->setReadOnly(true);
    log_->setMaximumBlockCount(5000);
    auto* log_dock = new QDockWidget("Log", this);
    log_dock->setObjectName("LogDock");
    log_dock->setWidget(log_);
    addDockWidget(Qt::BottomDockWidgetArea, log_dock);
    tabifyDockWidget(plot_dock, log_dock);
    plot_dock->raise();
}

void MainWindow::build_actions() {
    QMenu* file = menuBar()->addMenu("&File");
    QToolBar* tb = addToolBar("Main");
    tb->setObjectName("MainToolBar");

    auto add = [&](QMenu* menu, const QString& text, const QKeySequence& key, auto slot,
                   bool toolbar) {
        auto* a = new QAction(text, this);
        if (!key.isEmpty()) {
            a->setShortcut(key);
        }
        connect(a, &QAction::triggered, this, slot);
        menu->addAction(a);
        if (toolbar) {
            tb->addAction(a);
        }
        return a;
    };
    add(file, "&New case", QKeySequence::New, [this] { new_case(); }, true);
    add(file, "&Open case...", QKeySequence::Open, [this] { open_case_dialog(); }, true);
    add(file, "&Save case", QKeySequence::Save, [this] { save_case(); }, true);
    add(file, "Save case &as...", QKeySequence::SaveAs, [this] { save_case_dialog(); }, false);
    file->addSeparator();
    add(file, "Save &screenshot...", QKeySequence(), [this] { screenshot_dialog(); }, false);
    file->addSeparator();
    add(file, "&Quit", QKeySequence::Quit, [this] { close(); }, false);

    QMenu* run = menuBar()->addMenu("&Run");
    tb->addSeparator();
    mesh_action_ =
        add(run, "Generate &mesh", QKeySequence("Ctrl+M"), [this] { generate_mesh(); }, true);
    run_action_ = add(run, "&Run", QKeySequence("Ctrl+R"), [this] { start_run(); }, true);
    stop_action_ = add(run, "S&top", QKeySequence("Ctrl+T"), [this] { stop_run(); }, true);

    QMenu* view = menuBar()->addMenu("&View");
    tb->addSeparator();
    add(view, "&Reset camera", QKeySequence("Ctrl+0"), [this] { view_->reset_camera(); }, true);
    add(view, "View +X", QKeySequence(), [this] { view_->set_view_direction(0, true); }, false);
    add(view, "View +Y", QKeySequence(), [this] { view_->set_view_direction(1, true); }, false);
    add(view, "View +Z", QKeySequence(), [this] { view_->set_view_direction(2, true); }, true);
    view->addSeparator();
    for (QDockWidget* d : findChildren<QDockWidget*>()) {
        view->addAction(d->toggleViewAction());
    }

    QMenu* help = menuBar()->addMenu("&Help");
    add(
        help, "&About", QKeySequence(),
        [this] {
            QMessageBox::about(
                this, "About CFD",
                QString("CFD %1\nIncompressible finite-volume solver (SIMPLE/PISO, "
                        "k-omega SST).")
                    .arg(QString::fromStdString(std::string(core::k_version_string))));
        },
        false);
    update_actions();
}

void MainWindow::log(const QString& text) {
    log_->appendPlainText(text);
    status_->setText(text);
}

void MainWindow::update_title() {
    const QString name = case_file_.isEmpty() ? "untitled" : QFileInfo(case_file_).fileName();
    setWindowTitle(QString("CFD - %1").arg(name));
}

void MainWindow::update_actions() {
    if (run_action_ != nullptr) {
        run_action_->setEnabled(!running_);
        stop_action_->setEnabled(running_);
        mesh_action_->setEnabled(!running_);
    }
    if (editor_ != nullptr) {
        editor_->setEnabled(!running_);
    }
}

cases::CaseConfig MainWindow::current_config() const {
    cases::CaseConfig c = editor_->config();
    c.base_directory =
        case_file_.isEmpty()
            ? std::filesystem::path(QDir::currentPath().toStdString())
            : std::filesystem::path(QFileInfo(case_file_).absolutePath().toStdString());
    return c;
}

void MainWindow::new_case() {
    if (running_) {
        return;
    }
    case_file_.clear();
    const cases::CaseConfig c = default_case();
    editor_->set_patches(editor_->box_patches());
    editor_->set_config(c);
    editor_->set_patches(editor_->box_patches());
    update_title();
    log("New case");
    generate_mesh();
}

bool MainWindow::open_case(const QString& file) {
    try {
        cases::CaseConfig c = cases::load_case(file.toStdString());
        case_file_ = QFileInfo(file).absoluteFilePath();
        editor_->set_config(c);
        if (c.mesh.source == cases::MeshSource::Box) {
            editor_->set_patches(editor_->box_patches());
        }
        update_title();
        log(QString("Opened %1").arg(case_file_));
        return generate_mesh();
    } catch (const std::exception& e) {
        log(QString("Error: %1").arg(e.what()));
        QMessageBox::warning(this, "Open case", e.what());
        return false;
    }
}

void MainWindow::open_case_dialog() {
    const QString file =
        QFileDialog::getOpenFileName(this, "Open case", QString(), "Case files (*.json)");
    if (!file.isEmpty()) {
        open_case(file);
    }
}

bool MainWindow::save_case_as(const QString& file) {
    try {
        cases::save_case(editor_->config(), file.toStdString());
        case_file_ = QFileInfo(file).absoluteFilePath();
        update_title();
        log(QString("Saved %1").arg(case_file_));
        return true;
    } catch (const std::exception& e) {
        log(QString("Error: %1").arg(e.what()));
        QMessageBox::warning(this, "Save case", e.what());
        return false;
    }
}

void MainWindow::save_case() {
    if (case_file_.isEmpty()) {
        save_case_dialog();
    } else {
        save_case_as(case_file_);
    }
}

void MainWindow::save_case_dialog() {
    const QString file =
        QFileDialog::getSaveFileName(this, "Save case", case_file_, "Case files (*.json)");
    if (!file.isEmpty()) {
        save_case_as(file.endsWith(".json") ? file : file + ".json");
    }
}

void MainWindow::screenshot_dialog() {
    const QString file =
        QFileDialog::getSaveFileName(this, "Save screenshot", "view.png", "PNG images (*.png)");
    if (!file.isEmpty()) {
        save_view_screenshot(file);
    }
}

bool MainWindow::generate_mesh() {
    if (running_) {
        return false;
    }
    try {
        const cases::CaseConfig c = current_config();
        if (c.mesh.source == cases::MeshSource::Gmsh) {
            mesh_ = std::make_unique<mesh::Mesh>(mesh::read_gmsh(c.resolve(c.mesh.file)));
        } else {
            mesh_ = std::make_unique<mesh::Mesh>(mesh::make_box_mesh(c.mesh.box));
        }
        editor_->set_patches(patches_of(*mesh_));
        view_->set_mesh(*mesh_);
        view_->set_title(QString("Mesh: %1 cells, %2 patches")
                             .arg(mesh_->n_cells())
                             .arg(mesh_->patches().size()));
        fields_seen_ = false;
        log(QString("Mesh: %1 cells, %2 faces, %3 patches")
                .arg(mesh_->n_cells())
                .arg(mesh_->n_faces())
                .arg(mesh_->patches().size()));
        return true;
    } catch (const std::exception& e) {
        log(QString("Mesh error: %1").arg(e.what()));
        return false;
    }
}

void MainWindow::start_run() {
    if (running_ || !generate_mesh()) {
        return;
    }
    const cases::CaseConfig c = current_config();
    plot_->clear();
    plot_->set_tolerance(c.solver.mode == cases::SolverMode::Steady ? c.solver.tolerance : 0.0);
    running_ = true;
    update_actions();
    log(QString("Starting %1 run")
            .arg(c.solver.mode == cases::SolverMode::Steady ? "steady" : "transient"));
    SolverWorker* w = worker_;
    QMetaObject::invokeMethod(w, [w, c] { w->run(c); }, Qt::QueuedConnection);
}

void MainWindow::stop_run() {
    if (running_) {
        worker_->stop();
        log("Stop requested");
    }
}

bool MainWindow::is_running() const {
    return running_;
}

void MainWindow::on_progress(int step, double time, double ux, double uy, double uz, double p,
                             double k, double omega, double continuity) {
    plot_->add_point(step, ux, uy, uz, p, k, omega);
    status_->setText(QString("Step %1  t = %2  |  U %3  p %4  continuity %5")
                         .arg(step)
                         .arg(time, 0, 'g', 4)
                         .arg(std::max({ux, uy, uz}), 0, 'e', 2)
                         .arg(p, 0, 'e', 2)
                         .arg(continuity, 0, 'e', 2));
}

void MainWindow::on_snapshot() {
    const RunSnapshot s = worker_->latest_snapshot();
    if (!mesh_ || s.fields.u.size() != static_cast<std::size_t>(mesh_->n_cells())) {
        return;
    }
    view_->set_fields(s.fields);
    post_->set_fields(scalar_field_names(s.fields));
    if (!fields_seen_) {
        fields_seen_ = true;
        DisplaySettings d = display_;
        if (d.field.empty()) {
            d.field = "|U|";
        }
        post_->set_settings(d);
        set_display(d);
    }
    const auto [lo, hi] = view_->field_range();
    post_->set_range(lo, hi);
    view_->set_title(QString("Step %1%2")
                         .arg(s.step)
                         .arg(s.time > 0.0 ? QString("  t = %1").arg(s.time) : QString()));
}

void MainWindow::on_finished(bool ok, const QString& message, const QString& result_file) {
    running_ = false;
    update_actions();
    on_snapshot();
    log(message);
    if (!result_file.isEmpty()) {
        log(QString("Results written to %1").arg(result_file));
    }
    emit run_finished(ok);
}

void MainWindow::set_display(const DisplaySettings& settings) {
    display_ = settings;
    post_->set_settings(settings); // keeps the panel in sync (emits nothing)
    view_->set_display(settings);
    if (view_->has_fields()) {
        const auto [lo, hi] = view_->field_range();
        if (settings.auto_range) {
            post_->set_range(lo, hi);
        }
    }
}

bool MainWindow::save_view_screenshot(const QString& file) {
    const bool ok = view_->save_screenshot(file);
    log(ok ? QString("Screenshot saved to %1").arg(file) : QString("Screenshot failed"));
    return ok;
}

bool MainWindow::save_window_screenshot(const QString& file) {
    view_->render();
    const bool ok = grab().save(file);
    log(ok ? QString("Window screenshot saved to %1").arg(file) : QString("Screenshot failed"));
    return ok;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (running_) {
        worker_->stop();
    }
    event->accept();
}

} // namespace cfd::gui
