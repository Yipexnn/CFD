#pragma once

#include "display_settings.hpp"

#include <QMainWindow>
#include <QString>
#include <QThread>
#include <cfd/case/case_config.hpp>
#include <cfd/mesh/mesh.hpp>
#include <memory>

class QAction;
class QLabel;
class QPlainTextEdit;

namespace cfd::gui {

class CaseEditor;
class PostPanel;
class RenderView;
class ResidualPlot;
class SolverWorker;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;
    MainWindow(const MainWindow&) = delete;
    MainWindow& operator=(const MainWindow&) = delete;
    MainWindow(MainWindow&&) = delete;
    MainWindow& operator=(MainWindow&&) = delete;

    bool open_case(const QString& file);
    bool save_case_as(const QString& file);
    bool generate_mesh();
    void start_run();
    void stop_run();
    [[nodiscard]] bool is_running() const;

    void set_display(const DisplaySettings& settings);
    [[nodiscard]] DisplaySettings display() const { return display_; }
    bool save_view_screenshot(const QString& file);
    bool save_window_screenshot(const QString& file);

signals:
    void run_finished(bool ok);

protected:
    void closeEvent(class QCloseEvent* event) override;

private:
    void build_actions();
    void build_docks();
    void new_case();
    void open_case_dialog();
    void save_case();
    void save_case_dialog();
    void screenshot_dialog();
    void log(const QString& text);
    void update_title();
    void update_actions();
    [[nodiscard]] cases::CaseConfig current_config() const;

    void on_progress(int step, double time, double ux, double uy, double uz, double p, double k,
                     double omega, double continuity);
    void on_snapshot();
    void on_finished(bool ok, const QString& message, const QString& result_file);

    CaseEditor* editor_ = nullptr;
    PostPanel* post_ = nullptr;
    RenderView* view_ = nullptr;
    ResidualPlot* plot_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
    QLabel* status_ = nullptr;

    QAction* run_action_ = nullptr;
    QAction* stop_action_ = nullptr;
    QAction* mesh_action_ = nullptr;

    QThread worker_thread_;
    SolverWorker* worker_ = nullptr;
    bool running_ = false;
    bool fields_seen_ = false;

    QString case_file_;
    std::unique_ptr<mesh::Mesh> mesh_;
    DisplaySettings display_;
};

} // namespace cfd::gui
