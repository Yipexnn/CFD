#pragma once

#include "vtk_convert.hpp"

#include <QObject>
#include <QString>
#include <atomic>
#include <cfd/case/case_runner.hpp>
#include <chrono>
#include <memory>
#include <mutex>

namespace cfd::gui {

// Snapshot of a running case, copied from the solver thread.
struct RunSnapshot {
    int step = 0;
    double time = 0.0;
    FieldSet fields;
};

// Runs a case in a worker thread (moveToThread). Progress and field snapshots are
// delivered through queued signals; stop() may be called from any thread.
class SolverWorker : public QObject {
    Q_OBJECT
public:
    explicit SolverWorker(QObject* parent = nullptr);

    void stop() { stop_requested_ = true; }
    [[nodiscard]] bool running() const { return running_; }

    // Latest field snapshot (thread safe).
    [[nodiscard]] RunSnapshot latest_snapshot() const;

    // Interval between field snapshots while running.
    void set_snapshot_interval(std::chrono::milliseconds interval) { interval_ = interval; }

public slots:
    // mesh: pre-built case mesh (e.g. the preview), or nullptr to build it from the config.
    void run(const cfd::cases::CaseConfig& config,
             const std::shared_ptr<const cfd::mesh::Mesh>& mesh = nullptr);

signals:
    void started(int n_cells);
    void progress(int step, double time, double ux, double uy, double uz, double p, double k,
                  double omega, double continuity);
    void snapshot_ready();
    void finished(bool ok, const QString& message, const QString& result_file);

private:
    void take_snapshot(const cases::CaseRunner& runner);

    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};
    std::chrono::milliseconds interval_{1000};
    mutable std::mutex mutex_;
    RunSnapshot snapshot_;
};

} // namespace cfd::gui
