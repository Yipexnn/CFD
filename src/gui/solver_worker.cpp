#include "solver_worker.hpp"

#include <exception>

namespace cfd::gui {

SolverWorker::SolverWorker(QObject* parent) : QObject(parent) {}

RunSnapshot SolverWorker::latest_snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

void SolverWorker::take_snapshot(const cases::CaseRunner& runner) {
    RunSnapshot s;
    s.step = runner.step();
    s.time = runner.time();
    s.fields.u = runner.flow().velocity();
    s.fields.p = runner.flow().pressure();
    if (const auto* sst = runner.sst()) {
        s.fields.k = sst->k();
        s.fields.omega = sst->omega();
        s.fields.nut = sst->nut();
        s.fields.y_plus = sst->y_plus();
    }
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        snapshot_ = std::move(s);
    }
    emit snapshot_ready();
}

void SolverWorker::run(const cases::CaseConfig& config,
                       const std::shared_ptr<const mesh::Mesh>& mesh) {
    stop_requested_ = false;
    running_ = true;
    try {
        cases::CaseRunner runner =
            mesh ? cases::CaseRunner(config, *mesh) : cases::CaseRunner(config);
        emit started(runner.mesh().n_cells());
        auto last = std::chrono::steady_clock::now();
        auto last_progress = last - std::chrono::seconds(1);
        const auto result = runner.run([&](int step, double time, const fv::Residuals& r) {
            const auto now = std::chrono::steady_clock::now();
            // At most ~100 progress signals per second (fast cases would flood the GUI).
            if (now - last_progress >= std::chrono::milliseconds(10)) {
                emit progress(step, time, r.u.x(), r.u.y(), r.u.z(), r.p, r.k, r.omega,
                              r.continuity);
                last_progress = now;
            }
            if (now - last >= interval_) {
                take_snapshot(runner);
                last = now;
            }
            return !stop_requested_.load();
        });
        take_snapshot(runner);
        running_ = false;
        QString message;
        if (result.stopped) {
            message = QString("Stopped after %1 steps").arg(result.steps);
        } else if (result.converged) {
            message =
                config.solver.mode == cases::SolverMode::Steady
                    ? QString("Converged after %1 iterations").arg(result.steps)
                    : QString("Finished %1 time steps (t = %2)").arg(result.steps).arg(result.time);
        } else {
            message = QString("Not converged after %1 iterations").arg(result.steps);
        }
        emit finished(true, message, QString::fromStdString(result.result_file.string()));
    } catch (const std::exception& e) {
        running_ = false;
        emit finished(false, QString("Error: %1").arg(e.what()), QString());
    }
}

} // namespace cfd::gui
