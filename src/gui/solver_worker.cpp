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
    // Flow fields live on the fluid zone; the view shows the full mesh (zero in solids).
    const auto full = [&](const std::vector<double>& v) {
        return runner.to_full_cells(v.data(), 1);
    };
    const std::vector<core::Vec3> u_cells = runner.velocity();
    const std::vector<double> u = runner.to_full_cells(u_cells.data()->data(), 3);
    s.fields.u.resize(u.size() / 3);
    for (std::size_t c = 0; c < s.fields.u.size(); ++c) {
        s.fields.u[c] = core::Vec3(u[3 * c], u[3 * c + 1], u[3 * c + 2]);
    }
    s.fields.p = full(runner.pressure());
    if (const auto* cs = runner.compressible()) {
        s.fields.t = full(cs->temperature());
        s.fields.rho = full(cs->density());
        s.fields.mach = full(cs->mach());
        if (cs->settings().turbulence) {
            std::vector<Scalar> nut = cs->eddy_viscosity();
            const std::vector<Scalar> rho = cs->density();
            for (std::size_t c = 0; c < nut.size(); ++c) {
                nut[c] /= rho[c];
            }
            s.fields.k = full(cs->k());
            s.fields.omega = full(cs->omega());
            s.fields.nut = full(nut);
        }
    }
    if (const auto* mp = runner.multiphase()) {
        s.fields.alpha = full(mp->alpha());
    }
    if (const auto* sst = runner.sst()) {
        s.fields.k = full(sst->k());
        s.fields.omega = full(sst->omega());
        s.fields.nut = full(sst->nut());
        s.fields.y_plus = full(sst->y_plus());
    }
    if (const auto* energy = runner.energy()) {
        s.fields.t = energy->temperature();
        if (const auto* rad = runner.radiation()) {
            s.fields.g = rad->incident_radiation();
        }
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
        emit started(runner.full_mesh().n_cells());
        auto last = std::chrono::steady_clock::now();
        auto last_progress = last - std::chrono::seconds(1);
        const auto result = runner.run([&](int step, double time, const fv::Residuals& r) {
            const auto now = std::chrono::steady_clock::now();
            // At most ~100 progress signals per second (fast cases would flood the GUI).
            if (now - last_progress >= std::chrono::milliseconds(10)) {
                emit progress(step, time, r.u.x(), r.u.y(), r.u.z(), r.p, r.k, r.omega,
                              r.continuity, r.t);
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
