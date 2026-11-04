#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QTimer>
#include <cfd/case/case_config.hpp>

namespace cfd::gui {

// Runs a case on several processes: `mpiexec -n N cfd run <case>` as a child process. The
// live residuals come from <output>/residuals.csv, the final fields from the checkpoint.
class MpiRunner : public QObject {
    Q_OBJECT
public:
    explicit MpiRunner(QObject* parent = nullptr);
    ~MpiRunner() override;
    MpiRunner(const MpiRunner&) = delete;
    MpiRunner& operator=(const MpiRunner&) = delete;
    MpiRunner(MpiRunner&&) = delete;
    MpiRunner& operator=(MpiRunner&&) = delete;

    // The cfd command-line program: next to the GUI, in the build tree, or on PATH.
    [[nodiscard]] static QString cfd_executable();
    // mpiexec: $CFD_MPIEXEC, PATH, or %MSMPI_BIN% (MS-MPI). Empty if not found.
    [[nodiscard]] static QString mpiexec_executable();

    // Writes <output>/case_mpi.json (absolute paths) and starts the run. Returns false with
    // a message if the programs are missing or cfd was built without MPI.
    bool start(const cases::CaseConfig& config, int processes, QString& error);
    void stop();
    [[nodiscard]] bool running() const { return process_.state() != QProcess::NotRunning; }
    [[nodiscard]] QString output_directory() const { return output_dir_; }

signals:
    void progress(int step, double time, double ux, double uy, double uz, double p, double k,
                  double omega, double continuity);
    void output(const QString& line);
    void finished(bool ok, const QString& message, const QString& result_file);

private:
    void poll_residuals();
    void on_finished(int exit_code, QProcess::ExitStatus status);

    QProcess process_;
    QTimer timer_;
    QString output_dir_;
    QString result_file_;
    QString last_line_;
    qint64 residual_pos_ = 0;
    bool stop_requested_ = false;
};

} // namespace cfd::gui
