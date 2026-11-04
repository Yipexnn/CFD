#include "mpi_runner.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QStringList>
#include <filesystem>

namespace cfd::gui {

namespace {

#ifdef _WIN32
constexpr const char* k_exe_suffix = ".exe";
#else
constexpr const char* k_exe_suffix = "";
#endif

// Copy of the case with absolute paths, so it can be stored anywhere.
cases::CaseConfig absolute_paths(cases::CaseConfig c) {
    c.mesh.file = c.resolve(c.mesh.file);
    for (auto& body : c.mesh.geometry.bodies) {
        if (body.kind == geometry::Body::Kind::File) {
            body.file = c.resolve(body.file);
        }
    }
    c.output.directory = c.resolve(c.output.directory);
    if (c.restart) {
        c.restart = c.resolve(*c.restart);
    }
    return c;
}

} // namespace

MpiRunner::MpiRunner(QObject* parent) : QObject(parent) {
    process_.setProcessChannelMode(QProcess::MergedChannels);
    connect(&process_, &QProcess::readyReadStandardOutput, this, [this] {
        while (process_.canReadLine()) {
            const QString line = QString::fromLocal8Bit(process_.readLine()).trimmed();
            if (!line.isEmpty()) {
                if (line.startsWith("Converged") || line.startsWith("Not converged") ||
                    line.startsWith("Finished") || line.contains("error")) {
                    last_line_ = line;
                }
                emit output(line);
            }
        }
    });
    connect(&process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
            &MpiRunner::on_finished);
    connect(&process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) {
            timer_.stop();
            emit finished(false, "Failed to start mpiexec: " + process_.errorString(), {});
        }
    });
    timer_.setInterval(300);
    connect(&timer_, &QTimer::timeout, this, &MpiRunner::poll_residuals);
}

MpiRunner::~MpiRunner() {
    if (running()) {
        process_.kill();
        process_.waitForFinished(3000);
    }
}

QString MpiRunner::cfd_executable() {
    const QString exe =
        QDir(QCoreApplication::applicationDirPath()).filePath(QString("cfd") + k_exe_suffix);
    if (QFileInfo::exists(exe)) {
        return exe;
    }
#ifdef CFD_CLI_PATH
    if (QFileInfo::exists(QStringLiteral(CFD_CLI_PATH))) {
        return QStringLiteral(CFD_CLI_PATH); // development build tree
    }
#endif
    return QStandardPaths::findExecutable("cfd");
}

QString MpiRunner::mpiexec_executable() {
    const QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (const QString custom = env.value("CFD_MPIEXEC"); !custom.isEmpty()) {
        return custom;
    }
    if (const QString found = QStandardPaths::findExecutable("mpiexec"); !found.isEmpty()) {
        return found;
    }
    if (const QString msmpi = env.value("MSMPI_BIN"); !msmpi.isEmpty()) {
        const QString exe = QDir(msmpi).filePath("mpiexec.exe");
        if (QFileInfo::exists(exe)) {
            return exe;
        }
    }
    return {};
}

bool MpiRunner::start(const cases::CaseConfig& config, int processes, QString& error) {
    if (running()) {
        error = "A parallel run is already active";
        return false;
    }
    const QString cfd = cfd_executable();
    const QString mpiexec = mpiexec_executable();
    if (cfd.isEmpty()) {
        error = "cfd executable not found next to the GUI";
        return false;
    }
    if (mpiexec.isEmpty()) {
        error = "mpiexec not found (install MS-MPI on Windows / OpenMPI on Linux, or set "
                "CFD_MPIEXEC)";
        return false;
    }
    QProcess info;
    info.start(cfd, {"--info"});
    if (!info.waitForFinished(10000) ||
        !QString::fromLocal8Bit(info.readAllStandardOutput()).contains("MPI: yes")) {
        error = "cfd was built without MPI support (configure with -DCFD_ENABLE_MPI=ON)";
        return false;
    }

    const cases::CaseConfig c = absolute_paths(config);
    const std::filesystem::path dir = c.output.directory;
    try {
        std::filesystem::create_directories(dir);
        std::filesystem::remove(dir / "residuals.csv");
        cases::save_case(c, dir / "case_mpi.json");
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
    output_dir_ = QString::fromStdString(dir.string());
    result_file_ = QString::fromStdString((dir / (c.name + ".vtu")).string());
    residual_pos_ = 0;
    stop_requested_ = false;
    last_line_.clear();

    QStringList args{"-n", QString::number(processes)};
    if (const QString extra = QProcessEnvironment::systemEnvironment().value("CFD_MPIEXEC_FLAGS");
        !extra.isEmpty()) {
        args << extra.split(' ', Qt::SkipEmptyParts);
    }
    args << cfd << "run" << QString::fromStdString((dir / "case_mpi.json").string());
    process_.start(mpiexec, args);
    timer_.start();
    return true;
}

void MpiRunner::stop() {
    if (running()) {
        stop_requested_ = true;
        process_.kill(); // mpiexec terminates its processes
    }
}

void MpiRunner::poll_residuals() {
    QFile file(QDir(output_dir_).filePath("residuals.csv"));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text) || !file.seek(residual_pos_)) {
        return;
    }
    while (!file.atEnd()) {
        const QByteArray raw = file.readLine();
        if (!raw.endsWith('\n')) {
            break; // incomplete line: read again next time
        }
        residual_pos_ += raw.size();
        const QStringList v = QString::fromLatin1(raw).trimmed().split(',');
        bool ok = false;
        const int step = v.isEmpty() ? 0 : v[0].toInt(&ok);
        if (!ok || v.size() < 9) {
            continue; // header
        }
        emit progress(step, v[1].toDouble(), v[2].toDouble(), v[3].toDouble(), v[4].toDouble(),
                      v[5].toDouble(), v[6].toDouble(), v[7].toDouble(), v[8].toDouble(),
                      v.size() > 9 ? v[9].toDouble() : 0.0);
    }
}

void MpiRunner::on_finished(int exit_code, QProcess::ExitStatus status) {
    timer_.stop();
    poll_residuals();
    if (stop_requested_) {
        emit finished(false, "Parallel run stopped", {});
        return;
    }
    const bool ok = status == QProcess::NormalExit && (exit_code == 0 || exit_code == 2);
    const QString message =
        ok ? last_line_
           : QString("Parallel run failed (exit code %1): %2").arg(exit_code).arg(last_line_);
    emit finished(ok, message, ok ? result_file_ : QString());
}

} // namespace cfd::gui
