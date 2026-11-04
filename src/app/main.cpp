#include <cfd/case/case_config.hpp>
#include <cfd/case/case_runner.hpp>
#include <cfd/core/build_info.hpp>
#include <cfd/core/version.hpp>
#include <cfd/fv/flow_solver.hpp>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/mesh/gmsh_reader.hpp>
#include <cfd/mesh/mesh.hpp>
#include <cfd/mesh/quality.hpp>
#include <cfd/mesh/vtu_writer.hpp>
#include <cfd/parallel/comm.hpp>
#include <charconv>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void print_usage(std::ostream& os) {
    os << "Usage: cfd <command>\n"
       << "  --version                  Print version\n"
       << "  --info                     Print build information\n"
       << "  --help                     Show this help\n"
       << "  mesh-box <nx> <ny> <nz>    Create a unit box mesh and print statistics\n"
       << "  check-mesh <file.msh>      Read a Gmsh 4.1 mesh and print statistics\n"
       << "  run-cavity <n> <Re>        Lid-driven cavity demo, writes cavity.vtu\n"
       << "  run <case.json>            Run a case file (see docs/case_format.md)\n"
       << "                             MPI: mpiexec -n <N> cfd run <case.json>\n";
}

// Parses a strictly positive integer. Returns false on non-numeric input or values < 1.
bool parse_positive(std::string_view text, cfd::mesh::Label& value) {
    cfd::mesh::Label parsed = 0;
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), last, parsed);
    if (ec != std::errc{} || ptr != last || parsed < 1) {
        return false;
    }
    value = parsed;
    return true;
}

// Parses a strictly positive, finite real number. Returns false on invalid input.
bool parse_positive_real(std::string_view text, double& value) {
    double parsed = 0.0;
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), last, parsed);
    if (ec != std::errc{} || ptr != last || !(parsed > 0.0) || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

void print_mesh_summary(const cfd::mesh::Mesh& mesh) {
    std::cout << "Points:          " << mesh.n_points() << "\n"
              << "Faces:           " << mesh.n_faces() << " (internal " << mesh.n_internal_faces()
              << ", boundary " << mesh.n_boundary_faces() << ")\n"
              << "Cells:           " << mesh.n_cells() << "\n"
              << "Patches:\n";
    for (const auto& patch : mesh.patches()) {
        std::cout << "  " << patch.name << " (" << cfd::mesh::to_string(patch.type)
                  << "): " << patch.n_faces << " faces\n";
    }
    std::cout << "\n" << cfd::mesh::to_string(cfd::mesh::compute_quality(mesh)) << "\n";
}

int run_mesh_box(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "cfd: error: mesh-box expects 3 arguments\n";
        print_usage(std::cerr);
        return 1;
    }

    cfd::mesh::BoxSpec spec;
    if (!parse_positive(argv[2], spec.nx) || !parse_positive(argv[3], spec.ny) ||
        !parse_positive(argv[4], spec.nz)) {
        std::cerr << "cfd: error: mesh-box dimensions must be integers >= 1\n";
        print_usage(std::cerr);
        return 1;
    }

    print_mesh_summary(cfd::mesh::make_box_mesh(spec));
    return 0;
}

int run_check_mesh(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "cfd: error: check-mesh expects 1 argument\n";
        print_usage(std::cerr);
        return 1;
    }

    print_mesh_summary(cfd::mesh::read_gmsh(argv[2]));
    return 0;
}

int run_cavity(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "cfd: error: run-cavity expects 2 arguments\n";
        print_usage(std::cerr);
        return 1;
    }

    cfd::mesh::Label n = 0;
    if (!parse_positive(argv[2], n) || n < 2) {
        std::cerr << "cfd: error: run-cavity n must be an integer >= 2\n";
        print_usage(std::cerr);
        return 1;
    }
    double re = 0.0;
    if (!parse_positive_real(argv[3], re)) {
        std::cerr << "cfd: error: run-cavity Re must be a positive number\n";
        print_usage(std::cerr);
        return 1;
    }

    cfd::mesh::BoxSpec spec;
    spec.nx = n;
    spec.ny = n;
    spec.nz = 1;
    spec.max = cfd::core::Vec3(1.0, 1.0, 1.0 / static_cast<double>(n));
    spec.patch_types[4] = cfd::mesh::PatchType::Symmetry; // zmin
    spec.patch_types[5] = cfd::mesh::PatchType::Symmetry; // zmax
    const cfd::mesh::Mesh mesh = cfd::mesh::make_box_mesh(spec);

    cfd::fv::FlowSettings settings;
    settings.nu = 1.0 / re;
    const std::map<std::string, cfd::fv::BoundaryCondition> conditions{
        {"ymax", {cfd::fv::FlowBoundary::Wall, cfd::core::Vec3(1.0, 0.0, 0.0), 0.0}}};

    cfd::fv::FlowSolver solver(mesh, conditions, settings);
    const int iterations = solver.solve_steady(1e-6, 5000);

    const auto& history = solver.history();
    const bool converged = !history.empty() && history.back().max() < 1e-6;
    const cfd::fv::Residuals last = history.empty() ? cfd::fv::Residuals{} : history.back();

    if (converged) {
        std::cout << "Converged in " << iterations << " iterations\n";
    } else {
        std::cout << "Not converged after " << iterations << " iterations\n";
    }
    std::cout << std::scientific << std::setprecision(3) << "u " << last.u.maxCoeff() << "  p "
              << last.p << "  continuity " << last.continuity << "\n";

    cfd::mesh::write_vtu("cavity.vtu", mesh, {{"p", &solver.pressure()}},
                         {{"U", &solver.velocity()}});
    std::cout << "Wrote cavity.vtu\n";

    return converged ? 0 : 2;
}

int run_case(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "cfd: error: run expects 1 argument\n";
        print_usage(std::cerr);
        return 1;
    }

    const cfd::parallel::Comm& comm = cfd::parallel::Comm::world();
    std::ostream null_stream(nullptr);
    std::ostream& out = comm.is_root() ? std::cout : null_stream;

    const cfd::cases::CaseConfig config = cfd::cases::load_case(argv[2]);
    cfd::cases::CaseRunner runner(config);
    const cfd::cases::CaseConfig& cfg = runner.config();
    const bool transient = cfg.solver.mode == cfd::cases::SolverMode::Transient;
    const bool laminar = cfg.turbulence == cfd::cases::TurbulenceKind::Laminar;

    out << "Case: " << cfg.name << "  (" << runner.global_mesh().n_cells() << " cells, "
        << (transient ? "transient" : "steady") << ", " << (laminar ? "laminar" : "kOmegaSST");
    if (comm.parallel()) {
        out << ", " << comm.size() << " processes";
    }
    out << ")" << std::endl;

    const int log_interval = cfg.output.log_interval;
    const bool energy = cfg.energy.has_value();
    const auto progress = [log_interval, laminar, energy, &out](
                              int step, cfd::cases::Scalar sim_time, const cfd::fv::Residuals& r) {
        if (step == 1 || (log_interval > 0 && step % log_interval == 0)) {
            out << "step " << step << "  time " << sim_time << "  U " << std::scientific
                << std::setprecision(3) << r.u.maxCoeff() << "  p " << r.p;
            if (!laminar) {
                out << "  k " << r.k << "  omega " << r.omega;
            }
            if (energy) {
                out << "  T " << r.t;
            }
            out << "  cont " << r.continuity << std::defaultfloat << std::setprecision(6)
                << std::endl;
        }
        return true;
    };

    const cfd::cases::RunResult result = runner.run(progress);

    if (result.converged && transient) {
        out << "Finished " << result.steps << " steps (end time " << result.time << ")\n";
    } else if (result.converged) {
        out << "Converged after " << result.steps << " steps\n";
    } else {
        out << "Not converged after " << result.steps << " steps\n";
    }
    out << "Results: " << result.result_file.string() << "\n";

    return result.converged ? 0 : 2;
}

} // namespace

int main(int argc, char* argv[]) {
    const cfd::parallel::Environment mpi(argc, argv);
    if (argc < 2) {
        print_usage(std::cout);
        return 0;
    }

    const std::string_view arg{argv[1]};

    if (arg == "--help" || arg == "-h") {
        print_usage(std::cout);
        return 0;
    }
    if (arg == "--version") {
        std::cout << "cfd " << cfd::core::k_version_string << "\n";
        return 0;
    }
    if (arg == "--info") {
        std::cout << cfd::core::build_info_string()
                  << "MPI: " << (cfd::parallel::Environment::mpi_available() ? "yes" : "no")
                  << "\n";
        return 0;
    }

    if (arg == "mesh-box" || arg == "check-mesh" || arg == "run-cavity" || arg == "run") {
        try {
            if (arg == "mesh-box") {
                return run_mesh_box(argc, argv);
            }
            if (arg == "check-mesh") {
                return run_check_mesh(argc, argv);
            }
            if (arg == "run") {
                return run_case(argc, argv);
            }
            return run_cavity(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "cfd: error: " << e.what() << std::endl;
            if (cfd::parallel::Comm::world().parallel()) {
                cfd::parallel::Environment::abort(1); // other processes may wait in MPI calls
            }
            return 1;
        }
    }

    std::cerr << "cfd: unknown argument '" << arg << "'\n";
    print_usage(std::cerr);
    return 1;
}
