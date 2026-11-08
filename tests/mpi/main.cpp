#include <cfd/parallel/comm.hpp>
#include <cstdlib>
#include <gtest/gtest.h>

namespace {

bool env_set(const char* name) {
#ifdef _MSC_VER
    char* value = nullptr;
    std::size_t size = 0;
    const bool set = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
    std::free(value);
    return set;
#else
    return std::getenv(name) != nullptr;
#endif
}

} // namespace

// gtest main for tests run with mpiexec: only rank 0 prints results.
int main(int argc, char** argv) {
    const cfd::parallel::Environment env(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    // CFD_MPI_ALL_OUTPUT=1 keeps the output of every process (debugging).
    if (!cfd::parallel::Comm::world().is_root() && !env_set("CFD_MPI_ALL_OUTPUT")) {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }
    return RUN_ALL_TESTS();
}
