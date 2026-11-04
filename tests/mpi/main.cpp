#include <cfd/parallel/comm.hpp>
#include <gtest/gtest.h>

// gtest main for tests run with mpiexec: only rank 0 prints results.
int main(int argc, char** argv) {
    const cfd::parallel::Environment env(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    if (!cfd::parallel::Comm::world().is_root()) {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }
    return RUN_ALL_TESTS();
}
