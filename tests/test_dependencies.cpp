#include <Eigen/Geometry>
#include <cfd/core/types.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

TEST(Dependencies, EigenCrossProductOfUnitVectors) {
    using cfd::core::Vec3;
    const Vec3 cross = Vec3::UnitX().cross(Vec3::UnitY());
    EXPECT_TRUE(cross.isApprox(Vec3::UnitZ()));
}

TEST(Dependencies, NlohmannJsonRoundTrip) {
    const nlohmann::json original = {{"rho", 1.225}, {"n", 42}, {"name", "air"}};
    const nlohmann::json parsed = nlohmann::json::parse(original.dump());

    EXPECT_EQ(parsed, original);
    EXPECT_DOUBLE_EQ(parsed.at("rho").get<double>(), 1.225);
    EXPECT_EQ(parsed.at("n").get<int>(), 42);
    EXPECT_EQ(parsed.at("name").get<std::string>(), "air");
}

TEST(Dependencies, OpenMpParallelReduction) {
#ifdef _OPENMP
    long long sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (cfd::core::Label i = 0; i < 1000000; ++i) {
        sum += i;
    }
    EXPECT_EQ(sum, 999999LL * 1000000LL / 2);
#else
    GTEST_SKIP() << "OpenMP disabled";
#endif
}
