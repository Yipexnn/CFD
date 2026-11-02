#include <cfd/core/build_info.hpp>
#include <cfd/core/version.hpp>
#include <gtest/gtest.h>
#include <string>

namespace {

std::string expected_version_string() {
    return std::to_string(cfd::core::k_version_major) + "." +
           std::to_string(cfd::core::k_version_minor) + "." +
           std::to_string(cfd::core::k_version_patch);
}

} // namespace

TEST(Version, StringMatchesComponents) {
    EXPECT_EQ(std::string(cfd::core::k_version_string), expected_version_string());
}

TEST(BuildInfo, VersionMatchesVersionConstant) {
    const cfd::core::BuildInfo info = cfd::core::build_info();
    EXPECT_EQ(info.version, std::string(cfd::core::k_version_string));
}

TEST(BuildInfo, FieldsArePopulated) {
    const cfd::core::BuildInfo info = cfd::core::build_info();
    EXPECT_FALSE(info.compiler.empty());
    EXPECT_FALSE(info.eigen_version.empty());
    EXPECT_FALSE(info.build_type.empty());
    EXPECT_GE(info.openmp_max_threads, 1);
}

TEST(BuildInfo, StringContainsVersion) {
    const std::string text = cfd::core::build_info_string();
    const std::string version(cfd::core::k_version_string);
    EXPECT_NE(text.find(version), std::string::npos);
}
