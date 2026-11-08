// Case editor round trip (headless): settings without dedicated widgets survive, the new
// compressible / property / radiation / boundary fields map to the config.
#include "case_editor.hpp"

#include <QApplication>
#include <cstdlib>
#include <gtest/gtest.h>

namespace {

QApplication& app() {
#ifndef _WIN32
    setenv("QT_QPA_PLATFORM", "offscreen", 0); // Windows: static Qt ships the native plugin only
#endif
    static int argc = 1;
    static char name[] = "cfd_case_editor_tests";
    static char* argv[] = {name, nullptr};
    static QApplication application(argc, argv);
    return application;
}

} // namespace

TEST(CaseEditor, RoundTripOfNewSettings) {
    (void)app();
    using namespace cfd;
    cases::CaseConfig c;
    c.name = "editor";
    fv::CompressibleSettings cs;
    cs.mu = 1.8e-5;
    cs.mu_law.kind = fv::PropertyLaw::Kind::Sutherland;
    cs.mu_law.reference = 1.8e-5;
    cs.implicit = true;
    cs.krylov_dim = 25;
    cs.limiter = fv::Limiter::Venkatakrishnan;
    cs.cfl = 300.0;
    c.compressible = cs;
    fv::EnergySettings es;
    es.nu_law.kind = fv::PropertyLaw::Kind::Polynomial;
    es.nu_law.coefficients = {1e-3, 2e-5};
    c.energy = es;
    fv::RadiationSettings rs;
    rs.absorption = 2.5;
    rs.emissivity = 0.7;
    rs.patch_emissivity["xmin"] = 0.3;
    c.radiation = rs;
    fv::BoundaryCondition wall{fv::FlowBoundary::Wall, core::Vec3::Zero(), 0.0};
    wall.contact_angle = 60.0;
    wall.temperature = 350.0;
    c.boundaries["xmin"] = wall;

    gui::CaseEditor editor;
    editor.set_config(c);
    editor.set_patches({{"xmin", mesh::PatchType::Wall}, {"xmax", mesh::PatchType::Outlet}});
    const cases::CaseConfig b = editor.config();

    ASSERT_TRUE(b.compressible.has_value());
    EXPECT_EQ(b.compressible->mu_law.kind, fv::PropertyLaw::Kind::Sutherland);
    EXPECT_DOUBLE_EQ(b.compressible->mu, 1.8e-5);
    EXPECT_TRUE(b.compressible->implicit);
    EXPECT_EQ(b.compressible->krylov_dim, 25);
    EXPECT_EQ(b.compressible->limiter, fv::Limiter::Venkatakrishnan);
    EXPECT_DOUBLE_EQ(b.compressible->cfl, 300.0);
    ASSERT_TRUE(b.energy.has_value());
    EXPECT_EQ(b.energy->nu_law.kind, fv::PropertyLaw::Kind::Polynomial);
    ASSERT_EQ(b.energy->nu_law.coefficients.size(), 2U);
    EXPECT_DOUBLE_EQ(b.energy->nu_law.coefficients[1], 2e-5);
    ASSERT_TRUE(b.radiation.has_value());
    EXPECT_DOUBLE_EQ(b.radiation->absorption, 2.5);
    EXPECT_DOUBLE_EQ(b.radiation->emissivity, 0.7);
    EXPECT_DOUBLE_EQ(b.radiation->patch_emissivity.at("xmin"), 0.3); // no widget: kept
    EXPECT_DOUBLE_EQ(b.boundaries.at("xmin").contact_angle, 60.0);
    EXPECT_DOUBLE_EQ(b.boundaries.at("xmin").temperature, 350.0);
}
