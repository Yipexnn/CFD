#include "main_window.hpp"

#include <QApplication>
#include <QCommandLineParser>
#include <QStringList>
#include <QSurfaceFormat>
#include <QTimer>
#include <QVTKOpenGLNativeWidget.h>
#include <cfd/core/version.hpp>

// Usage: cfd_gui [case.json] [--run] [--show surface,slice,contours,vectors,streamlines,edges]
//                [--field NAME] [--slice-axis 0|1|2] [--screenshot view.png]
//                [--window-screenshot window.png] [--quit]
// The screenshot/quit options make the GUI scriptable for automated checks.
int main(int argc, char* argv[]) {
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication app(argc, argv);
    QApplication::setApplicationName("CFD");
    QApplication::setApplicationVersion(
        QString::fromStdString(std::string(cfd::core::k_version_string)));

    QCommandLineParser parser;
    parser.setApplicationDescription("CFD desktop application");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("case", "Case file (*.json) to open.");
    const QCommandLineOption run_opt("run", "Run the case after opening it.");
    const QCommandLineOption show_opt("show", "Display elements to enable (comma separated).",
                                      "list");
    const QCommandLineOption field_opt("field", "Field to colour by.", "name");
    const QCommandLineOption axis_opt("slice-axis", "Slice normal axis 0/1/2.", "axis");
    const QCommandLineOption shot_opt("screenshot", "Save a 3D view screenshot.", "file");
    const QCommandLineOption wshot_opt("window-screenshot", "Save a window screenshot.", "file");
    const QCommandLineOption quit_opt("quit", "Quit after the other actions.");
    parser.addOptions({run_opt, show_opt, field_opt, axis_opt, shot_opt, wshot_opt, quit_opt});
    parser.process(app);

    cfd::gui::MainWindow window;
    window.show();

    const QStringList positional = parser.positionalArguments();
    if (!positional.isEmpty()) {
        window.open_case(positional.front());
    }

    auto finish = [&] {
        cfd::gui::DisplaySettings d = window.display();
        if (parser.isSet(show_opt)) {
            const QStringList items = parser.value(show_opt).split(',');
            d.show_surface = items.contains("surface");
            d.show_slice = items.contains("slice");
            d.show_contours = items.contains("contours");
            d.show_vectors = items.contains("vectors");
            d.show_streamlines = items.contains("streamlines");
            d.show_edges = items.contains("edges");
        }
        if (parser.isSet(field_opt)) {
            d.field = parser.value(field_opt).toStdString();
        }
        if (parser.isSet(axis_opt)) {
            d.slice_axis = parser.value(axis_opt).toInt();
        }
        window.set_display(d);
        if (parser.isSet(shot_opt)) {
            window.save_view_screenshot(parser.value(shot_opt));
        }
        if (parser.isSet(wshot_opt)) {
            window.save_window_screenshot(parser.value(wshot_opt));
        }
        if (parser.isSet(quit_opt)) {
            QTimer::singleShot(0, &app, &QApplication::quit);
        }
    };

    if (parser.isSet(run_opt)) {
        QObject::connect(&window, &cfd::gui::MainWindow::run_finished, &app,
                         [&](bool) { QTimer::singleShot(200, &app, finish); });
        QTimer::singleShot(200, &app, [&] { window.start_run(); });
    } else {
        QTimer::singleShot(500, &app, finish);
    }
    return QApplication::exec();
}
