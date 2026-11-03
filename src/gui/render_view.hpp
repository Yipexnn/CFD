#pragma once

#include "display_settings.hpp"
#include "vtk_convert.hpp"

#include <QString>
#include <QWidget>
#include <cfd/mesh/mesh.hpp>
#include <utility>
#include <vtkSmartPointer.h>

class QVTKOpenGLNativeWidget;
class vtkActor;
class vtkCellDataToPointData;
class vtkContourFilter;
class vtkCutter;
class vtkDataSetMapper;
class vtkGenericOpenGLRenderWindow;
class vtkGeometryFilter;
class vtkGlyph3D;
class vtkLineSource;
class vtkLookupTable;
class vtkMaskPoints;
class vtkPlane;
class vtkPolyDataMapper;
class vtkRenderer;
class vtkScalarBarActor;
class vtkStreamTracer;
class vtkTextActor;
class vtkUnstructuredGrid;

namespace cfd::gui {

// 3D view of a mesh and its solution fields (VTK pipelines behind a Qt widget).
class RenderView : public QWidget {
    Q_OBJECT
public:
    explicit RenderView(QWidget* parent = nullptr);
    ~RenderView() override;
    RenderView(const RenderView&) = delete;
    RenderView& operator=(const RenderView&) = delete;
    RenderView(RenderView&&) = delete;
    RenderView& operator=(RenderView&&) = delete;

    // Shows a new mesh (boundary coloured by patch until fields arrive).
    void set_mesh(const mesh::Mesh& mesh);
    // Updates the solution fields (sizes must match the mesh).
    void set_fields(const FieldSet& fields);
    void clear_fields();
    void set_display(const DisplaySettings& settings);

    [[nodiscard]] bool has_fields() const { return has_fields_; }
    // Range of the currently coloured field over the whole grid.
    [[nodiscard]] std::pair<double, double> field_range() const;

    void set_title(const QString& text);
    void reset_camera();
    void set_view_direction(int axis, bool positive);
    bool save_screenshot(const QString& file);
    void render();

private:
    void rebuild();

    QVTKOpenGLNativeWidget* widget_ = nullptr;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> window_;
    vtkSmartPointer<vtkRenderer> renderer_;

    vtkSmartPointer<vtkUnstructuredGrid> grid_;
    vtkSmartPointer<vtkUnstructuredGrid> boundary_;
    vtkSmartPointer<vtkCellDataToPointData> point_data_;
    vtkSmartPointer<vtkGeometryFilter> surface_;
    vtkSmartPointer<vtkPlane> plane_;
    vtkSmartPointer<vtkCutter> slice_;
    vtkSmartPointer<vtkContourFilter> contours_;
    vtkSmartPointer<vtkMaskPoints> mask_;
    vtkSmartPointer<vtkGlyph3D> glyphs_;
    vtkSmartPointer<vtkLineSource> seeds_;
    vtkSmartPointer<vtkStreamTracer> streamlines_;

    vtkSmartPointer<vtkLookupTable> lut_;
    vtkSmartPointer<vtkLookupTable> patch_lut_;
    vtkSmartPointer<vtkDataSetMapper> surface_mapper_;
    vtkSmartPointer<vtkPolyDataMapper> slice_mapper_;
    vtkSmartPointer<vtkPolyDataMapper> contour_mapper_;
    vtkSmartPointer<vtkPolyDataMapper> glyph_mapper_;
    vtkSmartPointer<vtkPolyDataMapper> stream_mapper_;
    vtkSmartPointer<vtkActor> surface_actor_;
    vtkSmartPointer<vtkActor> slice_actor_;
    vtkSmartPointer<vtkActor> contour_actor_;
    vtkSmartPointer<vtkActor> glyph_actor_;
    vtkSmartPointer<vtkActor> stream_actor_;
    vtkSmartPointer<vtkScalarBarActor> scalar_bar_;
    vtkSmartPointer<vtkTextActor> title_;

    DisplaySettings settings_;
    bool has_mesh_ = false;
    bool has_fields_ = false;
    std::vector<std::string> patch_names_;
    double bounds_[6] = {0, 1, 0, 1, 0, 1};
};

} // namespace cfd::gui
