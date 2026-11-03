#include "render_view.hpp"

#include <QVBoxLayout>
#include <QVTKOpenGLNativeWidget.h>
#include <algorithm>
#include <cmath>
#include <vtkActor.h>
#include <vtkArrowSource.h>
#include <vtkCamera.h>
#include <vtkCellData.h>
#include <vtkCellDataToPointData.h>
#include <vtkContourFilter.h>
#include <vtkCutter.h>
#include <vtkDataArray.h>
#include <vtkDataSetMapper.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkGeometryFilter.h>
#include <vtkGlyph3D.h>
#include <vtkLineSource.h>
#include <vtkLookupTable.h>
#include <vtkMaskPoints.h>
#include <vtkPNGWriter.h>
#include <vtkPlane.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkScalarBarActor.h>
#include <vtkStreamTracer.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>

namespace cfd::gui {

namespace {

constexpr double k_patch_colours[][3] = {
    {0.65, 0.65, 0.65}, {0.20, 0.47, 0.85}, {0.90, 0.35, 0.25}, {0.30, 0.70, 0.35},
    {0.95, 0.75, 0.20}, {0.60, 0.40, 0.80}, {0.20, 0.75, 0.80}, {0.85, 0.50, 0.70},
};

void configure_field_mapper(vtkMapper& mapper, const std::string& field, bool point_data,
                            vtkLookupTable* lut, double lo, double hi) {
    if (field.empty()) {
        mapper.ScalarVisibilityOff();
        return;
    }
    mapper.ScalarVisibilityOn();
    if (point_data) {
        mapper.SetScalarModeToUsePointFieldData();
    } else {
        mapper.SetScalarModeToUseCellFieldData();
    }
    mapper.SelectColorArray(field.c_str());
    mapper.SetLookupTable(lut);
    mapper.SetScalarRange(lo, hi);
    mapper.UseLookupTableScalarRangeOff();
}

} // namespace

RenderView::RenderView(QWidget* parent) : QWidget(parent) {
    // Lines (streamlines, edges) drawn on top of coincident polygons (slice plane).
    vtkMapper::SetResolveCoincidentTopologyToPolygonOffset();
    widget_ = new QVTKOpenGLNativeWidget(this);
    window_ = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    widget_->setRenderWindow(window_);
    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    renderer_->GradientBackgroundOn();
    renderer_->SetBackground(0.32, 0.34, 0.43);
    renderer_->SetBackground2(0.82, 0.85, 0.90);
    window_->AddRenderer(renderer_);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(widget_);

    lut_ = vtkSmartPointer<vtkLookupTable>::New();
    lut_->SetHueRange(0.667, 0.0); // blue -> red
    lut_->SetNumberOfTableValues(256);
    lut_->Build();
    patch_lut_ = vtkSmartPointer<vtkLookupTable>::New();

    grid_ = vtkSmartPointer<vtkUnstructuredGrid>::New();
    boundary_ = vtkSmartPointer<vtkUnstructuredGrid>::New();

    point_data_ = vtkSmartPointer<vtkCellDataToPointData>::New();
    point_data_->SetInputData(grid_);
    point_data_->PassCellDataOn();

    surface_ = vtkSmartPointer<vtkGeometryFilter>::New();
    surface_->SetInputData(grid_);
    surface_mapper_ = vtkSmartPointer<vtkDataSetMapper>::New();
    surface_actor_ = vtkSmartPointer<vtkActor>::New();
    surface_actor_->SetMapper(surface_mapper_);
    surface_actor_->GetProperty()->SetEdgeColor(0.1, 0.1, 0.1);

    plane_ = vtkSmartPointer<vtkPlane>::New();
    slice_ = vtkSmartPointer<vtkCutter>::New();
    slice_->SetCutFunction(plane_);
    slice_->SetInputConnection(point_data_->GetOutputPort());
    slice_mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    slice_mapper_->SetInputConnection(slice_->GetOutputPort());
    slice_actor_ = vtkSmartPointer<vtkActor>::New();
    slice_actor_->SetMapper(slice_mapper_);

    contours_ = vtkSmartPointer<vtkContourFilter>::New();
    contours_->SetInputConnection(point_data_->GetOutputPort());
    contours_->ComputeScalarsOn();
    contour_mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    contour_mapper_->SetInputConnection(contours_->GetOutputPort());
    contour_actor_ = vtkSmartPointer<vtkActor>::New();
    contour_actor_->SetMapper(contour_mapper_);

    mask_ = vtkSmartPointer<vtkMaskPoints>::New();
    mask_->SetInputConnection(slice_->GetOutputPort());
    mask_->RandomModeOff();
    auto arrow = vtkSmartPointer<vtkArrowSource>::New();
    glyphs_ = vtkSmartPointer<vtkGlyph3D>::New();
    glyphs_->SetInputConnection(mask_->GetOutputPort());
    glyphs_->SetSourceConnection(arrow->GetOutputPort());
    glyphs_->SetVectorModeToUseVector();
    glyphs_->SetScaleModeToScaleByVector();
    glyphs_->OrientOn();
    glyphs_->SetColorModeToColorByVector();
    glyphs_->SetInputArrayToProcess(1, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "U");
    glyph_mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    glyph_mapper_->SetInputConnection(glyphs_->GetOutputPort());
    glyph_actor_ = vtkSmartPointer<vtkActor>::New();
    glyph_actor_->SetMapper(glyph_mapper_);

    seeds_ = vtkSmartPointer<vtkLineSource>::New();
    streamlines_ = vtkSmartPointer<vtkStreamTracer>::New();
    streamlines_->SetInputConnection(point_data_->GetOutputPort());
    streamlines_->SetSourceConnection(seeds_->GetOutputPort());
    streamlines_->SetIntegrationDirectionToBoth();
    streamlines_->SetIntegratorTypeToRungeKutta45();
    streamlines_->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "U");
    stream_mapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    stream_mapper_->SetInputConnection(streamlines_->GetOutputPort());
    stream_actor_ = vtkSmartPointer<vtkActor>::New();
    stream_actor_->SetMapper(stream_mapper_);
    stream_actor_->GetProperty()->SetLineWidth(1.5);

    scalar_bar_ = vtkSmartPointer<vtkScalarBarActor>::New();
    scalar_bar_->SetLookupTable(lut_);
    scalar_bar_->SetNumberOfLabels(5);
    scalar_bar_->SetWidth(0.08);
    scalar_bar_->SetHeight(0.6);
    scalar_bar_->SetPosition(0.88, 0.2);
    scalar_bar_->UnconstrainedFontSizeOn();
    scalar_bar_->SetLabelFormat("%-#8.3g");
    scalar_bar_->GetLabelTextProperty()->SetFontSize(14);
    scalar_bar_->GetLabelTextProperty()->ShadowOff();
    scalar_bar_->GetLabelTextProperty()->ItalicOff();
    scalar_bar_->GetTitleTextProperty()->SetFontSize(16);
    scalar_bar_->GetTitleTextProperty()->ItalicOff();

    title_ = vtkSmartPointer<vtkTextActor>::New();
    title_->GetTextProperty()->SetFontSize(16);
    title_->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
    title_->SetPosition(10, 10);

    for (vtkActor* a : {surface_actor_.Get(), slice_actor_.Get(), contour_actor_.Get(),
                        glyph_actor_.Get(), stream_actor_.Get()}) {
        a->VisibilityOff();
        renderer_->AddActor(a);
    }
    renderer_->AddActor2D(scalar_bar_);
    renderer_->AddActor2D(title_);
    scalar_bar_->VisibilityOff();
}

RenderView::~RenderView() = default;

void RenderView::set_mesh(const mesh::Mesh& mesh) {
    grid_->DeepCopy(mesh_to_vtk(mesh));
    boundary_->DeepCopy(boundary_to_vtk(mesh));
    grid_->GetBounds(bounds_);
    patch_names_.clear();
    for (const auto& p : mesh.patches()) {
        patch_names_.push_back(p.name);
    }
    const auto n_patches = static_cast<vtkIdType>(std::max<std::size_t>(patch_names_.size(), 1));
    patch_lut_->SetNumberOfTableValues(n_patches);
    patch_lut_->IndexedLookupOn();
    patch_lut_->ResetAnnotations();
    for (vtkIdType i = 0; i < n_patches; ++i) {
        const auto* c = k_patch_colours[static_cast<std::size_t>(i) % 8];
        patch_lut_->SetTableValue(i, c[0], c[1], c[2], 1.0);
        if (static_cast<std::size_t>(i) < patch_names_.size()) {
            patch_lut_->SetAnnotation(vtkVariant(static_cast<int>(i)),
                                      patch_names_[static_cast<std::size_t>(i)].c_str());
        }
    }
    patch_lut_->Build();
    has_mesh_ = true;
    has_fields_ = false;
    rebuild();
    reset_camera();
}

void RenderView::set_fields(const FieldSet& fields) {
    if (!has_mesh_) {
        return;
    }
    gui::set_fields(*grid_, fields);
    point_data_->Modified();
    surface_->Modified();
    has_fields_ = !fields.u.empty() || !fields.p.empty();
    rebuild();
}

void RenderView::clear_fields() {
    gui::set_fields(*grid_, FieldSet{});
    has_fields_ = false;
    rebuild();
}

void RenderView::set_display(const DisplaySettings& settings) {
    settings_ = settings;
    rebuild();
}

std::pair<double, double> RenderView::field_range() const {
    if (!has_fields_ || settings_.field.empty()) {
        return {0.0, 1.0};
    }
    vtkDataArray* a = grid_->GetCellData()->GetArray(settings_.field.c_str());
    if (a == nullptr) {
        return {0.0, 1.0};
    }
    double r[2];
    a->GetRange(r, 0);
    if (r[1] - r[0] < 1e-30) {
        r[1] = r[0] + 1e-30;
    }
    return {r[0], r[1]};
}

void RenderView::set_title(const QString& text) {
    title_->SetInput(text.toUtf8().constData());
    render();
}

void RenderView::rebuild() {
    if (!has_mesh_) {
        render();
        return;
    }
    const bool fields = has_fields_;
    const std::string field =
        fields && grid_->GetCellData()->GetArray(settings_.field.c_str()) != nullptr
            ? settings_.field
            : std::string();
    auto [lo, hi] = field_range();
    if (!settings_.auto_range) {
        lo = settings_.range_min;
        hi = std::max(settings_.range_max, lo + 1e-30);
    }
    lut_->SetTableRange(lo, hi);

    // Surface: patches before fields exist, then the outer surface coloured by field.
    if (fields) {
        surface_mapper_->SetInputConnection(surface_->GetOutputPort());
        configure_field_mapper(*surface_mapper_, field, false, lut_, lo, hi);
    } else {
        surface_mapper_->SetInputData(boundary_);
        surface_mapper_->ScalarVisibilityOn();
        surface_mapper_->SetScalarModeToUseCellFieldData();
        surface_mapper_->SelectColorArray("patch");
        surface_mapper_->SetLookupTable(patch_lut_);
        surface_mapper_->UseLookupTableScalarRangeOn();
    }
    surface_actor_->SetVisibility(settings_.show_surface || !fields);
    surface_actor_->GetProperty()->SetOpacity(fields ? settings_.surface_opacity : 1.0);
    surface_actor_->GetProperty()->SetEdgeVisibility(settings_.show_edges);

    // Slice plane.
    const int axis = std::clamp(settings_.slice_axis, 0, 2);
    double origin[3] = {0.5 * (bounds_[0] + bounds_[1]), 0.5 * (bounds_[2] + bounds_[3]),
                        0.5 * (bounds_[4] + bounds_[5])};
    const double lo_b = bounds_[2 * axis];
    const double hi_b = bounds_[2 * axis + 1];
    origin[axis] = lo_b + std::clamp(settings_.slice_position, 0.0, 1.0) * (hi_b - lo_b);
    // Keep the plane strictly inside the domain so the cut is never empty.
    origin[axis] =
        std::clamp(origin[axis], lo_b + 1e-6 * (hi_b - lo_b), hi_b - 1e-6 * (hi_b - lo_b));
    double normal[3] = {0.0, 0.0, 0.0};
    normal[axis] = 1.0;
    plane_->SetOrigin(origin);
    plane_->SetNormal(normal);
    const bool slice_needed = fields && (settings_.show_slice || settings_.show_vectors);
    slice_actor_->SetVisibility(fields && settings_.show_slice);
    configure_field_mapper(*slice_mapper_, field, true, lut_, lo, hi);

    // Contours.
    contour_actor_->SetVisibility(fields && settings_.show_contours && !field.empty());
    if (fields && settings_.show_contours && !field.empty()) {
        contours_->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,
                                          field.c_str());
        const int n = std::max(settings_.contour_count, 1);
        contours_->GenerateValues(n, lo + (hi - lo) / (n + 1), hi - (hi - lo) / (n + 1));
        configure_field_mapper(*contour_mapper_, field, true, lut_, lo, hi);
    }

    // Vectors on the slice plane.
    const double diag =
        std::sqrt(std::pow(bounds_[1] - bounds_[0], 2) + std::pow(bounds_[3] - bounds_[2], 2) +
                  std::pow(bounds_[5] - bounds_[4], 2));
    glyph_actor_->SetVisibility(fields && settings_.show_vectors);
    if (fields && settings_.show_vectors && slice_needed) {
        mask_->SetOnRatio(std::max(settings_.vector_stride, 1));
        mask_->SetMaximumNumberOfPoints(100000);
        double umax = 1e-30;
        if (vtkDataArray* u = grid_->GetCellData()->GetArray("|U|")) {
            double r[2];
            u->GetRange(r, 0);
            umax = std::max(r[1], 1e-30);
        }
        glyphs_->SetScaleFactor(0.08 * diag / umax * settings_.vector_scale);
        // Arrows are coloured by |U| (the glyph filter's vector-magnitude scalars).
        glyph_mapper_->ScalarVisibilityOn();
        glyph_mapper_->SetScalarModeToUsePointData();
        glyph_mapper_->SetLookupTable(lut_);
        glyph_mapper_->SetScalarRange(0.0, umax);
        glyph_mapper_->UseLookupTableScalarRangeOff();
    }

    // Streamlines seeded on the slice-plane diagonal.
    stream_actor_->SetVisibility(fields && settings_.show_streamlines);
    if (fields && settings_.show_streamlines) {
        double p1[3] = {bounds_[0], bounds_[2], bounds_[4]};
        double p2[3] = {bounds_[1], bounds_[3], bounds_[5]};
        for (int i = 0; i < 3; ++i) {
            const double inset = 0.01 * (bounds_[2 * i + 1] - bounds_[2 * i]);
            p1[i] += inset;
            p2[i] -= inset;
        }
        p1[axis] = origin[axis];
        p2[axis] = origin[axis];
        seeds_->SetPoint1(p1);
        seeds_->SetPoint2(p2);
        seeds_->SetResolution(std::max(settings_.streamline_seeds - 1, 1));
        streamlines_->SetMaximumPropagation(3.0 * diag);
        streamlines_->SetInitialIntegrationStep(0.2);
        streamlines_->SetMaximumNumberOfSteps(20000);
        // On top of a coloured slice, field-coloured lines would be invisible.
        configure_field_mapper(*stream_mapper_, settings_.show_slice ? std::string() : field, true,
                               lut_, lo, hi);
        stream_actor_->GetProperty()->SetColor(0.05, 0.05, 0.05);
    }

    // Colour bar: field range, or patch legend before fields exist.
    if (fields && !field.empty()) {
        scalar_bar_->SetLookupTable(lut_);
        scalar_bar_->SetTitle(field.c_str());
        scalar_bar_->SetVisibility(settings_.show_scalar_bar);
    } else {
        scalar_bar_->SetLookupTable(patch_lut_);
        scalar_bar_->SetTitle("patches");
        scalar_bar_->SetVisibility(!patch_names_.empty());
    }
    render();
}

void RenderView::reset_camera() {
    renderer_->ResetCamera();
    render();
}

void RenderView::set_view_direction(int axis, bool positive) {
    vtkCamera* cam = renderer_->GetActiveCamera();
    const double c[3] = {0.5 * (bounds_[0] + bounds_[1]), 0.5 * (bounds_[2] + bounds_[3]),
                         0.5 * (bounds_[4] + bounds_[5])};
    double pos[3] = {c[0], c[1], c[2]};
    pos[std::clamp(axis, 0, 2)] += positive ? 1.0 : -1.0;
    cam->SetFocalPoint(c);
    cam->SetPosition(pos);
    cam->SetViewUp(axis == 1 ? 0.0 : 0.0, axis == 1 ? 0.0 : 1.0, axis == 1 ? 1.0 : 0.0);
    renderer_->ResetCamera();
    render();
}

bool RenderView::save_screenshot(const QString& file) {
    window_->Render();
    auto capture = vtkSmartPointer<vtkWindowToImageFilter>::New();
    capture->SetInput(window_);
    capture->ReadFrontBufferOff();
    capture->Update();
    auto writer = vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(file.toUtf8().constData());
    writer->SetInputConnection(capture->GetOutputPort());
    writer->Write();
    return writer->GetErrorCode() == 0;
}

void RenderView::render() {
    if (window_ != nullptr) {
        window_->Render();
    }
}

} // namespace cfd::gui
