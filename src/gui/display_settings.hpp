#pragma once

#include <string>

namespace cfd::gui {

// What the 3D view shows; edited by the post-processing panel.
struct DisplaySettings {
    std::string field = "|U|"; // scalar array to colour by ("" = solid colour)
    bool show_surface = true;  // outer/boundary surface coloured by field
    bool show_edges = false;   // mesh edges on the surface
    double surface_opacity = 1.0;

    bool show_slice = false;     // planar cut coloured by field
    int slice_axis = 2;          // 0 = x, 1 = y, 2 = z (plane normal)
    double slice_position = 0.5; // 0..1 across the domain bounds

    bool show_contours = false; // iso-surfaces of the field
    int contour_count = 10;

    bool show_vectors = false; // arrow glyphs of U on the slice plane
    double vector_scale = 1.0; // relative to an automatic length
    int vector_stride = 4;     // every n-th slice cell

    bool show_streamlines = false; // stream tracer seeded on a line through the domain
    int streamline_seeds = 30;

    bool auto_range = true;
    double range_min = 0.0;
    double range_max = 1.0;
    bool show_scalar_bar = true;
};

} // namespace cfd::gui
