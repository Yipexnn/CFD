#pragma once

#include "display_settings.hpp"

#include <QWidget>
#include <string>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QSlider;
class QSpinBox;

namespace cfd::gui {

// Post-processing controls. Edits a DisplaySettings and reports every user change.
class PostPanel : public QWidget {
    Q_OBJECT

public:
    explicit PostPanel(QWidget* parent = nullptr);

    // Replaces the field list. "(solid)" is always first and maps to an empty field name.
    void set_fields(const std::vector<std::string>& names);

    [[nodiscard]] DisplaySettings settings() const;
    void set_settings(const DisplaySettings& s);

    // Updates the min/max boxes (e.g. to show the automatic range). Emits nothing.
    void set_range(double min, double max);

signals:
    void settings_changed(const cfd::gui::DisplaySettings& value);

private:
    void emit_changed();
    void update_enabled();

    bool updating_ = false;

    QComboBox* field_ = nullptr;
    QCheckBox* auto_range_ = nullptr;
    QDoubleSpinBox* range_min_ = nullptr;
    QDoubleSpinBox* range_max_ = nullptr;
    QCheckBox* scalar_bar_ = nullptr;

    QCheckBox* show_surface_ = nullptr;
    QCheckBox* show_edges_ = nullptr;
    QSlider* opacity_ = nullptr;

    QCheckBox* show_slice_ = nullptr;
    QComboBox* slice_axis_ = nullptr;
    QSlider* slice_position_ = nullptr;

    QCheckBox* show_contours_ = nullptr;
    QSpinBox* contour_count_ = nullptr;

    QCheckBox* show_vectors_ = nullptr;
    QDoubleSpinBox* vector_scale_ = nullptr;
    QSpinBox* vector_stride_ = nullptr;

    QCheckBox* show_streamlines_ = nullptr;
    QSpinBox* streamline_seeds_ = nullptr;
};

} // namespace cfd::gui
