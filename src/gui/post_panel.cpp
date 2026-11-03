#include "post_panel.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>

namespace cfd::gui {

namespace {

QString solid_label() {
    return QStringLiteral("(solid)");
}

QFormLayout* add_group(QVBoxLayout* root, const QString& title, QWidget* parent) {
    auto* box = new QGroupBox(title, parent);
    root->addWidget(box);
    return new QFormLayout(box);
}

QDoubleSpinBox* make_double(QWidget* parent, double lo, double hi, int decimals) {
    auto* box = new QDoubleSpinBox(parent);
    box->setDecimals(decimals);
    box->setRange(lo, hi);
    return box;
}

QSpinBox* make_int(QWidget* parent, int lo, int hi) {
    auto* box = new QSpinBox(parent);
    box->setRange(lo, hi);
    return box;
}

QSlider* make_slider(QWidget* parent, int lo, int hi) {
    auto* slider = new QSlider(Qt::Horizontal, parent);
    slider->setRange(lo, hi);
    return slider;
}

// Maps a unit-range double onto slider ticks 0..scale.
int to_ticks(double unit, int scale) {
    const long ticks = std::lround(unit * static_cast<double>(scale));
    return std::clamp(static_cast<int>(ticks), 0, scale);
}

double from_ticks(int ticks, int scale) {
    return static_cast<double>(ticks) / static_cast<double>(scale);
}

} // namespace

PostPanel::PostPanel(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);

    // Colour
    QFormLayout* colour = add_group(root, tr("Colour"), this);
    field_ = new QComboBox(this);
    field_->addItem(solid_label());
    colour->addRow(tr("Field"), field_);
    auto_range_ = new QCheckBox(tr("Auto range"), this);
    colour->addRow(auto_range_);
    range_min_ = make_double(this, -1e12, 1e12, 6);
    colour->addRow(tr("Min"), range_min_);
    range_max_ = make_double(this, -1e12, 1e12, 6);
    colour->addRow(tr("Max"), range_max_);
    scalar_bar_ = new QCheckBox(tr("Scalar bar"), this);
    colour->addRow(scalar_bar_);

    // Surface
    QFormLayout* surface = add_group(root, tr("Surface"), this);
    show_surface_ = new QCheckBox(tr("Show surface"), this);
    surface->addRow(show_surface_);
    show_edges_ = new QCheckBox(tr("Edges"), this);
    surface->addRow(show_edges_);
    opacity_ = make_slider(this, 0, 100);
    surface->addRow(tr("Opacity"), opacity_);

    // Slice
    QFormLayout* slice = add_group(root, tr("Slice"), this);
    show_slice_ = new QCheckBox(tr("Show slice"), this);
    slice->addRow(show_slice_);
    slice_axis_ = new QComboBox(this);
    slice_axis_->addItems({QStringLiteral("X"), QStringLiteral("Y"), QStringLiteral("Z")});
    slice->addRow(tr("Axis"), slice_axis_);
    slice_position_ = make_slider(this, 0, 1000);
    slice->addRow(tr("Position"), slice_position_);

    // Contours
    QFormLayout* contours = add_group(root, tr("Contours"), this);
    show_contours_ = new QCheckBox(tr("Show contours"), this);
    contours->addRow(show_contours_);
    contour_count_ = make_int(this, 1, 50);
    contours->addRow(tr("Count"), contour_count_);

    // Vectors
    QFormLayout* vectors = add_group(root, tr("Vectors"), this);
    show_vectors_ = new QCheckBox(tr("Show vectors"), this);
    vectors->addRow(show_vectors_);
    vector_scale_ = make_double(this, 0.01, 100.0, 2);
    vector_scale_->setSingleStep(0.1);
    vectors->addRow(tr("Scale"), vector_scale_);
    vector_stride_ = make_int(this, 1, 50);
    vectors->addRow(tr("Stride"), vector_stride_);

    // Streamlines
    QFormLayout* streamlines = add_group(root, tr("Streamlines"), this);
    show_streamlines_ = new QCheckBox(tr("Show streamlines"), this);
    streamlines->addRow(show_streamlines_);
    streamline_seeds_ = make_int(this, 2, 500);
    streamlines->addRow(tr("Seeds"), streamline_seeds_);

    root->addStretch(1);

    // Every user change goes through emit_changed(), which is muted while updating_ is set.
    connect(field_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { emit_changed(); });
    connect(auto_range_, &QCheckBox::toggled, this, [this](bool) {
        update_enabled();
        emit_changed();
    });
    connect(range_min_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { emit_changed(); });
    connect(range_max_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { emit_changed(); });
    connect(scalar_bar_, &QCheckBox::toggled, this, [this](bool) { emit_changed(); });

    connect(show_surface_, &QCheckBox::toggled, this, [this](bool) { emit_changed(); });
    connect(show_edges_, &QCheckBox::toggled, this, [this](bool) { emit_changed(); });
    connect(opacity_, &QSlider::valueChanged, this, [this](int) { emit_changed(); });

    connect(show_slice_, &QCheckBox::toggled, this, [this](bool) { emit_changed(); });
    connect(slice_axis_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { emit_changed(); });
    connect(slice_position_, &QSlider::valueChanged, this, [this](int) { emit_changed(); });

    connect(show_contours_, &QCheckBox::toggled, this, [this](bool) { emit_changed(); });
    connect(contour_count_, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { emit_changed(); });

    connect(show_vectors_, &QCheckBox::toggled, this, [this](bool) { emit_changed(); });
    connect(vector_scale_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { emit_changed(); });
    connect(vector_stride_, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { emit_changed(); });

    connect(show_streamlines_, &QCheckBox::toggled, this, [this](bool) { emit_changed(); });
    connect(streamline_seeds_, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int) { emit_changed(); });

    DisplaySettings defaults;
    defaults.field.clear();
    set_settings(defaults);
}

void PostPanel::set_fields(const std::vector<std::string>& names) {
    const QString previous = field_->currentText();

    updating_ = true;
    field_->clear();
    field_->addItem(solid_label());
    for (const std::string& name : names) {
        field_->addItem(QString::fromStdString(name));
    }

    int index = field_->findText(previous);
    if (index < 0) {
        index = field_->findText(QStringLiteral("|U|"));
    }
    if (index < 0) {
        index = names.empty() ? 0 : 1;
    }
    field_->setCurrentIndex(index);
    updating_ = false;
}

DisplaySettings PostPanel::settings() const {
    DisplaySettings s;
    s.field = field_->currentIndex() > 0 ? field_->currentText().toStdString() : std::string{};
    s.show_surface = show_surface_->isChecked();
    s.show_edges = show_edges_->isChecked();
    s.surface_opacity = from_ticks(opacity_->value(), 100);

    s.show_slice = show_slice_->isChecked();
    s.slice_axis = slice_axis_->currentIndex();
    s.slice_position = from_ticks(slice_position_->value(), 1000);

    s.show_contours = show_contours_->isChecked();
    s.contour_count = contour_count_->value();

    s.show_vectors = show_vectors_->isChecked();
    s.vector_scale = vector_scale_->value();
    s.vector_stride = vector_stride_->value();

    s.show_streamlines = show_streamlines_->isChecked();
    s.streamline_seeds = streamline_seeds_->value();

    s.auto_range = auto_range_->isChecked();
    s.range_min = range_min_->value();
    s.range_max = range_max_->value();
    s.show_scalar_bar = scalar_bar_->isChecked();
    return s;
}

void PostPanel::set_settings(const DisplaySettings& s) {
    updating_ = true;

    if (s.field.empty()) {
        field_->setCurrentIndex(0);
    } else {
        const QString name = QString::fromStdString(s.field);
        int index = field_->findText(name);
        if (index < 0) {
            field_->addItem(name);
            index = field_->count() - 1;
        }
        field_->setCurrentIndex(index);
    }

    show_surface_->setChecked(s.show_surface);
    show_edges_->setChecked(s.show_edges);
    opacity_->setValue(to_ticks(s.surface_opacity, 100));

    show_slice_->setChecked(s.show_slice);
    slice_axis_->setCurrentIndex(std::clamp(s.slice_axis, 0, 2));
    slice_position_->setValue(to_ticks(s.slice_position, 1000));

    show_contours_->setChecked(s.show_contours);
    contour_count_->setValue(s.contour_count);

    show_vectors_->setChecked(s.show_vectors);
    vector_scale_->setValue(s.vector_scale);
    vector_stride_->setValue(s.vector_stride);

    show_streamlines_->setChecked(s.show_streamlines);
    streamline_seeds_->setValue(s.streamline_seeds);

    auto_range_->setChecked(s.auto_range);
    range_min_->setValue(s.range_min);
    range_max_->setValue(s.range_max);
    scalar_bar_->setChecked(s.show_scalar_bar);

    updating_ = false;
    update_enabled();
}

void PostPanel::set_range(double min, double max) {
    updating_ = true;
    range_min_->setValue(min);
    range_max_->setValue(max);
    updating_ = false;
}

void PostPanel::emit_changed() {
    if (updating_) {
        return;
    }
    emit settings_changed(settings());
}

void PostPanel::update_enabled() {
    const bool manual = !auto_range_->isChecked();
    range_min_->setEnabled(manual);
    range_max_->setEnabled(manual);
}

} // namespace cfd::gui
