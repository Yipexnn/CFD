#include "residual_plot.hpp"

#include <QColor>
#include <QFont>
#include <QFontMetrics>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QRectF>
#include <QString>
#include <algorithm>
#include <cmath>
#include <limits>

namespace cfd::gui {

namespace {

constexpr std::size_t kMaxPoints = 4000;
constexpr int kXIntervals = 4;    // 5 x tick labels
constexpr int kMaxYIntervals = 8; // soft cap on decade ticks
constexpr int kDefaultLoExp = -8; // 1e-8
constexpr int kDefaultHiExp = 0;  // 1

struct SeriesStyle {
    const char* name;
    QColor color;
};

// Index order matches the add_point() argument order.
SeriesStyle style_of(std::size_t index) {
    switch (index) {
    case 0:
        return {"Ux", QColor(30, 90, 220)};
    case 1:
        return {"Uy", QColor(30, 160, 60)};
    case 2:
        return {"Uz", QColor(0, 180, 210)};
    case 3:
        return {"p", QColor(215, 40, 40)};
    case 4:
        return {"k", QColor(250, 140, 0)};
    case 5:
        return {"omega", QColor(140, 70, 190)};
    default:
        return {"T", QColor(120, 70, 20)};
    }
}

} // namespace

ResidualPlot::ResidualPlot(QWidget* parent) : QWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent, true);
}

void ResidualPlot::clear() {
    for (auto& series : series_) {
        series.clear();
    }
    max_step_ = 0;
    update();
}

void ResidualPlot::add_point(int step, double ux, double uy, double uz, double p, double k,
                             double omega, double t) {
    const std::array<double, kSeriesCount> values{ux, uy, uz, p, k, omega, t};
    max_step_ = std::max(max_step_, step);
    const double x = static_cast<double>(step);

    for (std::size_t i = 0; i < kSeriesCount; ++i) {
        // Skips zero, negative and non-finite values (laminar runs pass k = omega = 0).
        if (!(values[i] > 0.0) || !std::isfinite(values[i])) {
            continue;
        }
        auto& series = series_[i];
        series.emplace_back(x, std::log10(values[i]));

        if (series.size() > kMaxPoints) {
            // Keep every 2nd point, in place.
            std::size_t out = 0;
            for (std::size_t in = 0; in < series.size(); in += 2) {
                series[out++] = series[in];
            }
            series.resize(out);
        }
    }
    update();
}

void ResidualPlot::set_tolerance(double tol) {
    tolerance_ = tol;
    update();
}

QSize ResidualPlot::sizeHint() const {
    return QSize(600, 220);
}

QSize ResidualPlot::minimumSizeHint() const {
    return QSize(300, 150);
}

void ResidualPlot::paintEvent(QPaintEvent* /*event*/) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), Qt::white);

    const QFont base_font = painter.font();
    const QFontMetrics fm = painter.fontMetrics();
    const int line_h = fm.height();

    // Title.
    {
        QFont title_font = base_font;
        title_font.setBold(true);
        painter.setFont(title_font);
        painter.setPen(QColor(30, 30, 30));
        painter.drawText(QRectF(8, 4, width() - 16, line_h + 4),
                         static_cast<int>(Qt::AlignLeft | Qt::AlignVCenter),
                         QStringLiteral("Residuals"));
        painter.setFont(base_font);
    }

    // Y range over all stored values (and the tolerance, so its line stays visible).
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& series : series_) {
        for (const QPointF& pt : series) {
            min_y = std::min(min_y, pt.y());
            max_y = std::max(max_y, pt.y());
        }
    }
    if (tolerance_ > 0.0) {
        const double tol_y = std::log10(tolerance_);
        min_y = std::min(min_y, tol_y);
        max_y = std::max(max_y, tol_y);
    }
    const bool has_range = min_y <= max_y;
    const int y_lo = has_range ? static_cast<int>(std::floor(min_y)) : kDefaultLoExp;
    int y_hi = has_range ? static_cast<int>(std::ceil(max_y)) : kDefaultHiExp;
    if (y_hi <= y_lo) {
        y_hi = y_lo + 1;
    }

    const int y_step = std::max(1, (y_hi - y_lo + kMaxYIntervals - 1) / kMaxYIntervals);
    std::vector<int> y_ticks;
    for (int d = y_lo; d <= y_hi; d += y_step) {
        y_ticks.push_back(d);
    }

    int label_w = 0;
    for (int d : y_ticks) {
        label_w = std::max(label_w, fm.horizontalAdvance(QStringLiteral("1e%1").arg(d)));
    }

    // Plot area.
    const double x_hi = static_cast<double>(std::max(1, max_step_));
    const double plot_left = label_w + 14.0;
    const double plot_top = line_h + 12.0;
    const double plot_right = width() - 12.0;
    const double plot_bottom = height() - line_h - 10.0;
    if (plot_right - plot_left < 20.0 || plot_bottom - plot_top < 20.0) {
        return;
    }
    const QRectF plot(plot_left, plot_top, plot_right - plot_left, plot_bottom - plot_top);
    const double y_span = static_cast<double>(y_hi - y_lo);

    const auto x_px = [&](double x) { return plot.left() + x / x_hi * plot.width(); };
    const auto y_px = [&](double y) {
        return plot.bottom() - (y - static_cast<double>(y_lo)) / y_span * plot.height();
    };

    // Grid.
    painter.setPen(QPen(QColor(225, 225, 225), 1.0));
    for (int i = 1; i <= kXIntervals; ++i) {
        const double px = x_px(x_hi * i / kXIntervals);
        painter.drawLine(QPointF(px, plot.top()), QPointF(px, plot.bottom()));
    }
    for (int d : y_ticks) {
        const double py = y_px(d);
        painter.drawLine(QPointF(plot.left(), py), QPointF(plot.right(), py));
    }

    // Axis border.
    painter.setPen(QPen(QColor(120, 120, 120), 1.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(plot);

    // Tick labels.
    painter.setPen(QColor(60, 60, 60));
    for (int d : y_ticks) {
        const double py = y_px(d);
        painter.drawText(QRectF(0, py - line_h / 2.0, plot_left - 6.0, line_h),
                         static_cast<int>(Qt::AlignRight | Qt::AlignVCenter),
                         QStringLiteral("1e%1").arg(d));
    }
    for (int i = 0; i <= kXIntervals; ++i) {
        const double x_val = x_hi * i / kXIntervals;
        const double px = x_px(x_val);
        const QString label = QString::number(static_cast<long long>(std::llround(x_val)));
        painter.drawText(QRectF(px - 30.0, plot.bottom() + 3.0, 60.0, line_h),
                         static_cast<int>(Qt::AlignHCenter | Qt::AlignTop), label);
    }

    // Tolerance line (dashed).
    if (tolerance_ > 0.0) {
        const double ty = y_px(std::log10(tolerance_));
        painter.setPen(QPen(QColor(90, 90, 90), 1.0, Qt::DashLine));
        painter.drawLine(QPointF(plot.left(), ty), QPointF(plot.right(), ty));
    }

    // Series.
    painter.save();
    painter.setClipRect(plot);
    for (std::size_t i = 0; i < kSeriesCount; ++i) {
        const auto& pts = series_[i];
        if (pts.empty()) {
            continue;
        }
        QPainterPath path;
        path.moveTo(x_px(pts.front().x()), y_px(pts.front().y()));
        for (std::size_t j = 1; j < pts.size(); ++j) {
            path.lineTo(x_px(pts[j].x()), y_px(pts[j].y()));
        }
        painter.setPen(QPen(style_of(i).color, 1.5));
        painter.drawPath(path);
    }
    painter.restore();

    // Legend (top right, only series with data).
    const int row_h = line_h + 2;
    const int swatch_w = 18;
    const int pad = 6;
    int box_w = 0;
    int rows = 0;
    for (std::size_t i = 0; i < kSeriesCount; ++i) {
        if (!series_[i].empty()) {
            box_w = std::max(box_w, fm.horizontalAdvance(QString::fromLatin1(style_of(i).name)));
            ++rows;
        }
    }
    if (rows > 0) {
        box_w += swatch_w + 8 + 2 * pad;
        const int box_h = rows * row_h + 2 * pad - 2;
        const QRectF box(plot.right() - box_w - 6.0, plot.top() + 6.0, box_w, box_h);
        painter.fillRect(box, QColor(255, 255, 255, 230));
        painter.setPen(QPen(QColor(180, 180, 180), 1.0));
        painter.drawRect(box);

        double row_y = box.top() + pad;
        for (std::size_t i = 0; i < kSeriesCount; ++i) {
            if (series_[i].empty()) {
                continue;
            }
            const SeriesStyle style = style_of(i);
            const double cy = row_y + row_h / 2.0;
            painter.setPen(QPen(style.color, 2.0));
            painter.drawLine(QPointF(box.left() + pad, cy),
                             QPointF(box.left() + pad + swatch_w, cy));
            painter.setPen(QColor(40, 40, 40));
            painter.drawText(QPointF(box.left() + pad + swatch_w + 8.0, row_y + fm.ascent() + 1.0),
                             QString::fromLatin1(style.name));
            row_y += row_h;
        }
    }
}

} // namespace cfd::gui
