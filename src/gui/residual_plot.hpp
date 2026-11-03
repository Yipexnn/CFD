#pragma once

#include <QPointF>
#include <QSize>
#include <QWidget>
#include <array>
#include <cstddef>
#include <vector>

class QPaintEvent;

namespace cfd::gui {

// Log-scale plot of solver residuals (Ux, Uy, Uz, p, k, omega) versus iteration step.
class ResidualPlot : public QWidget {
    Q_OBJECT

public:
    explicit ResidualPlot(QWidget* parent = nullptr);

    void clear();
    void add_point(int step, double ux, double uy, double uz, double p, double k, double omega);
    void set_tolerance(double tol);

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    static constexpr std::size_t kSeriesCount = 6;

    // Each point is (x = step, y = log10(value)).
    std::array<std::vector<QPointF>, kSeriesCount> series_;
    int max_step_ = 0;
    double tolerance_ = 0.0;
};

} // namespace cfd::gui
