/**
 * Descriptive Statistics WASI Component in C++
 *
 * Exports a statistics interface with describe, median, and percentile
 * functions that operate on numeric data sets.
 */

#include <algorithm>
#include <cmath>
#include <vector>
#include "math_stats_cpp.h"

namespace exports {
namespace math {
namespace stats {
namespace statistics {

Summary Describe(wit::vector<double> values) {
    Summary result{};
    uint32_t count = static_cast<uint32_t>(values.size());
    if (count == 0) {
        return result;
    }

    double sum = 0.0;
    double min_val = values[0];
    double max_val = values[0];

    for (uint32_t i = 0; i < count; ++i) {
        double v = values[i];
        sum += v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    double mean = sum / count;

    double var_sum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        double diff = values[i] - mean;
        var_sum += diff * diff;
    }

    result.count = count;
    result.sum = sum;
    result.mean = mean;
    result.min_val = min_val;
    result.max_val = max_val;
    result.variance = var_sum / count;
    return result;
}

// Helper: copy wit::vector into a sorted std::vector.
static std::vector<double> to_sorted(wit::vector<double> const &values) {
    std::vector<double> sorted(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        sorted[i] = values[i];
    }
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

double Median(wit::vector<double> values) {
    if (values.empty()) return 0.0;

    auto sorted = to_sorted(values);
    size_t n = sorted.size();

    if (n % 2 == 0) {
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }
    return sorted[n / 2];
}

double Percentile(wit::vector<double> values, double p) {
    if (values.empty()) return 0.0;

    // Clamp p to [0, 100].
    if (p < 0.0) p = 0.0;
    if (p > 100.0) p = 100.0;

    auto sorted = to_sorted(values);
    size_t n = sorted.size();

    if (n == 1) return sorted[0];

    // Linear interpolation between closest ranks.
    double rank = (p / 100.0) * static_cast<double>(n - 1);
    size_t lo = static_cast<size_t>(rank);
    size_t hi = lo + 1;
    if (hi >= n) hi = n - 1;

    double frac = rank - static_cast<double>(lo);
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

}}}}  // namespace exports::math::stats::statistics
