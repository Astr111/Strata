// engine/include/strata/core/pewma.hpp - Peak EWMA (PEWMA) asymmetric time-series smoothing.
//
// PEWMA models an asymmetric tracker that records rising values instantaneously (zero rise latency)
// and smooths decaying values exponentially. In MoE expert caching, this ensures sudden activation bursts
// are recognized immediately by the cache admission policy, while dormant intervals decay smoothly
// without thrashing.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::core {

/// Peak EWMA (PEWMA) core implementation supporting both discrete steps and continuous time.
class PeakEWMA {
public:
    /// Default decay parameters
    static constexpr float kDefaultGamma = 0.7f;    ///< Discrete decay factor \gamma \in (0, 1]
    static constexpr float kDefaultTau   = 128.0f;  ///< Continuous time constant \tau (tokens/steps)

    /// Computes the decay weight for discrete steps: w = gamma
    static constexpr float weight_discrete(float gamma) {
        return gamma;
    }

    /// Computes the decay weight for continuous time intervals: w = exp(-dt / tau)
    static inline float weight_continuous(float dt, float tau) {
        if (tau <= 0.0f || dt <= 0.0f) {
            return 1.0f;
        }
        return std::exp(-dt / tau);
    }

    /// Computes decay weight using base-2 half-life (tau_half): w = 2^(-dt / tau_half)
    static inline float weight_halflife(float dt, float tau_half) {
        if (tau_half <= 0.0f || dt <= 0.0f) {
            return 1.0f;
        }
        return std::exp2(-dt / tau_half);
    }

    /// Single PEWMA update given weight w \in (0, 1]:
    /// S_t = x_t                          if x_t > S_{t-1}
    /// S_t = w * S_{t-1} + (1 - w) * x_t  if x_t <= S_{t-1}
    static inline float update(float s_prev, float x_t, float w) {
        if (x_t > s_prev) {
            return x_t;
        }
        return w * s_prev + (1.0f - w) * x_t;
    }

    /// Discrete step update with fixed gamma
    static inline float update_step(float s_prev, float x_t, float gamma = kDefaultGamma) {
        return update(s_prev, x_t, weight_discrete(gamma));
    }

    /// Continuous time interval update with time constant tau
    static inline float update_timed(float s_prev, float x_t, float dt, float tau = kDefaultTau) {
        return update(s_prev, x_t, weight_continuous(dt, tau));
    }

    /// Continuous time interval update with half-life T_{1/2}
    static inline float update_halflife(float s_prev, float x_t, float dt, float tau_half = kDefaultTau) {
        return update(s_prev, x_t, weight_halflife(dt, tau_half));
    }
};

/// Stateful Peak EWMA tracker for a single continuous time series.
class PeakEWMATracker {
public:
    explicit PeakEWMATracker(float initial_value = 0.0f, float tau = PeakEWMA::kDefaultTau)
        : value_(initial_value), tau_(tau), last_t_(0), initialized_(initial_value > 0.0f) {}

    void reset(float initial_value = 0.0f) {
        value_ = initial_value;
        last_t_ = 0;
        initialized_ = (initial_value > 0.0f);
    }

    /// Record new observation x_t at timestamp t
    float record(float x_t, int64_t t) {
        if (!initialized_) {
            value_ = x_t;
            last_t_ = t;
            initialized_ = true;
            return value_;
        }
        const float dt = (t >= last_t_) ? static_cast<float>(t - last_t_) : 0.0f;
        value_ = PeakEWMA::update_timed(value_, x_t, dt, tau_);
        last_t_ = t;
        return value_;
    }

    /// Query current value decayed to timestamp t without new observation (x_t = 0)
    float query(int64_t t) const {
        if (!initialized_) return 0.0f;
        const float dt = (t >= last_t_) ? static_cast<float>(t - last_t_) : 0.0f;
        const float w = PeakEWMA::weight_continuous(dt, tau_);
        return w * value_;
    }

    float value() const { return value_; }
    int64_t last_time() const { return last_t_; }

private:
    float value_ = 0.0f;
    float tau_ = PeakEWMA::kDefaultTau;
    int64_t last_t_ = 0;
    bool initialized_ = false;
};

}  // namespace strata::core
