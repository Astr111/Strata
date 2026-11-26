#pragma once
// --adapt-decay and --adapt-policy helpers: linear decay validation (#1284) and Peak EWMA (PEWMA) lifecycle.
#include "strata/core/pewma.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace strata::program {

constexpr float kAdaptDecayDefault  = 0.7f;
constexpr float kPewmaGammaDefault  = 0.7f;
constexpr float kPewmaTauDefault    = 128.0f;
constexpr float kPewmaInitDefault   = 0.0f;
constexpr float kPewmaImpulseAuto   = 0.0f;

enum class AdaptPolicy : uint8_t {
    Linear = 0,
    Pewma  = 1,
};

/// `v` itself when it is finite and in 0..1 (0 forgets everything at once, 1 never forgets: both work), else the default
/// with a line on stderr.
inline float adapt_decay_checked(double v, std::FILE* log = stderr) {
    if (std::isfinite(v) && v >= 0.0 && v <= 1.0) return (float) v;
    if (log != nullptr)
        std::fprintf(log, "strata: --adapt-decay %g is not between 0 and 1; using %g (counts above 1 grow without end and the swap gains turn NaN)\n",
                     v, (double) kAdaptDecayDefault);
    return kAdaptDecayDefault;
}

inline AdaptPolicy adapt_policy_parse(const char* s, std::FILE* log = stderr) {
    if (s != nullptr) {
        if (std::strcmp(s, "linear") == 0) return AdaptPolicy::Linear;
        if (std::strcmp(s, "pewma") == 0)  return AdaptPolicy::Pewma;
    }
    if (log != nullptr)
        std::fprintf(log, "strata: --adapt-policy '%s' is unknown; using linear\n", s ? s : "(null)");
    return AdaptPolicy::Linear;
}

inline float pewma_gamma_checked(double v, std::FILE* log = stderr) {
    if (std::isfinite(v) && v > 0.0 && v <= 1.0) return (float) v;
    if (log != nullptr)
        std::fprintf(log, "strata: --pewma-gamma %g must be in (0, 1]; using %g\n", v, (double) kPewmaGammaDefault);
    return kPewmaGammaDefault;
}

inline float pewma_tau_checked(double v, std::FILE* log = stderr) {
    if (std::isfinite(v) && v >= 0.0) return (float) v;
    if (log != nullptr)
        std::fprintf(log, "strata: --pewma-tau %g must be >= 0; using %g\n", v, (double) kPewmaTauDefault);
    return kPewmaTauDefault;
}

inline float pewma_init_checked(double v, std::FILE* log = stderr) {
    if (std::isfinite(v) && v >= 0.0) return (float) v;
    if (log != nullptr)
        std::fprintf(log, "strata: --pewma-init %g must be >= 0; using %g\n", v, (double) kPewmaInitDefault);
    return kPewmaInitDefault;
}

inline float pewma_impulse_checked(double v, std::FILE* log = stderr) {
    if (std::isfinite(v) && v >= 0.0) return (float) v;
    if (log != nullptr)
        std::fprintf(log, "strata: --pewma-impulse %g must be >= 0; using auto (0)\n", v);
    return kPewmaImpulseAuto;
}

/// Initializes usage tracking arrays on `ExpertDispatch`, lazily allocating `usage_last_tok`
/// only when `adapt_policy == Pewma && pewma_tau > 0`.
template <typename TDispatch>
inline void init_usage(TDispatch& d, size_t n_pairs) {
    d.usage.assign(n_pairs, 0.0f);
    d.usage_clock = 0;
    if (d.adapt_policy == AdaptPolicy::Pewma && d.pewma_tau > 0.0f) {
        d.usage_last_tok.assign(n_pairs, -1);
    } else {
        d.usage_last_tok.clear();
        d.usage_last_tok.shrink_to_fit();
    }
}

/// Records expert routing activations for the current layer on the request thread (#6560).
/// Linear mode is byte-identical to `d.usage[...] += 1.0f`.
/// PEWMA mode advances `usage_clock` by `+n_tok` on `layer == 0` and applies window-count-scaled Peak EWMA.
template <typename TDispatch>
inline void record_layer_usage(TDispatch& d, const int32_t* ids, int64_t n_tok, int64_t k) {
    if (d.usage.empty() || ids == nullptr || n_tok <= 0 || k <= 0) return;

    if (d.adapt_policy == AdaptPolicy::Linear) {
        for (int64_t i = 0; i < n_tok * k; ++i) {
            if (ids[i] >= 0 && ids[i] < d.n_expert) {
                d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
            }
        }
        return;
    }

    if (d.layers == 0) {
        d.usage_clock += n_tok;
    }

    const float impulse = (d.pewma_impulse > 0.0f)
        ? d.pewma_impulse
        : (n_tok == 1 ? 2.0f : 1.0f);

    thread_local std::vector<int32_t> counts;
    thread_local std::vector<int32_t> uniq;
    if ((int64_t) counts.size() < d.n_expert) {
        counts.assign((size_t) d.n_expert, 0);
    }
    uniq.clear();

    const int64_t total = n_tok * k;
    for (int64_t i = 0; i < total; ++i) {
        const int32_t e = ids[i];
        if (e >= 0 && e < d.n_expert) {
            if (counts[(size_t) e] == 0) uniq.push_back(e);
            counts[(size_t) e] += 1;
        }
    }

    const size_t base = (size_t) d.layers * (size_t) d.n_expert;
    const bool continuous = (d.pewma_tau > 0.0f && d.usage_last_tok.size() == d.usage.size());

    for (int32_t e : uniq) {
        const float x_t = (float) counts[(size_t) e] * impulse;
        counts[(size_t) e] = 0;
        const size_t idx = base + (size_t) e;
        if (idx >= d.usage.size()) continue;

        if (continuous) {
            if (d.usage_last_tok[idx] < 0) {
                d.usage[idx] = std::max(d.pewma_init, x_t);
            } else {
                const float dt = (d.usage_clock >= d.usage_last_tok[idx])
                    ? (float) (d.usage_clock - d.usage_last_tok[idx])
                    : 0.0f;
                d.usage[idx] = strata::core::PeakEWMA::update_timed(d.usage[idx], x_t, dt, d.pewma_tau);
            }
            d.usage_last_tok[idx] = d.usage_clock;
        } else {
            if (d.usage[idx] == 0.0f && d.pewma_init > 0.0f) {
                d.usage[idx] = std::max(d.pewma_init, x_t);
            } else {
                d.usage[idx] = strata::core::PeakEWMA::update_step(d.usage[idx], x_t, d.pewma_gamma);
            }
        }
    }
}

/// Takes a snapshot of `d.usage`, accumulates `heat`, and applies the configured decay on the request thread.
/// In continuous PEWMA mode (`tau > 0`), decays dormant entries once per elapsed `dt` and syncs `d.usage`
/// and `d.usage_last_tok` to `d.usage_clock` so `STRATA_RAM_ADAPT2` sees up-to-date scores without double-decay.
template <typename TDispatch>
inline std::vector<float> snapshot_and_decay_usage(TDispatch& d, std::vector<double>& heat, float linear_decay) {
    if (d.adapt_policy == AdaptPolicy::Linear || d.pewma_tau <= 0.0f || d.usage_last_tok.size() != d.usage.size()) {
        const float factor = (d.adapt_policy == AdaptPolicy::Linear) ? linear_decay : d.pewma_gamma;
        std::vector<float> snap = d.usage;
        const size_t nh = std::min(heat.size(), snap.size());
        for (size_t i = 0; i < nh; ++i) heat[i] += (double) snap[i];
        for (float& v : d.usage) v *= factor;
        return snap;
    }

    std::vector<float> snap(d.usage.size(), 0.0f);
    for (size_t i = 0; i < d.usage.size(); ++i) {
        if (d.usage_last_tok[i] >= 0) {
            const float dt = (d.usage_clock >= d.usage_last_tok[i])
                ? (float) (d.usage_clock - d.usage_last_tok[i])
                : 0.0f;
            const float decayed = d.usage[i] * strata::core::PeakEWMA::weight_continuous(dt, d.pewma_tau);
            snap[i] = decayed;
            d.usage[i] = decayed;
            d.usage_last_tok[i] = d.usage_clock;
        }
        if (i < heat.size()) heat[i] += (double) snap[i];
    }
    return snap;
}

}  // namespace strata::program
