// engine/tests/core/pewma_test.cpp - Unit tests for Peak EWMA (PEWMA) and ExpertDispatch lifecycle helpers.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "strata/core/pewma.hpp"
#include "strata/program/adapt_decay.hpp"

using strata::core::PeakEWMA;
using strata::core::PeakEWMATracker;
using strata::program::AdaptPolicy;
using strata::program::init_usage;
using strata::program::record_layer_usage;
using strata::program::snapshot_and_decay_usage;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
#define CHECK_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) { std::fprintf(stderr, "FAIL %s:%d: |%g - %g| > %g\n", __FILE__, __LINE__, (double)(a), (double)(b), (double)(eps)); ++failed; } } while (0)

namespace {
struct MockDispatch {
    int64_t n_expert = 8;
    int64_t layers = 0;
    std::vector<float> usage;
    AdaptPolicy adapt_policy = AdaptPolicy::Linear;
    float pewma_gamma = 0.7f;
    float pewma_tau = 128.0f;
    float pewma_init = 0.0f;
    float pewma_impulse = 0.0f;
    int64_t usage_clock = 0;
    std::vector<int64_t> usage_last_tok;
};
}  // namespace

int main() {
    std::puts("Running Peak EWMA (PEWMA) unit tests...");

    // Test 1: Instantaneous rise on single spike vs exponential decay
    {
        float s = 0.0f;
        const float gamma = 0.7f;
        s = PeakEWMA::update_step(s, 10.0f, gamma);
        CHECK_NEAR(s, 10.0f, 1e-5f);
        s = PeakEWMA::update_step(s, 0.0f, gamma);
        CHECK_NEAR(s, 7.0f, 1e-5f);
        s = PeakEWMA::update_step(s, 0.0f, gamma);
        CHECK_NEAR(s, 4.9f, 1e-5f);
        s = PeakEWMA::update_step(s, 0.0f, gamma);
        CHECK_NEAR(s, 3.43f, 1e-5f);
    }

    // Test 2: Series of rising peaks
    {
        float s = 0.0f;
        const float gamma = 0.8f;
        s = PeakEWMA::update_step(s, 5.0f, gamma);
        CHECK_NEAR(s, 5.0f, 1e-5f);
        s = PeakEWMA::update_step(s, 12.0f, gamma);
        CHECK_NEAR(s, 12.0f, 1e-5f);
        s = PeakEWMA::update_step(s, 8.0f, gamma);
        CHECK_NEAR(s, 11.2f, 1e-5f);
    }

    // Test 3: Continuous time interval formulation
    {
        const float tau = 100.0f;
        float s = 10.0f;
        s = PeakEWMA::update_timed(s, 0.0f, 0.0f, tau);
        CHECK_NEAR(s, 10.0f, 1e-5f);
        const float expected_w = std::exp(-1.0f);
        s = PeakEWMA::update_timed(10.0f, 0.0f, 100.0f, tau);
        CHECK_NEAR(s, 10.0f * expected_w, 1e-4f);
        s = PeakEWMA::update_timed(s, 20.0f, 50.0f, tau);
        CHECK_NEAR(s, 20.0f, 1e-5f);
    }

    // Test 4: Stateful PeakEWMATracker
    {
        PeakEWMATracker tracker(0.0f, 128.0f);
        CHECK_NEAR(tracker.value(), 0.0f, 1e-5f);
        tracker.record(15.0f, 100);
        CHECK_NEAR(tracker.value(), 15.0f, 1e-5f);
        CHECK(tracker.last_time() == 100);
        float q = tracker.query(228);
        CHECK_NEAR(q, 15.0f * std::exp(-1.0f), 1e-4f);
        CHECK_NEAR(tracker.value(), 15.0f, 1e-5f);
        tracker.record(25.0f, 250);
        CHECK_NEAR(tracker.value(), 25.0f, 1e-5f);
        CHECK(tracker.last_time() == 250);
    }

    // Test 5: Lazy allocation in init_usage()
    {
        MockDispatch d;
        d.adapt_policy = AdaptPolicy::Linear;
        init_usage(d, 16);
        CHECK(d.usage.size() == 16);
        CHECK(d.usage_last_tok.empty());

        d.adapt_policy = AdaptPolicy::Pewma;
        d.pewma_tau = 0.0f;
        init_usage(d, 16);
        CHECK(d.usage_last_tok.empty());

        d.pewma_tau = 128.0f;
        init_usage(d, 16);
        CHECK(d.usage_last_tok.size() == 16);
        CHECK(d.usage_last_tok[0] == -1);
    }

    // Test 6: Linear mode parity in record_layer_usage + snapshot_and_decay_usage
    {
        MockDispatch d;
        d.adapt_policy = AdaptPolicy::Linear;
        init_usage(d, 16);
        std::vector<double> heat(16, 0.0);
        const int32_t ids[] = {1, 3, 1, 2};  // n_tok=2, k=2
        d.layers = 0;
        record_layer_usage(d, ids, 2, 2);
        CHECK_NEAR(d.usage[1], 2.0f, 1e-6f);
        CHECK_NEAR(d.usage[2], 1.0f, 1e-6f);
        CHECK_NEAR(d.usage[3], 1.0f, 1e-6f);
        auto snap = snapshot_and_decay_usage(d, heat, 0.7f);
        CHECK_NEAR(snap[1], 2.0f, 1e-6f);
        CHECK_NEAR(d.usage[1], 1.4f, 1e-6f);
        CHECK_NEAR(heat[1], 2.0, 1e-6);
    }

    // Test 7: PEWMA continuous mode: layer-0 clock, auto impulse (1.0 batch / 2.0 decode), no double decay
    {
        MockDispatch d;
        d.adapt_policy = AdaptPolicy::Pewma;
        d.pewma_tau = 100.0f;
        d.pewma_init = 0.5f;
        init_usage(d, 16);
        std::vector<double> heat(16, 0.0);

        // Batch window (n_tok=4, k=2): expert 1 hit 3 times -> 3.0 >= 2.0; expert 2 hit once -> 1.0 < 2.0
        const int32_t batch_ids[] = {1, 2, 1, 3, 1, 4, 5, 6};
        d.layers = 0;
        record_layer_usage(d, batch_ids, 4, 2);
        CHECK(d.usage_clock == 4);
        CHECK_NEAR(d.usage[1], 3.0f, 1e-5f);
        CHECK_NEAR(d.usage[2], 1.0f, 1e-5f);

        // Layer 1 does not advance usage_clock
        d.layers = 1;
        const int32_t l1_ids[] = {0, 7, 0, 7, 0, 7, 0, 7};
        record_layer_usage(d, l1_ids, 4, 2);
        CHECK(d.usage_clock == 4);
        CHECK_NEAR(d.usage[8 + 0], 4.0f, 1e-5f);

        // Advance clock by 100 tokens without hitting expert 1, then snapshot_and_decay_usage
        d.layers = 0;
        const int32_t decode_ids[] = {3, 4};  // n_tok=1, k=2 -> auto impulse 2.0
        for (int t = 0; t < 100; ++t) {
            record_layer_usage(d, decode_ids, 1, 2);
        }
        CHECK(d.usage_clock == 104);
        CHECK_NEAR(d.usage[3], 2.0f, 1e-5f);

        auto snap1 = snapshot_and_decay_usage(d, heat, 0.7f);
        const float expected_e1 = 3.0f * std::exp(-100.0f / 100.0f);
        CHECK_NEAR(snap1[1], expected_e1, 1e-4f);
        CHECK_NEAR(d.usage[1], expected_e1, 1e-4f);
        CHECK(d.usage_last_tok[1] == 104);

        // Calling snapshot_and_decay_usage again at same clock must NOT double-decay
        auto snap2 = snapshot_and_decay_usage(d, heat, 0.7f);
        CHECK_NEAR(snap2[1], expected_e1, 1e-4f);
        CHECK_NEAR(d.usage[1], expected_e1, 1e-4f);
    }

    if (failed) {
        std::fprintf(stderr, "pewma_test: %d check(s) FAILED\n", failed);
        return 1;
    }
    std::puts("pewma_test: ALL CHECKS PASSED (OK)");
    return 0;
}
