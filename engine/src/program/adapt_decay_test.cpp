// --adapt-decay validation (include/strata/program/adapt_decay.hpp), CPU only, no model (#1284).
#include "strata/program/adapt_decay.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

using strata::program::AdaptPolicy;
using strata::program::adapt_decay_checked;
using strata::program::adapt_policy_parse;
using strata::program::kAdaptDecayDefault;
using strata::program::kPewmaGammaDefault;
using strata::program::kPewmaImpulseAuto;
using strata::program::kPewmaInitDefault;
using strata::program::kPewmaTauDefault;
using strata::program::pewma_gamma_checked;
using strata::program::pewma_impulse_checked;
using strata::program::pewma_init_checked;
using strata::program::pewma_tau_checked;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    // inside the range: the value itself, no line (bit for bit what atof gave before, cast to float)
    for (double v : {0.0, 0.25, 0.5, 0.7, 0.9, 0.97, 0.999, 1.0}) {
        check(adapt_decay_checked(v, nullptr) == (float) v, "a value in 0..1 is kept");
    }
    // every default and tune_knobs value of the engine is unchanged
    check(adapt_decay_checked(0.7, nullptr) == kAdaptDecayDefault, "the default stays 0.7");
    // outside: the default
    for (double v : {1.0000001, 1.5, 2.0, 1e30, -0.1, -1.0, -1e30}) {
        check(adapt_decay_checked(v, nullptr) == kAdaptDecayDefault, "an out-of-range value falls back to the default");
    }
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(adapt_decay_checked(inf, nullptr) == kAdaptDecayDefault, "inf falls back");
    check(adapt_decay_checked(-inf, nullptr) == kAdaptDecayDefault, "-inf falls back");
    check(adapt_decay_checked(nan, nullptr) == kAdaptDecayDefault, "NaN falls back");
    // the result is always usable: finite and in 0..1
    for (double v : {-5.0, 0.3, 7.0, nan, inf}) {
        const float r = adapt_decay_checked(v, nullptr);
        check(std::isfinite(r) && r >= 0.f && r <= 1.f, "the result is finite and in 0..1");
    }
    // a counts vector driven by an accepted value stays finite over many adaptations (the bug: >1 overflowed to inf)
    float usage = 1.f;
    const float d = adapt_decay_checked(1.5, nullptr);
    for (int i = 0; i < 100000; ++i) usage = usage * d + 1.f;
    check(std::isfinite(usage), "counts stay finite with the value the engine ends up using");

    // PEWMA parameter validators
    check(adapt_policy_parse("linear", nullptr) == AdaptPolicy::Linear, "linear parses");
    check(adapt_policy_parse("pewma", nullptr) == AdaptPolicy::Pewma, "pewma parses");
    check(adapt_policy_parse("bogus", nullptr) == AdaptPolicy::Linear, "unknown policy falls back to linear");
    check(pewma_gamma_checked(0.7, nullptr) == 0.7f, "valid gamma kept");
    check(pewma_gamma_checked(0.0, nullptr) == kPewmaGammaDefault, "gamma 0 falls back");
    check(pewma_gamma_checked(1.2, nullptr) == kPewmaGammaDefault, "gamma > 1 falls back");
    check(pewma_tau_checked(0.0, nullptr) == 0.0f, "tau 0 allowed (discrete mode)");
    check(pewma_tau_checked(-1.0, nullptr) == kPewmaTauDefault, "negative tau falls back");
    check(pewma_init_checked(0.5, nullptr) == 0.5f, "valid init kept");
    check(pewma_init_checked(-0.1, nullptr) == kPewmaInitDefault, "negative init falls back");
    check(pewma_impulse_checked(0.0, nullptr) == kPewmaImpulseAuto, "impulse 0 (auto) allowed");
    check(pewma_impulse_checked(-2.0, nullptr) == kPewmaImpulseAuto, "negative impulse falls back");

    std::printf("adapt_decay_test: %d checks passed\n", checks);
    return 0;
}
