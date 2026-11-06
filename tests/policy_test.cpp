// Companion smoke test for the ported legacy conversation policy. The exhaustive semantic
// coverage lives in conversation_golden_test.cpp (a faithful port of the legacy ESP32 test);
// this file keeps the historical `policy_test` target pointing at the ported module and pins
// the documented defaults plus the preview/confirm/reset/veto basics.
#include "khadas_ows/conversation.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace khadas_ows;

namespace {
void check(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
float feed_frames(conversation::Awareness& awareness, const conversation::Evidence& evidence,
                  int frames, float dt_ms = 100.0f) {
    conversation::Output out;
    for (int i = 0; i < frames; ++i) out = awareness.update(evidence, dt_ms);
    return out.media_percent;
}
} // namespace

int main() {
    // The migrated parameters must stay the legacy ones.
    const conversation::Params defaults;
    check(defaults.cue_threshold == 0.5f, "ported cue_threshold default");
    check(defaults.own_voice_threshold == 0.5f, "ported own_voice_threshold default");
    check(defaults.preview_confirm_ms == 500.0f && defaults.onset_confirm_ms == 800.0f,
          "ported 500 ms preview / 800 ms confirmation");
    check(defaults.preview_floor_percent == 70.0f && defaults.duck_floor_percent == 60.0f,
          "ported 70% preview / 60% duck floors");
    check(defaults.hold_ms == 2500.0f && defaults.restore_ms == 400.0f,
          "ported 2500 ms hold / 400 ms restore");
    check(conversation::near_field_own_voice(-60.0f, 0.9f) == 0.0f,
          "near-field own voice ignores quiet input");
    check(conversation::near_field_own_voice(-20.0f, 0.8f) == 0.8f,
          "near-field own voice follows loudness and likelihood");

    conversation::Evidence cue;
    cue.conversation_cue = 0.9f;
    cue.speech_active = true;
    cue.voice_likelihood = 0.9f;
    cue.level_dbfs = -40.0f;

    conversation::Awareness awareness;
    check(feed_frames(awareness, cue, 4) == 100.0f, "no early duck before the preview window");
    const conversation::Output preview = awareness.update(cue, 100.0f);
    check(preview.state == conversation::State::anticipating && !preview.ambient_open,
          "500 ms of cue stays anticipating");
    check(preview.media_percent < 90.0f && preview.media_percent >= 69.0f,
          "500 ms preview respects the 70% floor");
    conversation::Output engaged = preview;
    for (int i = 0; i < 3; ++i) engaged = awareness.update(cue, 100.0f);
    check(engaged.state == conversation::State::engaged && engaged.ambient_open,
          "800 ms confirms and opens the ambient intent");
    check(engaged.media_percent < 80.0f, "confirmation ducks the media bed");

    awareness.reset();
    const conversation::Output after_reset = awareness.update(cue, 100.0f);
    check(after_reset.state == conversation::State::anticipating &&
              after_reset.media_percent == 100.0f && !after_reset.ambient_open,
          "reset clears confirmation and ducking");

    conversation::Awareness vetoed;
    feed_frames(vetoed, cue, 10);
    conversation::Evidence veto = cue;
    veto.dialogue_veto = true;
    const conversation::Output released = vetoed.update(veto, 100.0f);
    check(released.state == conversation::State::releasing && !released.ambient_open,
          "dialogue veto immediately releases an engaged conversation");

    std::cout << "policy_test passed\n";
}
