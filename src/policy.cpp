#include "khadas_ows/policy.hpp"
#include <algorithm>
namespace khadas_ows {
void ConversationPolicy::reset() { state_=ConversationState::idle; external_ms_=hold_ms_=0; }
PolicyOutput ConversationPolicy::update(const PolicyInput& input) {
    // Identity and invalid inputs fail closed: they may preserve audio, but cannot create dialogue actions.
    const bool blocked = !input.input_valid || (input.identity_fresh && input.identity_self);
    const bool external = input.external_speech && !input.self_speech && !blocked;
    if (external) {
        external_ms_ += std::max(0.0f,input.elapsed_ms); hold_ms_=0;
        state_ = external_ms_ >= 800 ? ConversationState::engaged :
                 external_ms_ >= 500 ? ConversationState::anticipating : ConversationState::idle;
    } else if (state_ != ConversationState::idle) {
        hold_ms_ += std::max(0.0f,input.elapsed_ms);
        if (hold_ms_ >= 1500) reset(); else state_=ConversationState::holding;
    } else external_ms_=0;
    PolicyOutput out; out.state=state_;
    if (state_==ConversationState::anticipating) out.media_percent=70;
    else if (state_==ConversationState::engaged || state_==ConversationState::holding) out.media_percent=60;
    out.envelope_enabled = state_==ConversationState::idle;
    return out;
}
}

