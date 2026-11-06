#pragma once
namespace khadas_ows {
enum class ConversationState { idle, anticipating, engaged, holding };
struct PolicyInput {
    bool external_speech = false;
    bool self_speech = false;
    bool identity_fresh = false;
    bool identity_self = false;
    bool input_valid = true;
    float elapsed_ms = 0;
};
struct PolicyOutput {
    ConversationState state = ConversationState::idle;
    int media_percent = 100;
    bool envelope_enabled = true;
};
class ConversationPolicy {
public:
    PolicyOutput update(const PolicyInput& input);
    void reset();
private:
    ConversationState state_ = ConversationState::idle;
    float external_ms_ = 0;
    float hold_ms_ = 0;
};
}

