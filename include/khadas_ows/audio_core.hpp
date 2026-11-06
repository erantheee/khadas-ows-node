#pragma once
#include "khadas_ows/conversation.hpp"
#include "khadas_ows/ring_buffer.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace khadas_ows {
enum class Mode { passthrough, cpu, npu };
const char* mode_name(Mode mode);
bool parse_mode(const std::string& value, Mode& mode);

#pragma pack(push, 1)
struct PacketHeader {
    std::uint32_t magic = 0x4b4f5753; // KOWS
    std::uint32_t schema = 1;
    std::uint64_t sequence = 0;
    std::uint32_t sample_rate = 44100;
    std::uint16_t channels = 2;
    std::uint16_t reserved = 0;
    std::uint32_t frames = 0;
};
#pragma pack(pop)

/// Worker TCP ports. Defaults are the documented production ports (9201/9202) but can be
/// overridden per instance (tests) or by env (KHADAS_CPU_PORT / KHADAS_NPU_PORT) so that test
/// processes never collide with a running CPU/NPU worker.
struct WorkerPorts {
    std::uint16_t cpu = 9201;
    std::uint16_t npu = 9202;
    static WorkerPorts from_env();
};

struct Status {
    Mode requested = Mode::passthrough;
    Mode actual = Mode::passthrough;
    std::string fallback_reason;
    std::string model_hash;
    std::uint64_t packets = 0;
    std::uint64_t fallbacks = 0;
    std::uint64_t rejected = 0;
    std::size_t buffer_samples = 0;
    double last_rtf = 0.0;
    // Conversation-policy evidence (last ingested frame; see ingest_evidence()).
    conversation::State policy_state = conversation::State::idle;
    bool engaged = false;
    bool ambient_intent = false;
    float media_percent = 100.0f;
    std::uint64_t evidence_frames = 0;
    /// Monotonic milliseconds since AudioCore construction of the last evidence frame; -1 = none.
    std::int64_t last_evidence_ms = -1;
    // Evidence-staleness fail-safe: a stalled/dead out-of-process evidence producer must not
    // hold the media down forever, so a non-idle policy is released to idle/100% once the feed
    // has been quiet for longer than evidence_ttl_ms.
    bool evidence_stale = false;
    int evidence_ttl_ms = 2000;
    std::uint64_t evidence_stale_events = 0;
};

class AudioCore {
public:
    /// evidence_ttl_ms: 0 = read KHADAS_EVIDENCE_TTL_MS, otherwise the documented 2000 ms
    /// default. See Status::evidence_stale.
    explicit AudioCore(std::size_t capacity_samples = 44100 * 2 * 4,
                       WorkerPorts worker_ports = WorkerPorts::from_env(),
                       int evidence_ttl_ms = 0);
    void request_mode(Mode mode);
    void set_model_hash(std::string value);
    Status status() const;
    std::vector<float> process(const PacketHeader& header, const std::vector<float>& input,
                               int timeout_ms = 1500);
    std::string status_json() const;
    /// Ingests one control-rate evidence frame (dt_ms defaults to the 100 ms control period) and
    /// advances the ported legacy conversation state machine. Returns the new policy output.
    conversation::Output ingest_evidence(const conversation::Evidence& evidence,
                                        float dt_ms = 100.0f);
    conversation::Output policy_output() const;
    /// Immutable after construction; exposed for startup logging.
    WorkerPorts worker_ports() const { return worker_ports_; }

private:
    bool call_worker(Mode mode, const PacketHeader& header, const std::vector<float>& input,
                     std::vector<float>& output, int timeout_ms, std::string& error);
    /// Applies the current media duck gain to `block` with a within-block linear ramp so that
    /// target changes cannot click/zipper. Bit-exact bypass (no multiplies) when the policy is
    /// idle and the applied gain is already unity.
    void apply_duck_gain_locked(std::vector<float>& block);
    mutable std::mutex mutex_;
    RingBuffer<float> ring_;
    Status status_;
    WorkerPorts worker_ports_;
    conversation::Awareness awareness_;
    conversation::Output policy_;
    float applied_gain_ = 1.0f;
    int evidence_ttl_ms_ = 2000;
    std::chrono::steady_clock::time_point last_evidence_at_{};
    bool has_evidence_ = false;
    std::chrono::steady_clock::time_point started_at_ = std::chrono::steady_clock::now();
};

/// Both servers block until `stop` is set (poll-based; pass nullptr to never stop). A client
/// disconnecting mid-request/mid-packet only closes that connection: the accept loop survives.
int run_audio_server(AudioCore& core, std::uint16_t port, std::atomic<bool>* stop = nullptr);
int run_http_server(AudioCore& core, std::uint16_t port, std::atomic<bool>* stop = nullptr);
} // namespace khadas_ows
