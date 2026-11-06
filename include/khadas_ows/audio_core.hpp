#pragma once
#include "khadas_ows/ring_buffer.hpp"
#include <atomic>
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
};

class AudioCore {
public:
    explicit AudioCore(std::size_t capacity_samples = 44100 * 2 * 4);
    void request_mode(Mode mode);
    void set_model_hash(std::string value);
    Status status() const;
    std::vector<float> process(const PacketHeader& header, const std::vector<float>& input,
                               int timeout_ms = 1500);
    std::string status_json() const;
private:
    bool call_worker(Mode mode, const PacketHeader& header, const std::vector<float>& input,
                     std::vector<float>& output, int timeout_ms, std::string& error);
    mutable std::mutex mutex_;
    RingBuffer<float> ring_;
    Status status_;
};

int run_audio_server(AudioCore& core, std::uint16_t port);
int run_http_server(AudioCore& core, std::uint16_t port);
}

