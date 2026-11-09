#include "khadas_ows/audio_core.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace khadas_ows {
namespace {

constexpr std::uint32_t kMagic = 0x4b4f5753; // KOWS
constexpr int kAcceptPollMs = 50;
// Idle timeout for an accepted client connection. It must exceed the largest gap between two
// packets of one stream: a fixed-shape NPU worker runs a 4.005 s window, so the probe client
// legitimately sends one 4 s packet every 4 s. 1 s (the original value) tore down such a
// connection between packets and looked like "unrecoverable stream loss" on the client.
constexpr int kDefaultClientTimeoutMs = 10000;
constexpr std::size_t kMaxRequestBytes = 64 * 1024;
constexpr std::size_t kMaxPacketSamples = 44100 * 2 * 8; // 8 s of stereo 44.1 kHz

bool read_all(int fd, void* destination, std::size_t bytes) {
    auto* out = static_cast<unsigned char*>(destination);
    while (bytes) {
        const auto n = ::recv(fd, out, bytes, 0);
        if (n <= 0) return false;
        out += n;
        bytes -= static_cast<std::size_t>(n);
    }
    return true;
}

bool write_all(int fd, const void* source, std::size_t bytes) {
    const auto* in = static_cast<const unsigned char*>(source);
    while (bytes) {
        const auto n = ::send(fd, in, bytes, MSG_NOSIGNAL);
        if (n <= 0) return false;
        in += n;
        bytes -= static_cast<std::size_t>(n);
    }
    return true;
}

int connect_local(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

int listen_on(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(fd, 8) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

void set_io_timeout(int fd, int timeout_ms) {
    timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

void set_no_delay(int fd) {
    int yes = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
}

/// Waits until the listening socket has a pending connection or `timeout_ms` elapses.
bool wait_input(int fd, int timeout_ms) {
    pollfd descriptor{fd, POLLIN, 0};
    int ready = 0;
    do {
        ready = ::poll(&descriptor, 1, timeout_ms);
    } while (ready < 0 && errno == EINTR);
    return ready > 0 && (descriptor.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
}

int int_from_env(const char* name, int fallback, int low, int high) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < low || value > high) return fallback;
    return static_cast<int>(value);
}

int client_timeout_from_env() {
    return int_from_env("KHADAS_CLIENT_TIMEOUT_MS", kDefaultClientTimeoutMs, 100, 600000);
}

int worker_timeout_from_env() {
    return int_from_env("KHADAS_WORKER_TIMEOUT_MS", 1500, 1, 60000);
}

int evidence_ttl_from_env() {
    constexpr int kDefaultTtlMs = 2000;
    const char* raw = std::getenv("KHADAS_EVIDENCE_TTL_MS");
    if (raw == nullptr || *raw == '\0') return kDefaultTtlMs;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value <= 0 || value > 3600000) return kDefaultTtlMs;
    return static_cast<int>(value);
}

std::uint16_t env_port(const char* name, std::uint16_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value <= 0 || value > 65535) return fallback;
    return static_cast<std::uint16_t>(value);
}

std::string trim_copy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) ++begin;
    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) --end;
    std::string result = value.substr(begin, end - begin);
    if (result.size() >= 2 && result.front() == '"' && result.back() == '"') {
        result = result.substr(1, result.size() - 2);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Minimal flat-JSON reader for the /v1/evidence body. This is a deliberately
// small subset parser: flat object, numeric/boolean fields, unknown fields
// ignored. A field that is present but has the wrong JSON type makes the whole
// body invalid (400), so a client typo cannot silently change policy state.
// ---------------------------------------------------------------------------
bool json_value_position(const std::string& body, const char* key, std::size_t& value_pos) {
    const std::string needle = std::string("\"") + key + "\"";
    const auto key_pos = body.find(needle);
    if (key_pos == std::string::npos) return false;
    auto pos = body.find(':', key_pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos])) != 0) ++pos;
    value_pos = pos;
    return value_pos < body.size();
}

bool read_bool_field(const std::string& body, const char* key, bool& value) {
    std::size_t pos = 0;
    if (!json_value_position(body, key, pos)) return true; // absent keeps the default
    if (body.compare(pos, 4, "true") == 0) {
        value = true;
        return true;
    }
    if (body.compare(pos, 5, "false") == 0) {
        value = false;
        return true;
    }
    return false;
}

bool read_number_field(const std::string& body, const char* key, float& value) {
    std::size_t pos = 0;
    if (!json_value_position(body, key, pos)) return true; // absent keeps the default
    if (body[pos] == '"') return false;
    char* end = nullptr;
    const double parsed = std::strtod(body.c_str() + pos, &end);
    if (end == body.c_str() + pos || !std::isfinite(parsed) || parsed < -1e9 || parsed > 1e9) {
        return false;
    }
    value = static_cast<float>(parsed);
    return true;
}

bool read_string_field(const std::string& body, const char* key, std::string& value) {
    std::size_t pos = 0;
    if (!json_value_position(body, key, pos)) return true; // absent keeps the default
    if (pos >= body.size() || body[pos] != '"') return false;
    const auto end = body.find('"', pos + 1);
    if (end == std::string::npos) return false;
    value = body.substr(pos + 1, end - pos - 1);
    return true;
}

bool skip_whitespace(const std::string& text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) ++pos;
    return pos < text.size();
}

bool consume_quoted_string(const std::string& text, std::size_t& pos) {
    if (pos >= text.size() || text[pos] != '"') return false;
    ++pos;
    while (pos < text.size()) {
        const char character = text[pos];
        if (character == '\\') {
            pos += 2;
            continue;
        }
        ++pos;
        if (character == '"') return true;
    }
    return false;
}

bool consume_json_scalar(const std::string& text, std::size_t& pos) {
    if (pos >= text.size()) return false;
    const char character = text[pos];
    if (character == '"') return consume_quoted_string(text, pos);
    if (character == 't') {
        if (text.compare(pos, 4, "true") != 0) return false;
        pos += 4;
        return true;
    }
    if (character == 'f') {
        if (text.compare(pos, 5, "false") != 0) return false;
        pos += 5;
        return true;
    }
    if (character == 'n') {
        if (text.compare(pos, 4, "null") != 0) return false;
        pos += 4;
        return true;
    }
    if (character == '{' || character == '[') return false; // flat schema only
    char* end = nullptr;
    (void)std::strtod(text.c_str() + pos, &end);
    if (end == text.c_str() + pos) return false;
    pos = static_cast<std::size_t>(end - text.c_str());
    return true;
}

/// Validates the whole body as a single flat JSON object (scalar values only).
bool valid_flat_json_object(const std::string& body) {
    std::size_t pos = 0;
    (void)skip_whitespace(body, pos);
    if (pos >= body.size() || body[pos] != '{') return false;
    ++pos;
    (void)skip_whitespace(body, pos);
    if (pos < body.size() && body[pos] == '}') {
        ++pos;
        (void)skip_whitespace(body, pos);
        return pos == body.size();
    }
    while (true) {
        (void)skip_whitespace(body, pos);
        if (!consume_quoted_string(body, pos)) return false;
        (void)skip_whitespace(body, pos);
        if (pos >= body.size() || body[pos] != ':') return false;
        ++pos;
        (void)skip_whitespace(body, pos);
        if (!consume_json_scalar(body, pos)) return false;
        (void)skip_whitespace(body, pos);
        if (pos >= body.size()) return false;
        if (body[pos] == ',') {
            ++pos;
            continue;
        }
        if (body[pos] == '}') {
            ++pos;
            (void)skip_whitespace(body, pos);
            return pos == body.size();
        }
        return false;
    }
}

bool parse_evidence_json(const std::string& body, conversation::Evidence& evidence, float& dt_ms) {
    if (!valid_flat_json_object(body)) return false;
    if (!read_bool_field(body, "speech_active", evidence.speech_active)) return false;
    if (!read_number_field(body, "voice_likelihood", evidence.voice_likelihood)) return false;
    if (!read_number_field(body, "conversation_cue", evidence.conversation_cue)) return false;
    if (!read_number_field(body, "own_voice", evidence.own_voice)) return false;
    if (!read_number_field(body, "level_dbfs", evidence.level_dbfs)) return false;
    if (!read_bool_field(body, "event_blocked", evidence.event_blocked)) return false;
    if (!read_bool_field(body, "dialogue_veto", evidence.dialogue_veto)) return false;
    if (!read_string_field(body, "event", evidence.event)) return false;
    if (!read_string_field(body, "environment", evidence.environment)) return false;
    if (!read_string_field(body, "context", evidence.context)) return false;
    if (!read_number_field(body, "context_score", evidence.context_score)) return false;
    if (!read_number_field(body, "elapsed_ms", dt_ms)) return false;
    if (!read_number_field(body, "dt_ms", dt_ms)) return false;
    dt_ms = std::clamp(dt_ms, 0.0f, 1000.0f);
    return true;
}

std::string lowercase_copy(const std::string& value) {
    std::string result = value;
    for (char& character : result) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return result;
}

/// Parses Content-Length out of an HTTP header block; returns 0 when absent.
bool http_content_length(const std::string& request, std::size_t header_end, std::size_t& length) {
    const std::string headers = lowercase_copy(request.substr(0, header_end));
    const auto at = headers.find("content-length:");
    if (at == std::string::npos) {
        length = 0;
        return true;
    }
    auto value_at = at + std::strlen("content-length:");
    while (value_at < headers.size() && (headers[value_at] == ' ' || headers[value_at] == '\t')) ++value_at;
    const auto value_end = headers.find("\r\n", value_at);
    const std::string value = headers.substr(value_at, value_end == std::string::npos
                                                          ? std::string::npos
                                                          : value_end - value_at);
    try {
        const unsigned long parsed = std::stoul(value);
        if (parsed > kMaxRequestBytes) return false;
        length = static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

/// Reads one HTTP request (headers + Content-Length body) from an accepted connection.
bool read_http_request(int fd, std::string& request) {
    char buffer[1024];
    while (request.size() < kMaxRequestBytes) {
        const auto header_end = request.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            std::size_t content_length = 0;
            if (!http_content_length(request, header_end, content_length)) return false;
            if (request.size() >= header_end + 4 + content_length) return true;
        }
        const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) return false; // client closed or stalled before the request was complete
        request.append(buffer, static_cast<std::size_t>(n));
    }
    return false;
}

const char* reason_phrase(int code) {
    switch (code) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        default: return "Error";
    }
}

void handle_http_client(AudioCore& core, int fd) {
    std::string request;
    if (!read_http_request(fd, request)) return;
    const auto header_end = request.find("\r\n\r\n");
    const std::string body =
        header_end == std::string::npos ? std::string() : request.substr(header_end + 4);
    int code = 200;
    std::string payload;
    if (request.rfind("GET /v1/status ", 0) == 0) {
        payload = core.status_json();
    } else if (request.rfind("POST /v1/mode ", 0) == 0) {
        Mode mode = Mode::passthrough;
        if (!parse_mode(trim_copy(body), mode)) {
            code = 400;
            payload = "{\"error\":\"expected passthrough, cpu or npu\"}";
        } else {
            core.request_mode(mode);
            payload = core.status_json();
        }
    } else if (request.rfind("POST /v1/evidence ", 0) == 0) {
        conversation::Evidence evidence;
        float dt_ms = 100.0f;
        if (!parse_evidence_json(body, evidence, dt_ms)) {
            code = 400;
            payload = "{\"error\":\"invalid evidence json\"}";
        } else {
            core.ingest_evidence(evidence, dt_ms);
            payload = core.status_json();
        }
    } else {
        code = 404;
        payload = "{\"error\":\"not found\"}";
    }
    const std::string response = "HTTP/1.1 " + std::to_string(code) + " " + reason_phrase(code) +
                                 "\r\nContent-Type: application/json\r\nContent-Length: " +
                                 std::to_string(payload.size()) +
                                 "\r\nConnection: close\r\n\r\n" + payload;
    write_all(fd, response.data(), response.size());
}

} // namespace

WorkerPorts WorkerPorts::from_env() {
    WorkerPorts ports;
    ports.cpu = env_port("KHADAS_CPU_PORT", ports.cpu);
    ports.npu = env_port("KHADAS_NPU_PORT", ports.npu);
    return ports;
}

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::cpu: return "cpu";
        case Mode::npu: return "npu";
        default: return "passthrough";
    }
}
bool parse_mode(const std::string& value, Mode& mode) {
    if (value == "passthrough") mode = Mode::passthrough;
    else if (value == "cpu") mode = Mode::cpu;
    else if (value == "npu") mode = Mode::npu;
    else return false;
    return true;
}

AudioCore::AudioCore(std::size_t capacity_samples, WorkerPorts worker_ports,
                     int evidence_ttl_ms)
    : ring_(capacity_samples),
      worker_ports_(worker_ports),
      evidence_ttl_ms_(evidence_ttl_ms > 0 ? evidence_ttl_ms : evidence_ttl_from_env()),
      default_worker_timeout_ms_(worker_timeout_from_env()) {
    status_.evidence_ttl_ms = evidence_ttl_ms_;
}

void AudioCore::request_mode(Mode mode) {
    std::lock_guard lock(mutex_);
    status_.requested = mode;
}
void AudioCore::set_model_hash(std::string value) {
    std::lock_guard lock(mutex_);
    status_.model_hash = std::move(value);
}
Status AudioCore::status() const {
    std::lock_guard lock(mutex_);
    return status_;
}
conversation::Output AudioCore::policy_output() const {
    std::lock_guard lock(mutex_);
    return policy_;
}

conversation::Output AudioCore::ingest_evidence(const conversation::Evidence& evidence,
                                                float dt_ms) {
    std::lock_guard lock(mutex_);
    policy_ = awareness_.update(evidence, dt_ms);
    status_.policy_state = policy_.state;
    status_.engaged = policy_.engaged;
    status_.ambient_intent = policy_.ambient_open;
    status_.media_percent = policy_.media_percent;
    status_.dmic_level_dbfs = evidence.level_dbfs;
    status_.dmic_voice_likelihood = evidence.voice_likelihood;
    status_.dmic_speech_active = evidence.speech_active;
    status_.dmic_event = evidence.event;
    status_.dmic_environment = evidence.environment;
    status_.dmic_context = evidence.context;
    status_.dmic_context_score = evidence.context_score;
    ++status_.evidence_frames;
    last_evidence_at_ = std::chrono::steady_clock::now();
    has_evidence_ = true;
    status_.evidence_stale = false;
    status_.last_evidence_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   last_evidence_at_ - started_at_)
                                   .count();
    return policy_;
}

void AudioCore::refresh_evidence_staleness_locked() {
    // Fail-open: evidence arrives from an out-of-process HTTP producer. If that producer
    // stalls or dies while the policy is not idle, holding the last duck forever would leave
    // the media attenuated with no recovery. Release to idle/100% and let the existing gain
    // ramp restore the audio; the next process() block is bit-exact again. Stale evidence is
    // never re-used: the state machine needs fresh frames to re-engage.
    const bool policy_active = policy_.state != conversation::State::idle;
    if (policy_active && has_evidence_ &&
        std::chrono::steady_clock::now() - last_evidence_at_ >
            std::chrono::milliseconds(evidence_ttl_ms_)) {
        awareness_.reset();
        policy_ = conversation::Output{};
        status_.policy_state = policy_.state;
        status_.engaged = policy_.engaged;
        status_.ambient_intent = policy_.ambient_open;
        status_.media_percent = policy_.media_percent;
        status_.evidence_stale = true;
        ++status_.evidence_stale_events;
    }
}

bool AudioCore::call_worker(Mode mode, const PacketHeader& header, const std::vector<float>& input,
                            std::vector<float>& output, int timeout_ms, std::string& error) {
    const std::uint16_t port = mode == Mode::npu ? worker_ports_.npu : worker_ports_.cpu;
    const int fd = connect_local(port);
    if (fd < 0) {
        error = "worker_unavailable";
        return false;
    }
    // Never block forever: a negative/zero timeout would turn poll() into an infinite wait.
    const int bounded_timeout_ms = std::clamp(timeout_ms, 1, 60000);
    set_io_timeout(fd, bounded_timeout_ms);
    const auto started = std::chrono::steady_clock::now();
    bool ok = write_all(fd, &header, sizeof(header)) &&
              write_all(fd, input.data(), input.size() * sizeof(float));
    if (!ok) {
        error = "worker_timeout";
        ::close(fd);
        return false;
    }
    if (!wait_input(fd, bounded_timeout_ms)) {
        error = "worker_timeout";
        ::close(fd);
        return false;
    }
    PacketHeader reply{};
    ok = read_all(fd, &reply, sizeof(reply));
    if (!ok || reply.magic != header.magic || reply.schema != header.schema ||
        reply.frames != header.frames || reply.channels != header.channels ||
        reply.sample_rate != header.sample_rate) {
        error = "worker_protocol";
        ::close(fd);
        return false;
    }
    output.resize(input.size());
    ok = read_all(fd, output.data(), output.size() * sizeof(float));
    ::close(fd);
    if (!ok) {
        error = "worker_short_read";
        return false;
    }
    for (float value : output) {
        if (!std::isfinite(value)) {
            error = "worker_non_finite";
            return false;
        }
    }
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::lock_guard lock(mutex_);
    status_.last_rtf = elapsed / (static_cast<double>(header.frames) / header.sample_rate);
    return true;
}

void AudioCore::apply_duck_gain_locked(std::vector<float>& block) {
    // The conversation policy is the source of truth. Once it returns to idle the duck is over,
    // so the target is unity (the module only leaves releasing for idle after its own media ramp
    // is back at ~99%, and the per-block linear ramp below smooths that last step). While a
    // conversation state is active the target follows the policy's already-ramped media_percent;
    // percentages at the 99.99% asymptote snap to unity so steady passthrough stays bit-exact.
    float target = 1.0f;
    if (policy_.state != conversation::State::idle) {
        target = std::clamp(policy_.media_percent / 100.0f, 0.0f, 1.0f);
        if (target >= 0.9999f) target = 1.0f;
    }
    const float start = applied_gain_;
    if (start == 1.0f && target == 1.0f) return; // bit-exact bypass
    const std::size_t samples = block.size();
    if (samples == 0) {
        applied_gain_ = target;
        return;
    }
    // Linear within-block ramp between the previous and current target gain: every block
    // boundary is continuous, so a target change cannot produce a click or zipper artifact.
    const float step = (target - start) / static_cast<float>(samples);
    float gain = start;
    for (std::size_t i = 0; i < samples; ++i) {
        gain += step;
        block[i] *= gain;
    }
    applied_gain_ = target;
}

std::vector<float> AudioCore::process(const PacketHeader& header, const std::vector<float>& input,
                                      int timeout_ms) {
    if (timeout_ms <= 0) timeout_ms = default_worker_timeout_ms_;
    const auto expected = static_cast<std::size_t>(header.frames) * header.channels;
    if (header.magic != kMagic || header.schema != 1 || header.channels != 2 ||
        header.sample_rate != 44100 || header.frames == 0 || input.size() != expected) {
        std::lock_guard lock(mutex_);
        ++status_.rejected;
        status_.fallback_reason = "invalid_packet";
        return {};
    }

    std::vector<float> block(input.size(), 0.0f);
    Mode requested = Mode::passthrough;
    const char* ring_fallback = nullptr;
    {
        std::lock_guard lock(mutex_);
        ++status_.packets;
        if (ring_.push(input.data(), input.size()) != input.size()) {
            // The packet does not fit the bounded ring: flush it and pass the audio through
            // without buffering rather than dropping it.
            ring_.clear();
            ++status_.fallbacks;
            ring_fallback = "ring_overflow";
            block = input;
        } else if (ring_.pop(block.data(), block.size()) != block.size()) {
            ring_.clear();
            ++status_.fallbacks;
            ring_fallback = "ring_underflow";
            block = input;
        }
        status_.buffer_samples = ring_.size();
        requested = status_.requested;
    }

    std::vector<float> result;
    std::string error;
    if (requested == Mode::passthrough) {
        result = std::move(block);
        std::lock_guard lock(mutex_);
        status_.actual = Mode::passthrough;
        status_.fallback_reason = ring_fallback == nullptr ? std::string() : ring_fallback;
    } else if (call_worker(requested, header, block, result, timeout_ms, error)) {
        std::lock_guard lock(mutex_);
        status_.actual = requested;
        status_.fallback_reason = ring_fallback == nullptr ? std::string() : ring_fallback;
    } else {
        result = std::move(block);
        std::lock_guard lock(mutex_);
        status_.actual = Mode::passthrough;
        status_.fallback_reason = error;
        ++status_.fallbacks;
    }
    {
        std::lock_guard lock(mutex_);
        // Fail-open on stalled evidence; shared with status_json() so a status poll can release
        // the duck even when no audio packet is flowing.
        refresh_evidence_staleness_locked();
        apply_duck_gain_locked(result);
    }
    return result;
}

std::string AudioCore::status_json() {
    {
        std::lock_guard lock(mutex_);
        refresh_evidence_staleness_locked();
    }
    const auto snapshot = status();
    std::int64_t evidence_age_ms = -1;
    if (snapshot.last_evidence_ms >= 0) {
        const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started_at_)
                                .count();
        evidence_age_ms = std::max<std::int64_t>(0, now_ms - snapshot.last_evidence_ms);
    }
    std::ostringstream out;
    out << "{\"schema\":1,\"requested_mode\":\"" << mode_name(snapshot.requested)
        << "\",\"actual_mode\":\"" << mode_name(snapshot.actual)
        << "\",\"fallback_reason\":\"" << snapshot.fallback_reason
        << "\",\"model_hash\":\"" << snapshot.model_hash << "\",\"packets\":" << snapshot.packets
        << ",\"fallbacks\":" << snapshot.fallbacks << ",\"rejected\":" << snapshot.rejected
        << ",\"buffer_samples\":" << snapshot.buffer_samples << ",\"last_rtf\":" << snapshot.last_rtf
        << ",\"policy_state\":\"" << conversation::state_name(snapshot.policy_state) << "\""
        << ",\"engaged\":" << (snapshot.engaged ? "true" : "false")
        << ",\"ambient_intent\":" << (snapshot.ambient_intent ? "true" : "false")
        << ",\"media_percent\":" << snapshot.media_percent
        << ",\"evidence_frames\":" << snapshot.evidence_frames
        << ",\"last_evidence_ms\":" << snapshot.last_evidence_ms
        << ",\"last_evidence_age_ms\":" << evidence_age_ms
        << ",\"evidence_stale\":" << (snapshot.evidence_stale ? "true" : "false")
        << ",\"evidence_ttl_ms\":" << snapshot.evidence_ttl_ms
        << ",\"evidence_stale_events\":" << snapshot.evidence_stale_events
        << ",\"dmic_level_dbfs\":" << snapshot.dmic_level_dbfs
        << ",\"dmic_voice_likelihood\":" << snapshot.dmic_voice_likelihood
        << ",\"dmic_speech_active\":" << (snapshot.dmic_speech_active ? "true" : "false")
        << ",\"dmic_event\":\"" << snapshot.dmic_event << "\""
        << ",\"dmic_environment\":\"" << snapshot.dmic_environment << "\""
        << ",\"dmic_context\":\"" << snapshot.dmic_context << "\""
        << ",\"dmic_context_score\":" << snapshot.dmic_context_score << "}";
    return out.str();
}

int run_audio_server(AudioCore& core, std::uint16_t port, std::atomic<bool>* stop) {
    const int server = listen_on(port);
    if (server < 0) return 1;
    const int client_timeout_ms = client_timeout_from_env();
    std::vector<unsigned char> wire; // reused across packets/clients: no per-packet allocation
    while (stop == nullptr || !stop->load()) {
        if (!wait_input(server, kAcceptPollMs)) continue;
        const int client = ::accept(server, nullptr, nullptr);
        if (client < 0) continue;
        set_io_timeout(client, client_timeout_ms);
        set_no_delay(client);
        PacketHeader header{};
        while (read_all(client, &header, sizeof(header))) {
            const auto samples = static_cast<std::size_t>(header.frames) * header.channels;
            if (samples > kMaxPacketSamples) break; // refuse to allocate; drop the connection
            std::vector<float> input(samples);
            if (!read_all(client, input.data(), samples * sizeof(float))) break;
            const auto output = core.process(header, input);
            if (output.empty()) break; // invalid packet: process() already counted it
            wire.resize(sizeof(PacketHeader) + output.size() * sizeof(float));
            std::memcpy(wire.data(), &header, sizeof(header));
            std::memcpy(wire.data() + sizeof(PacketHeader), output.data(),
                        output.size() * sizeof(float));
            if (!write_all(client, wire.data(), wire.size())) break;
        }
        ::close(client);
    }
    ::close(server);
    return 0;
}

int run_http_server(AudioCore& core, std::uint16_t port, std::atomic<bool>* stop) {
    const int server = listen_on(port);
    if (server < 0) return 1;
    const int client_timeout_ms = client_timeout_from_env();
    while (stop == nullptr || !stop->load()) {
        if (!wait_input(server, kAcceptPollMs)) continue;
        const int client = ::accept(server, nullptr, nullptr);
        if (client < 0) continue;
        set_io_timeout(client, client_timeout_ms);
        handle_http_client(core, client);
        ::close(client);
    }
    ::close(server);
    return 0;
}

} // namespace khadas_ows
