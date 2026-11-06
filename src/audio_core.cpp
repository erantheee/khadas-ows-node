#include "khadas_ows/audio_core.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <netinet/in.h>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>

namespace khadas_ows {
namespace {
bool read_all(int fd, void* destination, std::size_t bytes) {
    auto* out = static_cast<unsigned char*>(destination);
    while (bytes) {
        const auto n = ::recv(fd, out, bytes, 0);
        if (n <= 0) return false;
        out += n; bytes -= static_cast<std::size_t>(n);
    }
    return true;
}
bool write_all(int fd, const void* source, std::size_t bytes) {
    const auto* in = static_cast<const unsigned char*>(source);
    while (bytes) {
        const auto n = ::send(fd, in, bytes, MSG_NOSIGNAL);
        if (n <= 0) return false;
        in += n; bytes -= static_cast<std::size_t>(n);
    }
    return true;
}
int connect_local(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd); return -1;
    }
    return fd;
}
int listen_on(std::uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1; ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(fd, 8) != 0) { ::close(fd); return -1; }
    return fd;
}
}

const char* mode_name(Mode mode) {
    switch (mode) { case Mode::cpu: return "cpu"; case Mode::npu: return "npu"; default: return "passthrough"; }
}
bool parse_mode(const std::string& value, Mode& mode) {
    if (value == "passthrough") mode = Mode::passthrough;
    else if (value == "cpu") mode = Mode::cpu;
    else if (value == "npu") mode = Mode::npu;
    else return false;
    return true;
}
AudioCore::AudioCore(std::size_t capacity_samples) : ring_(capacity_samples) {}
void AudioCore::request_mode(Mode mode) { std::lock_guard lock(mutex_); status_.requested = mode; }
void AudioCore::set_model_hash(std::string value) { std::lock_guard lock(mutex_); status_.model_hash = std::move(value); }
Status AudioCore::status() const { std::lock_guard lock(mutex_); return status_; }

bool AudioCore::call_worker(Mode mode, const PacketHeader& header, const std::vector<float>& input,
                            std::vector<float>& output, int timeout_ms, std::string& error) {
    const int fd = connect_local(mode == Mode::npu ? 9202 : 9201);
    if (fd < 0) { error = "worker_unavailable"; return false; }
    const auto started = std::chrono::steady_clock::now();
    bool ok = write_all(fd, &header, sizeof(header)) &&
              write_all(fd, input.data(), input.size() * sizeof(float));
    pollfd pfd{fd, POLLIN, 0};
    if (!ok || ::poll(&pfd, 1, timeout_ms) <= 0) { error = "worker_timeout"; ::close(fd); return false; }
    PacketHeader reply{};
    ok = read_all(fd, &reply, sizeof(reply));
    if (!ok || reply.magic != header.magic || reply.frames != header.frames || reply.channels != header.channels) {
        error = "worker_protocol"; ::close(fd); return false;
    }
    output.resize(input.size());
    ok = read_all(fd, output.data(), output.size() * sizeof(float));
    ::close(fd);
    if (!ok) { error = "worker_short_read"; return false; }
    for (float value : output) if (!std::isfinite(value)) { error = "worker_non_finite"; return false; }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::lock_guard lock(mutex_);
    status_.last_rtf = elapsed / (static_cast<double>(header.frames) / header.sample_rate);
    return true;
}

std::vector<float> AudioCore::process(const PacketHeader& header, const std::vector<float>& input,
                                      int timeout_ms) {
    const auto expected = static_cast<std::size_t>(header.frames) * header.channels;
    if (header.magic != PacketHeader{}.magic || header.schema != 1 || header.channels != 2 ||
        header.sample_rate != 44100 || input.size() != expected) {
        std::lock_guard lock(mutex_); ++status_.rejected; status_.fallback_reason = "invalid_packet"; return {};
    }
    if (ring_.push(input.data(), input.size()) != input.size()) {
        ring_.clear(); std::lock_guard lock(mutex_); ++status_.fallbacks; status_.fallback_reason = "ring_overflow";
    }
    std::vector<float> block(input.size());
    ring_.pop(block.data(), block.size());
    Mode requested; { std::lock_guard lock(mutex_); requested = status_.requested; ++status_.packets; status_.buffer_samples = ring_.size(); }
    if (requested == Mode::passthrough) {
        std::lock_guard lock(mutex_); status_.actual = Mode::passthrough; status_.fallback_reason.clear(); return block;
    }
    std::vector<float> processed; std::string error;
    if (call_worker(requested, header, block, processed, timeout_ms, error)) {
        std::lock_guard lock(mutex_); status_.actual = requested; status_.fallback_reason.clear(); return processed;
    }
    std::lock_guard lock(mutex_); status_.actual = Mode::passthrough; status_.fallback_reason = error; ++status_.fallbacks;
    return block;
}

std::string AudioCore::status_json() const {
    const auto s = status(); std::ostringstream out;
    out << "{\"schema\":1,\"requested_mode\":\"" << mode_name(s.requested)
        << "\",\"actual_mode\":\"" << mode_name(s.actual)
        << "\",\"fallback_reason\":\"" << s.fallback_reason
        << "\",\"model_hash\":\"" << s.model_hash << "\",\"packets\":" << s.packets
        << ",\"fallbacks\":" << s.fallbacks << ",\"rejected\":" << s.rejected
        << ",\"buffer_samples\":" << s.buffer_samples << ",\"last_rtf\":" << s.last_rtf << "}";
    return out.str();
}

int run_audio_server(AudioCore& core, std::uint16_t port) {
    const int server = listen_on(port); if (server < 0) return 1;
    while (true) {
        const int client = ::accept(server, nullptr, nullptr); if (client < 0) continue;
        PacketHeader header{};
        while (read_all(client, &header, sizeof(header))) {
            const auto samples = static_cast<std::size_t>(header.frames) * header.channels;
            if (samples > 44100 * 2 * 8) break;
            std::vector<float> input(samples);
            if (!read_all(client, input.data(), samples * sizeof(float))) break;
            const auto output = core.process(header, input);
            if (output.empty() || !write_all(client, &header, sizeof(header)) ||
                !write_all(client, output.data(), output.size() * sizeof(float))) break;
        }
        ::close(client);
    }
}

int run_http_server(AudioCore& core, std::uint16_t port) {
    const int server = listen_on(port); if (server < 0) return 1;
    while (true) {
        const int client = ::accept(server, nullptr, nullptr); if (client < 0) continue;
        char buffer[4096]{}; const auto n = ::recv(client, buffer, sizeof(buffer) - 1, 0);
        std::string request(buffer, n > 0 ? static_cast<std::size_t>(n) : 0), body; int code = 200;
        const auto header_end = request.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            std::size_t content_length = 0;
            const auto length_at = request.find("Content-Length:");
            if (length_at != std::string::npos && length_at < header_end) {
                const auto value_at = length_at + std::strlen("Content-Length:");
                const auto value_end = request.find("\r\n", value_at);
                try { content_length = static_cast<std::size_t>(std::stoul(request.substr(value_at, value_end - value_at))); }
                catch (const std::exception&) { content_length = 0; }
            }
            const auto have = request.size() - (header_end + 4);
            if (content_length > have) {
                std::string remainder(content_length - have, '\0');
                if (read_all(client, remainder.data(), remainder.size())) request += remainder;
            }
        }
        if (request.rfind("GET /v1/status ", 0) == 0) body = core.status_json();
        else if (request.rfind("POST /v1/mode ", 0) == 0) {
            const auto pos = request.find("\r\n\r\n"); const auto payload = pos == std::string::npos ? "" : request.substr(pos + 4);
            Mode mode; if (!parse_mode(payload, mode)) { code = 400; body = "{\"error\":\"expected passthrough, cpu or npu\"}"; }
            else { core.request_mode(mode); body = core.status_json(); }
        } else { code = 404; body = "{\"error\":\"not found\"}"; }
        const std::string response = "HTTP/1.1 " + std::to_string(code) + (code == 200 ? " OK" : " Error") +
            "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
            "\r\nConnection: close\r\n\r\n" + body;
        write_all(client, response.data(), response.size()); ::close(client);
    }
}
}

