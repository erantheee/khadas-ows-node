// Deterministic in-process failure matrix for the C++ audio core.
//
// Everything here is local loopback / fake workers; no external network, no sleeps above
// 200 ms, and the whole executable is expected to finish in well under one second.
//
// Covered: ring buffer full/empty/wrap/underflow, packet larger than the bounded ring, worker
// absent / hung / non-finite output / wrong reply header / short read, invalid packet counting,
// passthrough->cpu->npu->passthrough churn with workers going up and down (recovery), real
// loopback HTTP for /v1/status, /v1/mode, /v1/evidence, media duck gain + bit-exact recovery,
// mid-packet disconnects, aborted HTTP connections and concurrent status requests.
#include "khadas_ows/audio_core.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace khadas_ows;

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string& name) {
    if (condition) {
        ++passed;
        std::cout << "PASS " << name << '\n';
    } else {
        ++failed;
        std::cout << "FAIL " << name << '\n';
    }
}

int connect_port(std::uint16_t port) {
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
    timeval tv{2, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    return fd;
}

bool read_exact(int fd, void* destination, std::size_t bytes) {
    auto* out = static_cast<unsigned char*>(destination);
    while (bytes > 0) {
        const auto n = ::recv(fd, out, bytes, 0);
        if (n <= 0) return false;
        out += n;
        bytes -= static_cast<std::size_t>(n);
    }
    return true;
}

bool write_exact(int fd, const void* source, std::size_t bytes) {
    const auto* in = static_cast<const unsigned char*>(source);
    while (bytes > 0) {
        const auto n = ::send(fd, in, bytes, MSG_NOSIGNAL);
        if (n <= 0) return false;
        in += n;
        bytes -= static_cast<std::size_t>(n);
    }
    return true;
}

/// Reserves an ephemeral loopback port; the caller binds it immediately afterwards.
std::uint16_t reserve_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
    int yes = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        return 0;
    }
    socklen_t length = sizeof(address);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length);
    ::close(fd);
    return ntohs(address.sin_port);
}

bool wait_listener(std::uint16_t port, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        const int fd = connect_port(port);
        if (fd >= 0) {
            ::close(fd);
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

std::vector<float> ramp_samples(std::size_t count) {
    std::vector<float> samples(count);
    for (std::size_t i = 0; i < count; ++i) {
        samples[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.4f;
    }
    return samples;
}

// ---------------------------------------------------------------------------
// Fake worker: a loopback listener that imitates the Python worker protocol and can misbehave
// in each of the ways the failure matrix needs.
// ---------------------------------------------------------------------------
class FakeWorker {
public:
    enum class Behavior {
        echo_ok,
        hang,
        reply_non_finite,
        reply_bad_header,
        reply_short_read,
    };

    FakeWorker(std::uint16_t port, Behavior behavior) : behavior_(behavior) {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) return;
        int yes = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(listen_fd_, 8) != 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
            return;
        }
        thread_ = std::thread([this] { serve(); });
    }
    ~FakeWorker() { stop(); }

    FakeWorker(const FakeWorker&) = delete;
    FakeWorker& operator=(const FakeWorker&) = delete;

    bool ok() const { return listen_fd_ >= 0; }

    void stop() {
        if (stopped_.load()) return;
        stopped_.store(true);
        if (thread_.joinable()) thread_.join();
        if (listen_fd_ >= 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
    }

private:
    void serve() {
        while (!stopped_.load()) {
            pollfd descriptor{listen_fd_, POLLIN, 0};
            if (::poll(&descriptor, 1, 10) <= 0) continue;
            const int client = ::accept(listen_fd_, nullptr, nullptr);
            if (client < 0) continue;
            handle(client);
            ::close(client);
        }
    }

    void handle(int client) {
        PacketHeader request{};
        if (!read_exact(client, &request, sizeof(request))) return;
        const std::size_t samples = static_cast<std::size_t>(request.frames) * request.channels;
        if (samples > 44100 * 2 * 4) return; // test-only guard
        std::vector<float> input(samples);
        if (samples > 0 && !read_exact(client, input.data(), samples * sizeof(float))) return;
        if (behavior_ == Behavior::hang) {
            while (!stopped_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            return;
        }
        PacketHeader reply = request;
        std::vector<float> output = input;
        if (behavior_ == Behavior::reply_non_finite) {
            output.assign(samples, std::numeric_limits<float>::quiet_NaN());
        } else if (behavior_ == Behavior::reply_bad_header) {
            reply.frames = request.frames + 1;
        } else if (behavior_ == Behavior::reply_short_read) {
            (void)write_exact(client, &reply, sizeof(reply));
            if (samples > 1) {
                (void)write_exact(client, output.data(), (samples / 2) * sizeof(float));
            }
            return;
        }
        (void)write_exact(client, &reply, sizeof(reply));
        (void)write_exact(client, output.data(), output.size() * sizeof(float));
    }

    Behavior behavior_;
    int listen_fd_ = -1;
    std::atomic<bool> stopped_{false};
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// Small HTTP client for the loopback server tests.
// ---------------------------------------------------------------------------
struct HttpReply {
    int code = -1;
    std::string body;
};

HttpReply http_request(std::uint16_t port, const std::string& method, const std::string& path,
                       const std::string& body) {
    HttpReply reply;
    const int fd = connect_port(port);
    if (fd < 0) return reply;
    const std::string request = method + " " + path +
                                " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n"
                                "Content-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;
    if (!write_exact(fd, request.data(), request.size())) {
        ::close(fd);
        return reply;
    }
    std::string raw;
    char buffer[512];
    while (raw.size() < (1U << 20)) {
        const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) break;
        raw.append(buffer, static_cast<std::size_t>(n));
    }
    ::close(fd);
    const auto first_space = raw.find(' ');
    if (first_space == std::string::npos || first_space + 3 >= raw.size()) return reply;
    reply.code = std::atoi(raw.c_str() + first_space + 1);
    const auto header_end = raw.find("\r\n\r\n");
    if (header_end != std::string::npos) reply.body = raw.substr(header_end + 4);
    return reply;
}

bool json_number(const std::string& json, const char* key, double& value) {
    const std::string needle = std::string("\"") + key + "\":";
    auto at = json.find(needle);
    if (at == std::string::npos) return false;
    at += needle.size();
    const char* begin = json.c_str() + at;
    char* end = nullptr;
    const double parsed = std::strtod(begin, &end);
    if (end == begin) return false;
    value = parsed;
    return true;
}

bool json_string(const std::string& json, const char* key, std::string& value) {
    const std::string needle = std::string("\"") + key + "\":\"";
    auto at = json.find(needle);
    if (at == std::string::npos) return false;
    at += needle.size();
    const auto end = json.find('"', at);
    if (end == std::string::npos) return false;
    value = json.substr(at, end - at);
    return true;
}

bool json_bool(const std::string& json, const char* key, bool& value) {
    const std::string needle = std::string("\"") + key + "\":";
    auto at = json.find(needle);
    if (at == std::string::npos) return false;
    at += needle.size();
    if (json.compare(at, 4, "true") == 0) {
        value = true;
        return true;
    }
    if (json.compare(at, 5, "false") == 0) {
        value = false;
        return true;
    }
    return false;
}

bool audio_round_trip(std::uint16_t port, const PacketHeader& request_header,
                      const std::vector<float>& input, PacketHeader& reply_header,
                      std::vector<float>& output) {
    const int fd = connect_port(port);
    if (fd < 0) return false;
    bool ok = write_exact(fd, &request_header, sizeof(request_header));
    if (ok && !input.empty()) ok = write_exact(fd, input.data(), input.size() * sizeof(float));
    if (ok) ok = read_exact(fd, &reply_header, sizeof(reply_header));
    if (ok) {
        output.assign(input.size(), 0.0f);
        if (!output.empty()) ok = read_exact(fd, output.data(), output.size() * sizeof(float));
    }
    ::close(fd);
    return ok;
}

// ---------------------------------------------------------------------------
// Test sections
// ---------------------------------------------------------------------------
void test_ring_buffer() {
    RingBuffer<float> ring(4);
    const float values[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    check(ring.push(values, 5) == 4, "ring: full push accepts only the capacity");
    check(ring.size() == 4 && ring.capacity() == 4, "ring: size clamps at capacity");
    check(ring.push(values, 1) == 0, "ring: push into a full buffer is rejected");
    float out[4]{};
    check(ring.pop(out, 2) == 2 && out[0] == 1.0f && out[1] == 2.0f,
          "ring: pop preserves FIFO order");
    check(ring.push(values, 1) == 1, "ring: push resumes after a partial pop");
    check(ring.pop(out, 4) == 3 && out[0] == 3.0f && out[1] == 4.0f && out[2] == 1.0f,
          "ring: wrapped pop preserves order");
    check(ring.size() == 0, "ring: pop empties the buffer");
    check(ring.pop(out, 1) == 0, "ring: pop from an empty buffer underflows safely");
    (void)ring.push(values, 3);
    ring.clear();
    check(ring.size() == 0 && ring.pop(out, 1) == 0, "ring: clear returns to the empty state");
}

void test_ring_overflow_via_core() {
    const WorkerPorts ports{reserve_port(), reserve_port()};
    AudioCore small(8, ports); // 8 samples of ring vs a 16-sample packet
    PacketHeader header;
    header.frames = 8;
    const auto input = ramp_samples(16);
    const auto output = small.process(header, input);
    const auto status = small.status();
    check(output == input, "ring-overflow: oversized packet bypasses buffering bit-exactly");
    check(status.fallbacks == 1, "ring-overflow: overflow counted as a fallback");
    check(status.fallback_reason == "ring_overflow", "ring-overflow: reason recorded");
    check(status.rejected == 0, "ring-overflow: not counted as a rejected packet");
    check(status.actual == Mode::passthrough && status.buffer_samples == 0,
          "ring-overflow: passthrough continues and the buffer is cleared");

    AudioCore exact(16, ports); // exactly one packet of capacity
    check(exact.process(header, input) == input, "ring-exact: first packet is bit-exact");
    check(exact.process(header, input) == input, "ring-exact: repeated packets are bit-exact");
    const auto exact_status = exact.status();
    check(exact_status.fallbacks == 0 && exact_status.rejected == 0,
          "ring-exact: no false fallback at exact capacity");
    check(exact_status.buffer_samples == 0, "ring-exact: buffer is drained after every packet");
}

void test_worker_absent() {
    AudioCore core(4096, WorkerPorts{reserve_port(), reserve_port()});
    PacketHeader header;
    header.frames = 16;
    const auto input = ramp_samples(32);
    core.request_mode(Mode::cpu);
    const auto output = core.process(header, input, 100);
    const auto status = core.status();
    check(output == input, "worker-absent: fallback is bit-exact");
    check(status.actual == Mode::passthrough && status.fallback_reason == "worker_unavailable",
          "worker-absent: fallback_reason is worker_unavailable");
    check(status.fallbacks == 1, "worker-absent: fallback counted once");
}

void test_worker_timeout() {
    const auto port = reserve_port();
    FakeWorker worker(port, FakeWorker::Behavior::hang);
    check(worker.ok(), "worker-hang: fake worker is listening");
    AudioCore core(4096, WorkerPorts{port, reserve_port()});
    PacketHeader header;
    header.frames = 16;
    const auto input = ramp_samples(32);
    core.request_mode(Mode::cpu);
    const auto started = std::chrono::steady_clock::now();
    const auto output = core.process(header, input, 100);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    const auto status = core.status();
    check(output == input, "worker-hang: timeout fallback is bit-exact");
    check(status.actual == Mode::passthrough && status.fallback_reason == "worker_timeout",
          "worker-hang: fallback_reason is worker_timeout");
    check(elapsed_ms >= 50 && elapsed_ms < 1000,
          "worker-hang: fallback happens at the configured deadline, not later");
}

void test_worker_non_finite() {
    const auto port = reserve_port();
    FakeWorker worker(port, FakeWorker::Behavior::reply_non_finite);
    check(worker.ok(), "worker-non-finite: fake worker is listening");
    AudioCore core(4096, WorkerPorts{port, reserve_port()});
    PacketHeader header;
    header.frames = 16;
    const auto input = ramp_samples(32);
    core.request_mode(Mode::cpu);
    const auto output = core.process(header, input, 200);
    const auto status = core.status();
    check(output == input, "worker-non-finite: fallback is bit-exact");
    check(status.fallback_reason == "worker_non_finite",
          "worker-non-finite: NaN reply is rejected with worker_non_finite");
    check(status.fallbacks == 1 && status.actual == Mode::passthrough,
          "worker-non-finite: fallback counted and mode reported");
}

void test_worker_protocol() {
    const auto port = reserve_port();
    FakeWorker worker(port, FakeWorker::Behavior::reply_bad_header);
    check(worker.ok(), "worker-protocol: fake worker is listening");
    AudioCore core(4096, WorkerPorts{port, reserve_port()});
    PacketHeader header;
    header.frames = 16;
    const auto input = ramp_samples(32);
    core.request_mode(Mode::cpu);
    const auto output = core.process(header, input, 200);
    const auto status = core.status();
    check(output == input, "worker-protocol: malformed reply falls back bit-exactly");
    check(status.fallback_reason == "worker_protocol",
          "worker-protocol: wrong reply header shape is rejected");
    check(status.fallbacks == 1, "worker-protocol: fallback counted once");
}

void test_worker_short_read() {
    const auto port = reserve_port();
    FakeWorker worker(port, FakeWorker::Behavior::reply_short_read);
    check(worker.ok(), "worker-short: fake worker is listening");
    AudioCore core(4096, WorkerPorts{port, reserve_port()});
    PacketHeader header;
    header.frames = 16;
    const auto input = ramp_samples(32);
    core.request_mode(Mode::cpu);
    const auto output = core.process(header, input, 200);
    const auto status = core.status();
    check(output == input, "worker-short: truncated reply falls back bit-exactly");
    check(status.fallback_reason == "worker_short_read",
          "worker-short: truncation is reported as worker_short_read");
    check(status.fallbacks == 1, "worker-short: fallback counted once");
}

void test_mode_churn_and_recovery() {
    const auto cpu_port = reserve_port();
    const auto npu_port = reserve_port();
    AudioCore core(4096, WorkerPorts{cpu_port, npu_port});
    PacketHeader header;
    header.frames = 32;
    const auto input = ramp_samples(64);

    // CPU worker down.
    core.request_mode(Mode::cpu);
    check(core.process(header, input, 100) == input, "churn: cpu without worker is bit-exact");
    check(core.status().fallback_reason == "worker_unavailable",
          "churn: cpu without worker reports worker_unavailable");

    // CPU worker up -> passthrough->cpu switch is clean.
    auto cpu_worker = std::make_unique<FakeWorker>(cpu_port, FakeWorker::Behavior::echo_ok);
    check(cpu_worker->ok(), "churn: cpu fake worker started");
    core.request_mode(Mode::cpu);
    check(core.process(header, input, 100) == input, "churn: cpu worker echo is bit-exact");
    check(core.status().actual == Mode::cpu && core.status().fallback_reason.empty(),
          "churn: actual mode is cpu while the worker is up");

    // NPU worker down, then up.
    core.request_mode(Mode::npu);
    check(core.process(header, input, 100) == input, "churn: npu without worker is bit-exact");
    check(core.status().fallback_reason == "worker_unavailable",
          "churn: npu without worker reports worker_unavailable");
    auto npu_worker = std::make_unique<FakeWorker>(npu_port, FakeWorker::Behavior::echo_ok);
    check(npu_worker->ok(), "churn: npu fake worker started");
    core.request_mode(Mode::npu);
    check(core.process(header, input, 100) == input, "churn: npu worker echo is bit-exact");
    check(core.status().actual == Mode::npu, "churn: actual mode is npu while the worker is up");

    // Back to passthrough: no worker involved, no fallback on the transition packet.
    {
        const auto before = core.status();
        core.request_mode(Mode::passthrough);
        check(core.process(header, input) == input, "churn: passthrough switch is bit-exact");
        const auto after = core.status();
        check(after.actual == Mode::passthrough && after.fallback_reason.empty(),
              "churn: switching to passthrough clears the fallback reason");
        check(after.fallbacks == before.fallbacks,
              "churn: switching to passthrough adds no fallback");
    }

    // Rapid passthrough -> cpu -> npu churn while both workers are alive.
    {
        const auto before = core.status();
        bool all_bit_exact = true;
        bool all_sized = true;
        for (int i = 0; i < 30; ++i) {
            const Mode mode =
                i % 3 == 0 ? Mode::passthrough : (i % 3 == 1 ? Mode::cpu : Mode::npu);
            core.request_mode(mode);
            const auto out = core.process(header, input, 100);
            all_sized = all_sized && out.size() == input.size();
            all_bit_exact = all_bit_exact && out == input;
        }
        const auto after = core.status();
        check(all_sized && all_bit_exact, "churn: 30 rapid mode switches keep bit-exact echo");
        check(after.packets == before.packets + 30,
              "churn: every churned packet is counted exactly once");
        check(after.fallbacks == before.fallbacks, "churn: no fallbacks while workers are up");
        check(after.actual == Mode::npu && after.requested == Mode::npu,
              "churn: mode converges to the last request");
    }

    // Worker loss after a live session, then an automatic recovery without touching the core.
    cpu_worker.reset();
    {
        const auto before = core.status();
        core.request_mode(Mode::cpu);
        check(core.process(header, input, 100) == input,
              "recovery: losing the worker mid-session falls back bit-exactly");
        const auto after = core.status();
        check(after.actual == Mode::passthrough &&
                  after.fallback_reason == "worker_unavailable",
              "recovery: lost worker reports worker_unavailable");
        check(after.fallbacks == before.fallbacks + 1, "recovery: loss counted once");
    }
    cpu_worker = std::make_unique<FakeWorker>(cpu_port, FakeWorker::Behavior::echo_ok);
    check(cpu_worker->ok(), "recovery: cpu worker restarted on the same port");
    core.request_mode(Mode::cpu);
    check(core.process(header, input, 100) == input, "recovery: restarted worker echo is bit-exact");
    const auto recovered = core.status();
    check(recovered.actual == Mode::cpu && recovered.fallback_reason.empty(),
          "recovery: actual mode converges back to cpu");

    // Counters must stay monotonic through the whole churn.
    check(recovered.packets == 37 && recovered.fallbacks == 3 && recovered.rejected == 0,
          "churn: packet/fallback/rejected counters stay sane");
}

void test_invalid_packets() {
    AudioCore core(4096, WorkerPorts{reserve_port(), reserve_port()});
    PacketHeader header;
    header.frames = 16;
    const auto input = ramp_samples(32);
    auto expect_rejected = [&](const PacketHeader& bad, const std::vector<float>& payload,
                               const std::string& name) {
        const auto before = core.status().rejected;
        const auto output = core.process(bad, payload);
        const auto after = core.status();
        check(output.empty(), "invalid:" + name + " returns no audio");
        check(after.rejected == before + 1, "invalid:" + name + " rejection is counted");
    };
    PacketHeader bad_magic = header;
    bad_magic.magic = 0;
    expect_rejected(bad_magic, input, "bad-magic");
    PacketHeader bad_schema = header;
    bad_schema.schema = 2;
    expect_rejected(bad_schema, input, "bad-schema");
    PacketHeader bad_channels = header;
    bad_channels.channels = 1;
    expect_rejected(bad_channels, input, "bad-channels");
    PacketHeader bad_rate = header;
    bad_rate.sample_rate = 48000;
    expect_rejected(bad_rate, input, "bad-sample-rate");
    PacketHeader empty_packet = header;
    empty_packet.frames = 0;
    expect_rejected(empty_packet, {}, "empty-frame");
    expect_rejected(header, std::vector<float>(16), "payload-size");

    const auto before_ok = core.status();
    check(core.process(header, input) == input, "invalid: a valid packet still passes afterwards");
    const auto after_ok = core.status();
    check(after_ok.rejected == before_ok.rejected && after_ok.packets == before_ok.packets + 1,
          "invalid: successful packets do not disturb the reject counter");
}

conversation::Evidence dialogue_cue() {
    conversation::Evidence cue;
    cue.speech_active = true;
    cue.voice_likelihood = 0.9f;
    cue.conversation_cue = 0.9f;
    cue.level_dbfs = -40.0f;
    return cue;
}

void test_evidence_staleness() {
    const auto cpu_port = reserve_port();
    const auto npu_port = reserve_port();
    PacketHeader header;
    header.frames = 32;
    const auto input = ramp_samples(64);

    // (c) A fresh feed must never be released prematurely, even right after engaging.
    {
        AudioCore core(4096, WorkerPorts{cpu_port, npu_port}, 1000);
        check(core.status().evidence_ttl_ms == 1000,
              "ttl: the constructor TTL is reported in status");
        for (int i = 0; i < 12; ++i) core.ingest_evidence(dialogue_cue(), 100.0f);
        const auto engaged = core.status();
        check(engaged.policy_state == conversation::State::engaged &&
                  engaged.media_percent < 65.0f,
              "ttl: cue frames engage and duck the media");
        const auto ducked = core.process(header, input);
        check(ducked != input, "ttl: engaged playback is ducked while the feed is fresh");
        const auto after = core.status();
        check(!after.evidence_stale && after.evidence_stale_events == 0,
              "ttl: fresh evidence is not released early");
        check(after.media_percent < 65.0f, "ttl: the duck target is preserved while fresh");
        check(after.fallbacks == 0, "ttl: ducking is not counted as a worker fallback");
    }

    // (a) Engage, then let the out-of-process feed go quiet: process() must fail open.
    {
        AudioCore core(4096, WorkerPorts{reserve_port(), reserve_port()}, 50);
        for (int i = 0; i < 12; ++i) core.ingest_evidence(dialogue_cue(), 100.0f);
        const auto ducked = core.process(header, input);
        check(ducked != input, "ttl: playback is ducked before the feed goes quiet");
        bool stale_seen = false;
        for (int i = 0; i < 200 && !stale_seen; ++i) {
            (void)core.process(header, input, 100);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            stale_seen = core.status().evidence_stale;
        }
        check(stale_seen, "ttl: a quiet evidence feed is detected and released (fail-open)");
        const auto released = core.status();
        check(released.policy_state == conversation::State::idle &&
                  released.media_percent == 100.0f,
              "ttl: staleness resets the policy to idle and 100% media");
        check(released.evidence_stale && released.evidence_stale_events == 1,
              "ttl: the stale event is counted exactly once");
        check(released.fallbacks == 0 && released.rejected == 0,
              "ttl: staleness is not conflated with worker fallbacks or rejects");
        const auto exact = core.process(header, input);
        check(exact == input, "ttl: passthrough is bit-exact once the fail-safe ramps back");
        check(core.status().evidence_stale,
              "ttl: the stale flag stays up until a fresh evidence frame arrives");

        // (b) Fresh evidence after the release must re-engage from scratch.
        for (int i = 0; i < 4; ++i) core.ingest_evidence(dialogue_cue(), 100.0f);
        const auto predicted = core.status();
        check(predicted.policy_state == conversation::State::anticipating,
              "ttl: post-release evidence starts a new prediction");
        check(!predicted.evidence_stale, "ttl: fresh evidence clears the stale flag");
        for (int i = 0; i < 8; ++i) core.ingest_evidence(dialogue_cue(), 100.0f);
        const auto reengaged = core.status();
        check(reengaged.policy_state == conversation::State::engaged,
              "ttl: post-release evidence re-engages the policy");
        check(reengaged.media_percent < 65.0f, "ttl: re-engagement restores the duck target");
        const auto ducked_again = core.process(header, input);
        check(ducked_again != input, "ttl: re-engaged playback is ducked again");
        check(core.status().evidence_stale_events == 1,
              "ttl: recovery does not fabricate stale events");
    }

    // The env override is honoured when no constructor TTL is given (0 = auto).
    {
        ::setenv("KHADAS_EVIDENCE_TTL_MS", "123", 1);
        AudioCore from_env;
        check(from_env.status().evidence_ttl_ms == 123,
              "ttl: KHADAS_EVIDENCE_TTL_MS configures the fail-safe");
        ::unsetenv("KHADAS_EVIDENCE_TTL_MS");
        AudioCore defaults;
        check(defaults.status().evidence_ttl_ms == 2000,
              "ttl: the default fail-safe TTL is 2000 ms");
    }
}

void test_loopback_servers() {
    const std::uint16_t audio_port = reserve_port();
    const std::uint16_t http_port = reserve_port();
    AudioCore core(4096, WorkerPorts{reserve_port(), reserve_port()});
    std::atomic<bool> stop{false};
    std::thread http_thread([&] { (void)run_http_server(core, http_port, &stop); });
    std::thread audio_thread([&] { (void)run_audio_server(core, audio_port, &stop); });

    bool http_ready = false;
    for (int i = 0; i < 200 && !http_ready; ++i) {
        const auto reply = http_request(http_port, "GET", "/v1/status", "");
        if (reply.code == 200) {
            http_ready = true;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    check(http_ready, "http: loopback server answers GET /v1/status");
    check(wait_listener(audio_port, 2000), "audio: loopback server accepts connections");

    // --- status JSON: backward compatible fields + new policy fields.
    {
        const auto reply = http_request(http_port, "GET", "/v1/status", "");
        std::string field;
        double number = -1.0;
        bool flag = true;
        check(reply.code == 200 && json_number(reply.body, "schema", number) && number == 1.0,
              "status: schema 1 is preserved");
        check(json_string(reply.body, "requested_mode", field) && field == "passthrough",
              "status: requested_mode is preserved");
        check(json_string(reply.body, "actual_mode", field) && field == "passthrough",
              "status: actual_mode is preserved");
        check(json_string(reply.body, "policy_state", field) && field == "IDLE",
              "status: policy_state reports IDLE");
        check(json_number(reply.body, "media_percent", number) && number == 100.0,
              "status: media_percent starts at 100");
        check(json_bool(reply.body, "ambient_intent", flag) && !flag,
              "status: ambient_intent starts closed");
        check(json_number(reply.body, "last_evidence_ms", number) && number == -1.0,
              "status: last_evidence_ms is -1 before any evidence");
        check(json_bool(reply.body, "evidence_stale", flag) && !flag,
              "status: evidence_stale starts false");
        check(json_number(reply.body, "evidence_ttl_ms", number) && number == 2000.0,
              "status: evidence_ttl_ms exposes the fail-safe TTL");
        check(json_number(reply.body, "evidence_stale_events", number) && number == 0.0,
              "status: evidence_stale_events starts at 0");
    }

    // --- mode endpoint.
    {
        auto reply = http_request(http_port, "POST", "/v1/mode", "cpu");
        std::string mode;
        check(reply.code == 200 && json_string(reply.body, "requested_mode", mode) &&
                  mode == "cpu",
              "mode: POST cpu is accepted");
        reply = http_request(http_port, "POST", "/v1/mode", "banana");
        check(reply.code == 400, "mode: unknown mode is rejected with 400");
        reply = http_request(http_port, "POST", "/v1/mode", "passthrough");
        check(reply.code == 200 && json_string(reply.body, "requested_mode", mode) &&
                  mode == "passthrough",
              "mode: passthrough is restored");
    }

    // --- evidence endpoint validation and half-duplex state machine over real HTTP.
    const std::string cue_json =
        R"({"speech_active":true,"voice_likelihood":0.9,"conversation_cue":0.9,)"
        R"("own_voice":0.0,"level_dbfs":-40.0,"event_blocked":false,)"
        R"("dialogue_veto":false,"elapsed_ms":100})";
    {
        auto reply = http_request(http_port, "POST", "/v1/evidence", "{not json");
        check(reply.code == 400, "evidence: malformed json is rejected with 400");
        reply = http_request(http_port, "POST", "/v1/evidence", R"({"conversation_cue":"high"})");
        check(reply.code == 400, "evidence: wrong field type is rejected with 400");
    }
    {
        double preview_media = 1000.0;
        bool preview_ok = true;
        for (int i = 0; i < 5; ++i) {
            const auto reply = http_request(http_port, "POST", "/v1/evidence", cue_json);
            double media = 0.0;
            if (reply.code != 200 || !json_number(reply.body, "media_percent", media)) {
                preview_ok = false;
                break;
            }
            preview_media = media;
        }
        check(preview_ok, "evidence: every cue frame is accepted over HTTP");
        check(preview_media < 90.0 && preview_media >= 69.0,
              "evidence: 500 ms of cue previews the media at the 70% floor");
        double engaged_media = 1000.0;
        std::string state;
        bool ambient = false;
        for (int i = 0; i < 7; ++i) {
            const auto reply = http_request(http_port, "POST", "/v1/evidence", cue_json);
            (void)json_number(reply.body, "media_percent", engaged_media);
            (void)json_string(reply.body, "policy_state", state);
            (void)json_bool(reply.body, "ambient_intent", ambient);
        }
        check(state == "ENGAGED", "evidence: 800 ms confirms the conversation over HTTP");
        check(ambient, "evidence: confirmation opens the ambient intent");
        check(engaged_media < 65.0 && engaged_media >= 59.0,
              "evidence: engaged media settles at the 60% floor");
        const auto status = core.status();
        check(status.evidence_frames == 12 && status.last_evidence_ms >= 0,
              "evidence: frame count and monotonic last-evidence timestamp are recorded");
    }

    // --- duck gain applied to the real audio socket.
    PacketHeader header;
    header.frames = 64;
    const std::vector<float> input(128, 0.5f);
    {
        const auto status = core.status();
        const float target = status.media_percent / 100.0f;
        PacketHeader reply_header{};
        std::vector<float> ducked;
        check(audio_round_trip(audio_port, header, input, reply_header, ducked),
              "duck: audio round trip succeeds while engaged");
        check(reply_header.magic == header.magic && reply_header.frames == header.frames,
              "duck: reply header echoes the request");
        bool bounded = ducked.size() == input.size();
        for (std::size_t i = 0; i < ducked.size() && bounded; ++i) {
            bounded = std::abs(ducked[i]) <= std::abs(input[i]) + 1e-6f;
        }
        check(bounded, "duck: media gain never exceeds unity");
        check(std::abs(ducked.back() - input.back() * target) < 1e-4f,
              "duck: the block ramp lands on the policy media target");
        PacketHeader reply_header2{};
        std::vector<float> ducked2;
        check(audio_round_trip(audio_port, header, input, reply_header2, ducked2),
              "duck: a second round trip succeeds");
        bool uniform = ducked2.size() == input.size();
        for (std::size_t i = 0; i < ducked2.size() && uniform; ++i) {
            uniform = std::abs(ducked2[i] - input[i] * target) < 1e-6f;
        }
        check(uniform, "duck: steady-state duck is a uniform per-sample gain (no zipper)");
    }

    // --- recovery: silence brings the policy back to idle and passthrough becomes bit-exact.
    {
        const std::string silence_json =
            R"({"speech_active":false,"voice_likelihood":0.0,"conversation_cue":0.0,)"
            R"("own_voice":0.0,"level_dbfs":-65.0,"event_blocked":false,)"
            R"("dialogue_veto":false,"elapsed_ms":100})";
        for (int i = 0; i < 60; ++i) {
            (void)http_request(http_port, "POST", "/v1/evidence", silence_json);
        }
        const auto status = core.status();
        check(status.policy_state == conversation::State::idle && status.media_percent >= 99.0f,
              "recovery: silence returns the policy to idle at ~100% media");
        PacketHeader reply_header{};
        std::vector<float> restore_block;
        check(audio_round_trip(audio_port, header, input, reply_header, restore_block),
              "recovery: first packet after the conversation succeeds");
        PacketHeader reply_header2{};
        std::vector<float> exact_block;
        check(audio_round_trip(audio_port, header, input, reply_header2, exact_block),
              "recovery: second packet after the conversation succeeds");
        check(exact_block == input, "recovery: passthrough is bit-exact again once idle");
    }

    // --- invalid packet over the wire: no reply, counted, server keeps serving.
    {
        PacketHeader bad = header;
        bad.magic = 0;
        PacketHeader reply_header{};
        std::vector<float> unused;
        check(!audio_round_trip(audio_port, bad, input, reply_header, unused),
              "wire: an invalid packet is rejected without a reply");
        check(core.status().rejected == 1, "wire: the invalid packet is counted exactly once");
        PacketHeader reply_header2{};
        std::vector<float> after_reject;
        check(audio_round_trip(audio_port, header, input, reply_header2, after_reject) &&
                  after_reject == input,
              "wire: the audio server still serves after a rejected packet");
    }

    // --- mid-packet disconnects must not kill the accept loop, and a connection reset must
    //     not reset any core state.
    const auto state_before_disconnects = core.status();
    {
        const int first = connect_port(audio_port);
        check(first >= 0, "robust: audio server accepts a health-check connection");
        if (first >= 0) {
            const char partial[5]{'K', 'O', 'W', 'S', '\0'};
            (void)write_exact(first, partial, sizeof(partial));
            ::close(first);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const int second = connect_port(audio_port);
        if (second >= 0) {
            (void)write_exact(second, &header, sizeof(header));
            const std::vector<float> half(input.begin(), input.begin() + input.size() / 2);
            (void)write_exact(second, half.data(), half.size() * sizeof(float));
            ::close(second);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        PacketHeader reply_header{};
        std::vector<float> after_disconnects;
        check(audio_round_trip(audio_port, header, input, reply_header, after_disconnects) &&
                  after_disconnects == input,
              "robust: mid-packet disconnects do not kill the audio server");
        const auto state_after_disconnects = core.status();
        check(state_after_disconnects.requested == state_before_disconnects.requested &&
                  state_after_disconnects.actual == state_before_disconnects.actual &&
                  state_after_disconnects.fallbacks == state_before_disconnects.fallbacks &&
                  state_after_disconnects.rejected == state_before_disconnects.rejected &&
                  state_after_disconnects.evidence_frames ==
                      state_before_disconnects.evidence_frames,
              "robust: connection resets do not reset core state");
    }

    // --- aborted HTTP connection and concurrent status requests.
    {
        const int fd = connect_port(http_port);
        if (fd >= 0) ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto reply = http_request(http_port, "GET", "/v1/status", "");
        check(reply.code == 200, "robust: HTTP server survives an aborted connection");
    }
    {
        std::atomic<int> successes{0};
        std::vector<std::thread> clients;
        for (int t = 0; t < 8; ++t) {
            clients.emplace_back([&] {
                for (int i = 0; i < 10; ++i) {
                    const auto reply = http_request(http_port, "GET", "/v1/status", "");
                    if (reply.code == 200 && reply.body.find("\"schema\":1") != std::string::npos) {
                        ++successes;
                    }
                }
            });
        }
        for (auto& client : clients) client.join();
        check(successes.load() == 80,
              "robust: 80 concurrent status requests all succeed (no deadlock)");
    }

    // --- end to end: missing NPU worker visible through both transports.
    {
        const auto mode_reply = http_request(http_port, "POST", "/v1/mode", "npu");
        check(mode_reply.code == 200, "e2e: POST /v1/mode npu is accepted");
        PacketHeader reply_header{};
        std::vector<float> fallback;
        check(audio_round_trip(audio_port, header, input, reply_header, fallback) &&
                  fallback == input,
              "e2e: a missing NPU worker falls back bit-exactly through the socket");
        const auto status_reply = http_request(http_port, "GET", "/v1/status", "");
        std::string actual;
        check(json_string(status_reply.body, "actual_mode", actual) && actual == "passthrough",
              "e2e: status JSON reports the fallback actual mode");
        check(core.status().fallback_reason == "worker_unavailable",
              "e2e: fallback_reason is worker_unavailable");
    }

    stop.store(true);
    http_thread.join();
    audio_thread.join();
}

} // namespace

int main() {
    ::signal(SIGPIPE, SIG_IGN);
    test_ring_buffer();
    test_ring_overflow_via_core();
    test_worker_absent();
    test_worker_timeout();
    test_worker_non_finite();
    test_worker_protocol();
    test_worker_short_read();
    test_mode_churn_and_recovery();
    test_invalid_packets();
    test_evidence_staleness();
    test_loopback_servers();
    std::cout << "\nfailure_matrix: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
