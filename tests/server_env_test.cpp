// Proves the real khadas-ows-node binary honours KHADAS_AUDIO_PORT / KHADAS_HTTP_PORT and that
// its server loop is robust against aborted clients. The binary path is injected by CMake via
// the KHADAS_NODE_BINARY compile definition.
#include <arpa/inet.h>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef KHADAS_NODE_BINARY
#error "KHADAS_NODE_BINARY must be defined by CMake"
#endif

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

std::uint16_t reserve_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 0;
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

std::string to_string(std::uint16_t value) { return std::to_string(static_cast<int>(value)); }

} // namespace

int main() {
    ::signal(SIGPIPE, SIG_IGN);
    if (::access(KHADAS_NODE_BINARY, X_OK) != 0) {
        std::cerr << "server_env_test: binary is not executable: " << KHADAS_NODE_BINARY << '\n';
        return 1;
    }

    // Pick free ports and hand them to the real binary through the documented env vars.
    const std::uint16_t audio_port = reserve_port();
    const std::uint16_t http_port = reserve_port();
    const std::uint16_t cpu_port = reserve_port();
    const std::uint16_t npu_port = reserve_port();
    check(audio_port != 0 && http_port != 0 && cpu_port != 0 && npu_port != 0,
          "env: reserved four distinct loopback ports");
    ::setenv("KHADAS_AUDIO_PORT", to_string(audio_port).c_str(), 1);
    ::setenv("KHADAS_HTTP_PORT", to_string(http_port).c_str(), 1);
    // Point the worker ports at free (closed) ports so a live production worker cannot answer.
    ::setenv("KHADAS_CPU_PORT", to_string(cpu_port).c_str(), 1);
    ::setenv("KHADAS_NPU_PORT", to_string(npu_port).c_str(), 1);
    // Short evidence TTL so the spawned binary's fail-open release is observable in-test.
    ::setenv("KHADAS_EVIDENCE_TTL_MS", "300", 1);

    const pid_t child = ::fork();
    if (child == 0) {
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            (void)::dup2(devnull, STDOUT_FILENO);
            (void)::dup2(devnull, STDERR_FILENO);
        }
        ::execl(KHADAS_NODE_BINARY, KHADAS_NODE_BINARY, static_cast<char*>(nullptr));
        ::_exit(127);
    }
    check(child > 0, "env: server process spawned");

    bool http_ready = false;
    for (int i = 0; i < 300 && child > 0 && !http_ready; ++i) {
        const auto reply = http_request(http_port, "GET", "/v1/status", "");
        if (reply.code == 200) {
            http_ready = true;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    check(http_ready, "env: KHADAS_HTTP_PORT is honoured by the binary");

    if (http_ready) {
        const auto reply = http_request(http_port, "GET", "/v1/status", "");
        std::string field;
        check(json_string(reply.body, "requested_mode", field) && field == "passthrough",
              "env: status is served on the env-configured HTTP port");

        const auto mode_reply = http_request(http_port, "POST", "/v1/mode", "cpu");
        check(mode_reply.code == 200 && json_string(mode_reply.body, "requested_mode", field) &&
                  field == "cpu",
              "env: POST /v1/mode works on the env-configured port");

        double media = 0.0;
        const std::string cue_json =
            R"({"speech_active":true,"voice_likelihood":0.9,"conversation_cue":0.9,)"
            R"("own_voice":0.0,"level_dbfs":-40.0,"event_blocked":false,)"
            R"("dialogue_veto":false,"elapsed_ms":100})";
        for (int i = 0; i < 10; ++i) {
            const auto evidence_reply = http_request(http_port, "POST", "/v1/evidence", cue_json);
            (void)json_number(evidence_reply.body, "media_percent", media);
        }
        check(media > 59.0 && media < 80.0,
              "env: POST /v1/evidence drives the policy through the binary");

        // Audio TCP on the env-configured port. The CPU worker port was redirected to a closed
        // port, so mode requests fall back and the policy must also be visible in the PCM path.
        const auto mode_back = http_request(http_port, "POST", "/v1/mode", "passthrough");
        check(mode_back.code == 200, "env: switched back to passthrough");
        if (mode_back.code == 200) {
            constexpr std::uint32_t kFrames = 32;
            constexpr std::uint32_t kMagic = 0x4b4f5753;
            // Wire layout (little endian): magic u32, schema u32, sequence u64, sample_rate u32,
            // channels u16, reserved u16, frames u32 -- 28 bytes total.
            std::uint32_t header[7]{kMagic, 1, 1, 0, 44100, 2, kFrames};
            std::vector<float> input(kFrames * 2);
            for (std::size_t i = 0; i < input.size(); ++i) {
                input[i] = std::sin(static_cast<float>(i) * 0.1f) * 0.3f;
            }
            auto round_trip = [&](std::vector<float>& output) {
                const int fd = connect_port(audio_port);
                if (fd < 0) return false;
                std::uint32_t reply_header[7]{};
                output.assign(input.size(), 0.0f);
                const bool ok =
                    write_exact(fd, header, sizeof(header)) &&
                    write_exact(fd, input.data(), input.size() * sizeof(float)) &&
                    read_exact(fd, reply_header, sizeof(reply_header)) &&
                    read_exact(fd, output.data(), output.size() * sizeof(float));
                ::close(fd);
                return ok && reply_header[0] == kMagic && reply_header[6] == kFrames;
            };
            const int fd = connect_port(audio_port);
            check(fd >= 0, "env: KHADAS_AUDIO_PORT accepts a TCP connection");
            if (fd >= 0) ::close(fd);
            // Keep the evidence timestamp fresh so the binary's stale-evidence TTL cannot fire
            // between the policy assertion above and this PCM assertion on a slow CI machine.
            (void)http_request(http_port, "POST", "/v1/evidence", cue_json);
            std::vector<float> ducked;
            check(round_trip(ducked), "env: audio packet is echoed on the env-configured audio port");
            bool bounded = ducked.size() == input.size();
            for (std::size_t i = 0; i < ducked.size() && bounded; ++i) {
                bounded = std::abs(ducked[i]) <= std::abs(input[i]) + 1e-6f;
            }
            check(bounded && std::abs(ducked.back() - input.back()) > 1e-4f,
                  "env: ingested evidence ducks the PCM path of the real binary");

            // Fail-open end-to-end: keep the evidence feed quiet and the real binary must
            // release the duck after KHADAS_EVIDENCE_TTL_MS instead of holding it forever.
            bool stale = false;
            std::vector<float> probe;
            for (int i = 0; i < 12 && !stale; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                (void)round_trip(probe); // drives AudioCore::process(), where the fail-safe lives
                const auto status_reply = http_request(http_port, "GET", "/v1/status", "");
                (void)json_bool(status_reply.body, "evidence_stale", stale);
            }
            check(stale, "env: a quiet evidence feed trips the stale fail-safe in the real binary");
            const auto stale_reply = http_request(http_port, "GET", "/v1/status", "");
            std::string policy_state;
            double number = 0.0;
            bool flag = false;
            check(json_string(stale_reply.body, "policy_state", policy_state) &&
                      policy_state == "IDLE",
                  "env: the stale fail-safe releases the real binary policy to idle");
            check(json_number(stale_reply.body, "media_percent", number) && number == 100.0,
                  "env: the stale fail-safe restores 100% media in the real binary");
            check(json_bool(stale_reply.body, "evidence_stale_events", flag) == false &&
                      json_number(stale_reply.body, "evidence_stale_events", number) &&
                      number >= 1.0,
                  "env: the stale event is counted in the real binary status");
            std::vector<float> restore;
            (void)round_trip(restore); // releases the duck ramp
            std::vector<float> exact;
            check(round_trip(exact), "env: audio still flows after the stale fail-safe");
            check(exact == input, "env: passthrough output is bit-exact after the stale fail-safe");

            // Aborted client (partial header) must not kill the accept loop.
            const int aborted = connect_port(audio_port);
            if (aborted >= 0) {
                const char partial[3]{'K', 'O', 'W'};
                (void)write_exact(aborted, partial, sizeof(partial));
                ::close(aborted);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const int second = connect_port(audio_port);
            if (second >= 0) ::close(second);
            const auto status_after = http_request(http_port, "GET", "/v1/status", "");
            check(status_after.code == 200, "env: the binary survives an aborted audio client");
        }
    }

    // The binary must shut down cleanly on SIGTERM (server loops poll the stop flag).
    if (child > 0) {
        ::kill(child, SIGTERM);
        bool reaped = false;
        int status = 0;
        for (int i = 0; i < 200; ++i) {
            const pid_t result = ::waitpid(child, &status, WNOHANG);
            if (result == child) {
                reaped = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!reaped) {
            ::kill(child, SIGKILL);
            (void)::waitpid(child, &status, 0);
        }
        check(reaped, "env: the binary exits on SIGTERM instead of hanging");
        check(reaped && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "env: the binary exits with code 0");
    }

    std::cout << "\nserver_env: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
