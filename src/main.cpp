#include "khadas_ows/audio_core.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true); }

std::uint16_t parse_port(const char* text, std::uint16_t fallback, const char* source) {
    if (text == nullptr || *text == '\0') return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 65535) {
        std::cerr << "invalid port for " << source << ": '" << text << "', using " << fallback
                  << '\n';
        return fallback;
    }
    return static_cast<std::uint16_t>(value);
}

std::uint16_t env_port(const char* name, std::uint16_t fallback) {
    return parse_port(std::getenv(name), fallback, name);
}

} // namespace

int main(int argc, char** argv) {
    // Ports: defaults are the documented 9100 (audio TCP) / 8787 (HTTP status). Both can be
    // overridden by env (used by tests / the Edge2 harness on a busy host) or by argv, where
    // argv wins over env so an explicit command line stays predictable.
    std::uint16_t audio_port = env_port("KHADAS_AUDIO_PORT", 9100);
    std::uint16_t http_port = env_port("KHADAS_HTTP_PORT", 8787);
    if (argc > 1) audio_port = parse_port(argv[1], audio_port, "argv[1]");
    if (argc > 2) http_port = parse_port(argv[2], http_port, "argv[2]");

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    khadas_ows::AudioCore core; // worker ports come from KHADAS_CPU_PORT / KHADAS_NPU_PORT
    if (const char* hash = std::getenv("KHADAS_MODEL_HASH")) core.set_model_hash(hash);
    const auto worker_ports = core.worker_ports();

    std::thread http([&] {
        const int result = khadas_ows::run_http_server(core, http_port, &g_stop);
        if (result != 0) {
            std::cerr << "http server failed to bind 0.0.0.0:" << http_port << '\n';
        }
    });
    std::cout << "audio tcp://0.0.0.0:" << audio_port << " status http://0.0.0.0:" << http_port
              << " workers cpu=" << worker_ports.cpu << " npu=" << worker_ports.npu << '\n'
              << std::flush;

    const int result = khadas_ows::run_audio_server(core, audio_port, &g_stop);
    if (result != 0) {
        std::cerr << "audio server failed to bind 0.0.0.0:" << audio_port << '\n';
    }
    g_stop.store(true);
    http.join();
    return result;
}
