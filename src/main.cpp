#include "khadas_ows/audio_core.hpp"
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>
int main(int argc, char** argv) {
    std::uint16_t audio_port = 9100, http_port = 8787;
    if (argc > 1) audio_port = static_cast<std::uint16_t>(std::stoi(argv[1]));
    khadas_ows::AudioCore core;
    if (const char* hash = std::getenv("KHADAS_MODEL_HASH")) core.set_model_hash(hash);
    std::thread http([&] { khadas_ows::run_http_server(core, http_port); });
    std::cout << "audio tcp://0.0.0.0:" << audio_port << " status http://0.0.0.0:" << http_port << "\n";
    const int result = khadas_ows::run_audio_server(core, audio_port);
    http.detach(); return result;
}

