#include "khadas_ows/audio_core.hpp"

#include <arpa/inet.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace khadas_ows;

namespace {
void check(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

// The production worker ports may legitimately have a live CPU/NPU worker attached on a
// developer machine, so this test isolates itself on freshly-reserved loopback ports.
std::uint16_t reserve_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
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
} // namespace

int main() {
    RingBuffer<int> ring(4);
    int in[]{1, 2, 3, 4, 5};
    int out[4]{};
    check(ring.push(in, 5) == 4, "ring bounds writes");
    check(ring.pop(out, 2) == 2 && out[0] == 1 && out[1] == 2, "ring preserves order");
    check(ring.push(in + 4, 1) == 1, "ring wraps");
    check(ring.pop(out, 4) == 3 && out[2] == 5, "ring wrapped order");

    AudioCore core(32, WorkerPorts{reserve_port(), reserve_port()});
    PacketHeader header;
    header.frames = 8;
    std::vector<float> pcm(16);
    for (std::size_t i = 0; i < pcm.size(); ++i) pcm[i] = static_cast<float>(i) / 16;
    auto pass = core.process(header, pcm);
    check(pass == pcm, "passthrough is bit exact");
    core.request_mode(Mode::npu);
    auto fallback = core.process(header, pcm, 5);
    check(fallback == pcm, "missing NPU worker falls back bit exact");
    auto status = core.status();
    check(status.actual == Mode::passthrough && status.fallbacks == 1, "fallback is visible");
    PacketHeader bad = header;
    bad.channels = 1;
    check(core.process(bad, std::vector<float>(8)).empty(), "invalid packet rejected");
    std::cout << "core_test passed\n";
}
