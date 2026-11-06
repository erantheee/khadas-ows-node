#include "khadas_ows/audio_core.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
using namespace khadas_ows;
void check(bool value, const char* message) { if (!value) { std::cerr << message << '\n'; std::exit(1); } }
int main() {
    RingBuffer<int> ring(4); int in[]{1,2,3,4,5}, out[4]{};
    check(ring.push(in, 5) == 4, "ring bounds writes");
    check(ring.pop(out, 2) == 2 && out[0] == 1 && out[1] == 2, "ring preserves order");
    check(ring.push(in + 4, 1) == 1, "ring wraps");
    check(ring.pop(out, 4) == 3 && out[2] == 5, "ring wrapped order");
    AudioCore core(32); PacketHeader header; header.frames = 8;
    std::vector<float> pcm(16); for (std::size_t i=0;i<pcm.size();++i) pcm[i]=static_cast<float>(i)/16;
    auto pass = core.process(header, pcm); check(pass == pcm, "passthrough is bit exact");
    core.request_mode(Mode::npu); auto fallback = core.process(header, pcm, 5);
    check(fallback == pcm, "missing NPU worker falls back bit exact");
    auto status = core.status(); check(status.actual == Mode::passthrough && status.fallbacks == 1, "fallback is visible");
    PacketHeader bad = header; bad.channels = 1; check(core.process(bad, std::vector<float>(8)).empty(), "invalid packet rejected");
    std::cout << "core_test passed\n";
}

