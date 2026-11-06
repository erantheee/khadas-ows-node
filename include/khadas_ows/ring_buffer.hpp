#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

namespace khadas_ows {
template <typename T> class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity) : data_(capacity) {}
    std::size_t size() const { return size_; }
    std::size_t capacity() const { return data_.size(); }
    std::size_t push(const T* input, std::size_t count) {
        const auto accepted = std::min(count, capacity() - size_);
        for (std::size_t i = 0; i < accepted; ++i) {
            data_[(write_ + i) % capacity()] = input[i];
        }
        write_ = (write_ + accepted) % capacity();
        size_ += accepted;
        return accepted;
    }
    std::size_t pop(T* output, std::size_t count) {
        const auto available = std::min(count, size_);
        for (std::size_t i = 0; i < available; ++i) {
            output[i] = data_[(read_ + i) % capacity()];
        }
        read_ = (read_ + available) % capacity();
        size_ -= available;
        return available;
    }
    void clear() { read_ = write_ = size_ = 0; }
private:
    std::vector<T> data_;
    std::size_t read_ = 0, write_ = 0, size_ = 0;
};
}

