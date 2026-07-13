#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace mindface {

enum class OverflowPolicy { Block, DropOldest, DropNewest };

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity, OverflowPolicy policy = OverflowPolicy::Block)
        : capacity_(capacity), policy_(policy) {
        if (capacity_ == 0) throw std::invalid_argument("BoundedQueue capacity must be positive");
    }

    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closed_) return false;
        if (queue_.size() >= capacity_) {
            if (policy_ == OverflowPolicy::DropNewest) {
                ++dropped_;
                return false;
            }
            if (policy_ == OverflowPolicy::DropOldest) {
                queue_.pop_front();
                ++dropped_;
            } else {
                not_full_.wait(lock, [&] { return queue_.size() < capacity_ || closed_; });
                if (closed_) return false;
            }
        }
        queue_.push_back(std::move(value));
        ++accepted_;
        not_empty_.notify_one();
        return true;
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    [[nodiscard]] std::size_t accepted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return accepted_;
    }

    [[nodiscard]] std::size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

    [[nodiscard]] std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    const std::size_t capacity_;
    const OverflowPolicy policy_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    bool closed_ = false;
    std::size_t accepted_ = 0;
    std::size_t dropped_ = 0;
};

}  // namespace mindface
