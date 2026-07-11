#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

struct MouthParams {
    float open;
    float width;
    float round;
};

template <typename T>
class SafeQueue {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(value);
        }
        cv_.notify_one();
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty() || stopped_; });
        if (queue_.empty()) return false;
        value = queue_.front();
        queue_.pop();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
};

int main() {
    SafeQueue<MouthParams> q;
    std::thread producer([&] {
        for (int i = 0; i < 30; ++i) {
            float open = static_cast<float>(i % 20) / 20.0f;
            q.push({open, 0.5f + 0.2f * open, 0.2f + 0.1f * open});
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        q.stop();
    });

    std::thread consumer([&] {
        MouthParams p{};
        while (q.pop(p)) {
            std::cout << "control output: open=" << p.open
                      << " width=" << p.width
                      << " round=" << p.round << std::endl;
        }
    });

    producer.join();
    consumer.join();
    return 0;
}
