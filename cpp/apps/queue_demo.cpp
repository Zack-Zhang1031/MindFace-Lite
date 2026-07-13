#include <chrono>
#include <iostream>
#include <thread>

#include "mindface/bounded_queue.hpp"
#include "mindface/mouth_params.hpp"

int main() {
    mindface::BoundedQueue<mindface::MouthParams> queue(8, mindface::OverflowPolicy::DropOldest);
    std::thread producer([&] {
        for (int index = 0; index < 30; ++index) {
            const float mouth_open = static_cast<float>(index % 20) / 20.0F;
            queue.push({mouth_open, 0.5F + 0.2F * mouth_open, 0.2F + 0.1F * mouth_open});
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        queue.close();
    });

    std::thread consumer([&] {
        mindface::MouthParams params{};
        while (queue.pop(params)) {
            std::cout << "control output: open=" << params.mouth_open << " width=" << params.mouth_width
                      << " round=" << params.lip_round << '\n';
        }
    });

    producer.join();
    consumer.join();
    std::cout << "accepted=" << queue.accepted() << " dropped=" << queue.dropped() << '\n';
    return 0;
}
