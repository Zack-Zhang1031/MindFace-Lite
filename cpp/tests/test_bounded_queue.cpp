#include <chrono>
#include <iostream>
#include <thread>

#include "mindface/bounded_queue.hpp"

int main() {
    using mindface::BoundedQueue;
    using mindface::OverflowPolicy;

    BoundedQueue<int> queue(2, OverflowPolicy::DropOldest);
    if (!queue.push(1) || !queue.push(2) || !queue.push(3)) return 1;
    if (queue.dropped() != 1) return 2;
    int value = 0;
    if (!queue.pop(value) || value != 2) return 3;
    if (!queue.pop(value) || value != 3) return 4;

    BoundedQueue<int> waiting_queue(1);
    bool released = false;
    std::thread consumer([&] {
        int item = 0;
        released = !waiting_queue.pop(item);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    waiting_queue.close();
    consumer.join();
    if (!released) return 5;

    std::cout << "bounded queue tests passed\n";
    return 0;
}
