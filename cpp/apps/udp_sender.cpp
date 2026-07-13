#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mindface/mouth_params.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

std::vector<mindface::MouthParams> load_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open csv: " + path);
    std::string line;
    std::getline(in, line); // header
    std::vector<mindface::MouthParams> rows;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> cols;
        while (std::getline(ss, item, ',')) cols.push_back(item);
        if (cols.size() >= 5) {
            rows.push_back({std::stof(cols[2]), std::stof(cols[3]), std::stof(cols[4])});
        } else if (cols.size() >= 4) {
            rows.push_back({std::stof(cols[1]), std::stof(cols[2]), std::stof(cols[3])});
        }
    }
    return rows;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: udp_sender <params.csv> <ip> <port> <fps>\n";
        return 1;
    }
    std::string csv_path = argv[1];
    std::string ip = argv[2];
    int port = std::stoi(argv[3]);
    int fps = std::stoi(argv[4]);
    auto rows = load_csv(csv_path);

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip.c_str());
    auto interval = std::chrono::milliseconds(1000 / std::max(1, fps));
    for (const auto& p : rows) {
        std::ostringstream msg;
        msg << "mouth_open=" << p.mouth_open << ",mouth_width=" << p.mouth_width << ",lip_round=" << p.lip_round;
        std::string s = msg.str();
        sendto(sock, s.c_str(), static_cast<int>(s.size()), 0, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        std::cout << "sent: " << s << std::endl;
        std::this_thread::sleep_for(interval);
    }

#ifdef _WIN32
    closesocket(sock);
    WSACleanup();
#else
    close(sock);
#endif
    return 0;
}
