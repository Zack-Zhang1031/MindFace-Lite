#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: serial_sender <serial_device_or_output_file> <params.csv> [fps]\n";
        return 1;
    }
    std::string device = argv[1];
    std::string csv_path = argv[2];
    int fps = argc >= 4 ? std::stoi(argv[3]) : 25;
    std::ifstream csv(csv_path);
    std::ofstream serial(device, std::ios::out | std::ios::app);
    if (!csv) { std::cerr << "Failed to open csv\n"; return 2; }
    if (!serial) { std::cerr << "Failed to open serial/output file\n"; return 3; }
    std::string line;
    std::getline(csv, line);
    auto interval = std::chrono::milliseconds(1000 / std::max(1, fps));
    while (std::getline(csv, line)) {
        std::stringstream ss(line);
        std::string col;
        std::vector<std::string> cols;
        while (std::getline(ss, col, ',')) cols.push_back(col);
        if (cols.size() >= 5) {
            std::string cmd = "MOUTH " + cols[2] + " " + cols[3] + " " + cols[4] + "\n";
            serial << cmd << std::flush;
            std::cout << "serial: " << cmd;
        } else if (cols.size() >= 4) {
            std::string cmd = "MOUTH " + cols[1] + " " + cols[2] + " " + cols[3] + "\n";
            serial << cmd << std::flush;
            std::cout << "serial: " << cmd;
        }
        std::this_thread::sleep_for(interval);
    }
    return 0;
}
