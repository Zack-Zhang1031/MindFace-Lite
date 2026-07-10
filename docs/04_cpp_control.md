# C++ Realtime Control

## What To Build

The C++ side demonstrates realtime control building blocks:

- CMake project.
- Producer-consumer queue.
- UDP sender for robot or actuator control.
- Serial-style output.
- ONNXRuntime C++ demo skeleton.

## Build

```powershell
cmake -S cpp -B build/cpp
cmake --build build/cpp --config Release
```

## Run

```powershell
build\cpp\queue_demo.exe
build\cpp\udp_sender.exe outputs\logs\pytorch_mlp_params.csv 127.0.0.1 9000 25
build\cpp\serial_sender.exe outputs\logs\serial_output.txt outputs\logs\pytorch_mlp_params.csv 25
```

## ONNXRuntime C++ Demo

The ONNXRuntime demo is optional because it requires the ONNXRuntime C++ SDK:

```powershell
cmake -S cpp -B build/cpp-ort -DBUILD_ONNXRUNTIME_DEMO=ON -DONNXRUNTIME_DIR=C:\path\to\onnxruntime
```

## Interview Explanation

The C++ side shows I understand the deployment and control boundary. Python is useful for training and prototyping, while C++ is common for realtime control loops, robot interfaces, and embedded deployment. UDP is useful for networked control; serial is common for microcontroller actuator boards.
