# C++ Realtime Control

## What To Build

The C++ side separates a reusable `mindface_runtime` library from executable apps:

```text
cpp/include/mindface/   public queue and mouth parameter contracts
cpp/src/                runtime library implementation
cpp/apps/               queue, UDP, serial and optional ONNXRuntime demos
cpp/tests/              CTest executables
cpp/CMakePresets.json   Windows, Linux and ARM64 presets
```

## Windows Build And Test

```powershell
Push-Location cpp
cmake --preset windows-release
cmake --build --preset windows-release
ctest --preset windows-release
Pop-Location
```

Manual equivalent:

```powershell
cmake -S cpp -B build/cpp -DBUILD_TESTING=ON
cmake --build build/cpp --config Release
ctest --test-dir build/cpp -C Release --output-on-failure
```

## Run

```powershell
build\cpp-windows\Release\queue_demo.exe
build\cpp-windows\Release\udp_sender.exe outputs\logs\pytorch_mlp_params.csv 127.0.0.1 9000 25
build\cpp-windows\Release\serial_sender.exe outputs\logs\serial_output.txt outputs\logs\pytorch_mlp_params.csv 25
```

## ONNXRuntime C++ Demo

```powershell
cmake -S cpp -B build/cpp-ort -DBUILD_ONNXRUNTIME_DEMO=ON -DONNXRUNTIME_DIR=C:\path\to\onnxruntime
cmake --build build/cpp-ort --config Release
```

## Queue Contract

`BoundedQueue<T>` supports bounded capacity, FIFO pop, blocking producers, drop-oldest/drop-newest overflow and close wakeup. The CTest executable verifies overflow, order and blocked-consumer release.

## Interview Explanation

Python is used for training and fast debugging. The C++ runtime owns bounded realtime queues and actuator-facing applications. Separating a library from apps allows UDP, serial and ONNXRuntime control programs to share the same concurrency and mouth-parameter contracts.
