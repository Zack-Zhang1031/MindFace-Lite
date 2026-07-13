#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: onnxruntime_cpp_demo <mlp_mouth.onnx>\n";
        return 1;
    }
    const char* model_path = argv[1];
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mindface_ort_cpp");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_shape{1, 70};
    std::vector<float> input(70, 0.5f);
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input.data(), input.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {"audio_features"};
    const char* output_names[] = {"mouth_params"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* out = outputs[0].GetTensorMutableData<float>();
    std::cout << "mouth_open=" << out[0] << " mouth_width=" << out[1] << " lip_round=" << out[2] << std::endl;
    return 0;
}
