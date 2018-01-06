/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Provides an API for running TF Lite using only C ABI.
//
// Example usage:
// std::vector<float> image_in, probs_out;
// if(!TfLiteLoadImageClassifyModel("foo.tflite", 4)) {
//   fprintf(stderr, TfLiteErrorString());
//   exit(1);
// }
// if(!TfLiteRunImageClassify(
//     width, height, i
//     image_in.data(), image_in.size() * sizeof(float),
//     probs_out.data(), probs_out.size() * sizeof(float))) {
//   fprintf(stderr, TfLiteErrorString());
//   exit(1);
// }

extern "C"{
#include "tflite_c.h"
}
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include <vector>
#include <algorithm>
#include <memory>
using namespace tflite;

namespace{

class SimpleErrorReporter:public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    char buf[1024];
    int ret = vsnprintf(buf, sizeof(buf), format, args);
    error_buffer += buf;
    return ret;
  }

  int Report(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int code = Report(format, args);
    va_end(args);
    return code;
  }

  void Clear() {
    error_buffer.clear();
  }

  const std::string& Str() {
    return error_buffer;
  }

  std::string error_buffer;
};


struct State {
  SimpleErrorReporter errors;
  std::unique_ptr<FlatBufferModel> model;
  std::unique_ptr<Interpreter> interpreter;
  // Remember last size initialized.
  int current_height = -1;
  int current_width = -1;
  int current_chans = -1;
};

// State& state() {
//   static State* state = new State();
//   return *state;
// };

}  // namespace

extern "C" {

const char* TfLiteErrorString(void* state) {
  if(!state) return "<model state has null ptr>";
  return reinterpret_cast<State*>(state)->errors.Str().c_str();
}

bool TfLiteValid(void* raw_state) {
  if(!raw_state) return false;
  State& state = *reinterpret_cast<State*>(raw_state);
  return state.interpreter && state.model;
}

void* TfLiteLoadImageClassifyModel(const char* filename, int threads){
  std::unique_ptr<State> state(new State());

  state->errors.Clear();
  state->model = FlatBufferModel::BuildFromFile(filename, &state->errors);
  if(!state->model) return state.release();
  tflite::ops::builtin::BuiltinOpResolver builtins;
  tflite::InterpreterBuilder(*state->model, builtins)(
      &state->interpreter);
  state->interpreter->SetNumThreads(4);
  if(state->interpreter->inputs().size() != 1) {
    state->errors.ReportError(0, "Expected model to have 1 input tensor.\n");
    state->interpreter.reset();
    return state.release();
  }
  if(state->interpreter->outputs().size() != 1) {
    state->errors.Report(
        0, "Expected 1 output tensor.\n");
    state->interpreter.reset();
    return state.release();
  }

  return state.release();
}

bool TfLiteIsUint8Quant(void* raw_state) {
  if(!raw_state) return false;
  State& state = *reinterpret_cast<State*>(raw_state);
  if(!state.interpreter) return false;
  if(state.interpreter->inputs().size() != 1) {
    state.errors.ReportError(0, "Expected 1 input");
    return false;
  }
  if(state.interpreter->outputs().size() != 1) {
    state.errors.ReportError(0, "Expected 1 output");
    return false;
  }
  int input = state.interpreter->inputs()[0];
  int output = state.interpreter->outputs()[0];

  if(TfLiteTensor* tensor = state.interpreter->tensor(output)){
    if(tensor->type != kTfLiteUInt8) return false;
  } else {
    return false;
  }
  if(TfLiteTensor* tensor = state.interpreter->tensor(input)){
    if(tensor->type != kTfLiteUInt8) return false;
  } else {
    return false;
  }
  return true;

}

bool TfLiteRunImageClassify(void* raw_state, int height, int width, int chans,
                            unsigned char* src_buf, int src_buffer_size,
                            unsigned char* out_probs_buf, int out_probs_size) {
  if(!raw_state) return false;
  State& state = *reinterpret_cast<State*>(raw_state);
  state.errors.Clear();

  // TODO(aselle): This could be optimized to avoid copying buffers.
  // by using the input and output buffers directly from tflite.

  // We must have an interpreter.
  if(!state.interpreter) {
    state.errors.Report(
        "TfLiteRunImageClassify must be called after loading model.\n");
    return false;
  }

  int input = state.interpreter->inputs()[0];
  int output = state.interpreter->outputs()[0];
  if(state.current_height != height || state.current_width != height
      || state.current_chans  != chans){
    // Reallocate the image buffer if input size changes.
    state.current_height = height;
    state.current_width = width;
    state.current_chans = chans;
    state.interpreter->ResizeInputTensor(input, {1, height, width, chans});
    state.interpreter->AllocateTensors();
  }

  // Copy the input image into the buffer.
  TfLiteTensor* tensor = state.interpreter->tensor(input);
  char* dest_buf = state.interpreter->tensor(input)->data.raw;
  if(tensor->bytes != src_buffer_size) {
    state.errors.Report(
        "Provided input buffer was size %d compared to expected "
        "input tensor %d bytes.\n", src_buffer_size, tensor->bytes);
    return false;
  }
  memcpy(dest_buf, src_buf, tensor->bytes);
  state.interpreter->Invoke();

  // Copy the output data to the provided output tensor.
  if(TfLiteTensor* out_tensor = state.interpreter->tensor(output)){
    if(out_tensor->bytes != out_probs_size) {
      state.errors.Report("Provided output buffer was size %d bytes while the output"
              "tensor was size %d bytes.\n",
              out_probs_size, out_tensor->bytes);
      return false;
    }

    char* out_data = out_tensor->data.raw;
    memcpy(out_probs_buf, out_data, out_tensor->bytes);
    return true;
  } else {
    return false;
  }
}

void TfLiteFree(void* state) {
  delete reinterpret_cast<State*>(state);
}



}

