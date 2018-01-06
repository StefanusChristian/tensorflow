/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
}
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

// Returns true if the model referred to by `handle` is valid
bool TfLiteValid(void* model_handle);

// Returns the error string for the model
const char* TfLiteErrorString(void* model_handle);

// Loads the model from tflite file `filename` and sets it to execute with
// `threads` in parallel. Returns a handle ptr that must be freed with
// `TfLiteFree`.
void* TfLiteLoadImageClassifyModel(const char* filename, int threads);

// Classifies an image of `height`, `width`, `chans` input in `src_buf`
// buffer which  is `src_buffer_size`.  Populates `out_probs_buf` with
// output probabilities for each label.
//
// Make sure you input the right type i.e.
// If your model is float `src_buf` should be `height*width*chans*sizeof(float)`
// bytes.
bool TfLiteRunImageClassify(void* model_handle,
                            int height, int width, int chans,
                            unsigned char* src_buf, int src_buffer_size,
                            unsigned char* out_probs_buf, int out_probs_size);

// Returns true if the model is uint8 quantized inputs/outputs.
bool TfLiteIsUint8Quant(void* handle);


// Frees the model referred by the handle.
void TfLiteFree(void* model_handle);
