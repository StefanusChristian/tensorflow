#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from cffi import FFI
import sys
from matplotlib import pyplot
import os
import time

# Find the header and dso for the library
SCRIPT_LOCATION = os.path.dirname(sys.argv[0])
if not SCRIPT_LOCATION: SCRIPT_LOCATION = "./"

# Library path is either in dev spot or in same directory as script.
LIB_LOCATION = "tensorflow/contrib/lite/gen/bin"
if not os.path.isdir(LIB_LOCATION):
  LIB_LOCATION = SCRIPT_LOCATION

# Finally get the headers and the dso final paths
TFLITE_H = os.path.join(SCRIPT_LOCATION, "tflite_c.h")
TFLITE_DSO = os.path.join(LIB_LOCATION, "tflite_c.so")

# Load the TFLite DSO
ffi = FFI()
ffi.cdef(open(TFLITE_H).read())
tflite_c_lib = ffi.dlopen(TFLITE_DSO)


class Model:
  """A TFLite Image Classification Inference Model."""

  def __init__(self, filename, labels, threads=4):
    """Load a model.

    Args:
      filename: Filename of the tflite model
      labels: List of labels one per line
      threads: Number of threads to use for inference.
    """
    self.c_lib = tflite_c_lib
    self.model_handle_ = self.c_lib.TfLiteLoadImageClassifyModel(filename, threads)
    self.labels = [x[:-1] for x in open(labels).readlines()]
    self.inference_type = np.uint8 if self.c_lib.TfLiteIsUint8Quant(self.model_handle_) else np.float32
    self.probs = np.zeros((len(self.labels),), self.inference_type)

    if not self.c_lib.TfLiteValid(self.model_handle_):
      raise RuntimeError(self.error())

  def error(self):
    """Get the error string from the model."""
    stuff = self.c_lib.TfLiteErrorString(self.model_handle_)
    return ffi.string(stuff)

  def invoke(self, image_in, num):
    """Invoke an inference on a numpy array of image_in.

    Args:
      image_in: A np.array() of type np.uint8
    Returns:
      the Top `num` inference labels as a sorted list of tuples.
      i.e. (label_index, label_name, probability).
    """
    if self.inference_type == np.float32:
      image_in = image_in.astype(np.float32) / 255.  - 0.5
    image_in_buf = image_in.__array_interface__['data'][0]
    image_in_cptr = ffi.cast("unsigned char*", image_in_buf )
    probs_out_buf = self.probs.__array_interface__['data'][0]
    probs_out_cptr = ffi.cast("unsigned char*", probs_out_buf)
    success = self.c_lib.TfLiteRunImageClassify(
        self.model_handle_,
        image_in.shape[1], image_in.shape[2],
        image_in.shape[3], image_in_cptr, image_in.nbytes,
        probs_out_cptr, self.probs.nbytes)
    index_and_probs = list(enumerate(self.probs))
    index_and_probs.sort(key=lambda x: -float(x[1]))

    def prob_convert(x):
      """Convert x to a probability [0,1] if the type is uint8."""
      return x/255. if self.inference_type==np.uint8 else x

    return [(index, self.labels[index], prob_convert(prob))
            for index,prob in index_and_probs[:num]]

    if not success:
      raise RuntimeError(self.error())

  def __del__(self):
    """Destructor for the tflite model."""
    self.c_lib.TfLiteFree(self.model_handle_)

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print ("Usage: %s <model tflite> <labels> <image> " % sys.argv[0])
    sys.exit(1)

  # Read the image
  import Image
  img = Image.open(sys.argv[3]).convert('RGB').resize((224,224))
  npimg = np.asarray(img)
  reshaped_img = np.reshape(npimg, [1] + list(npimg.shape ) )

  # Build the model
  foo = Model(sys.argv[1], sys.argv[2])
  # Inference and timing
  start = time.time()
  inference = foo.invoke(reshaped_img,4)
  elapsed = float(time.time() - start)
  print ("Inference Result:")
  for index, label, prob in inference:
    print ("%*s | %4.0f%% | %s (%d) " % (
        32, "="*int(prob * 32), prob*100., label, index))
  print ("Elapsed Time: %f s" % elapsed)

