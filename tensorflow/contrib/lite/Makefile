
# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = WINDOWS
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	        HOST_OS := LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := OSX
	endif
endif
HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)


# Self-hosting
TARGET_ARCH := ${HOST_ARCH}

# Cross compiling
ifeq ($(CROSS),rpi)
  TARGET_ARCH := armv7l
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabihf-
endif

ifeq ($(CROSS),stm32f7)
  TARGET_ARCH := armf7
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
endif
ifeq ($(CROSS),stm32f1)
  TARGET_ARCH := armm1
  TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-
endif



# Where compiled objects are stored.
OBJDIR := $(MAKEFILE_DIR)/gen/obj/
BINDIR := $(MAKEFILE_DIR)/gen/bin/
LIBDIR := $(MAKEFILE_DIR)/gen/lib/
GENDIR := $(MAKEFILE_DIR)/gen/obj/


ifeq ($(TARGET_ARCH),x86_64)
	CXXFLAGS += -fPIC -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -pthread # -msse4.2
endif

ifeq ($(TARGET_ARCH),armv7l)
	CXXFLAGS += -mfpu=neon -pthread -fPIC
endif

ifeq ($(TARGET_ARCH),armf7)
	CXXFLAGS += -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
	CXXFLAGS += -fno-rtti -fmessage-length=0 -fno-exceptions -fno-builtin -ffunction-sections -fdata-sections
	CXXFLAGS += -funsigned-char -MMD
endif
ifeq ($(TARGET_ARCH),armm1)
	CXXFLAGS += -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -mcpu=cortex-m1 -mthumb
	CXXFLAGS += -fno-rtti -fmessage-length=0 -fno-exceptions -fno-builtin -ffunction-sections -fdata-sections
	CXXFLAGS += -funsigned-char -MMD
endif

#AS      = 'arm-none-eabi-gcc' '-x' 'assembler-with-cpp' '-c' '-Wall' '-Wextra' '-Wno-unused-parameter' '-Wno-missing-field-initializers' '-fmessage-length=0' '-fno-exceptions' '-fno-builtin' '-ffunction-sections' '-fdata-sections' '-funsigned-char' '-MMD' '-fno-delete-null-pointer-checks' '-fomit-frame-pointer' '-Os' '-mcpu=cortex-m7' '-mthumb' '-mfpu=fpv5-sp-d16' '-mfloat-abi=softfp'
#CC      = 'arm-none-eabi-gcc' '-std=gnu99' '-c' '-Wall' '-Wextra' '-Wno-unused-parameter' '-Wno-missing-field-initializers' '-fmessage-length=0' '-fno-exceptions' '-fno-builtin' '-ffunction-sections' '-fdata-sections' '-funsigned-char' '-MMD' '-fno-delete-null-pointer-checks' '-fomit-frame-pointer' '-Os' '-mcpu=cortex-m7' '-mthumb' '-mfpu=fpv5-sp-d16' '-mfloat-abi=softfp'
#CPP     = 'arm-none-eabi-g++' '-std=gnu++11' '-fno-rtti' '-Wvla' '-c' '-Wall' '-Wextra' '-Wno-unused-parameter' '-Wno-missing-field-initializers' '-fmessage-length=0' '-fno-exceptions' '-fno-builtin' '-ffunction-sections' '-fdata-sections' '-funsigned-char' '-MMD' '-fno-delete-null-pointer-checks' '-fomit-frame-pointer' '-Os' '-mcpu=cortex-m7' '-mthumb' '-mfpu=fpv5-sp-d16' '-mfloat-abi=softfp'
#LD      = 'arm-none-eabi-gcc'


# Settings for the host compiler.
CXX := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}g++
CXXFLAGS += --std=c++11 -O3 -DNDEBUG
CXXFLAGS += '-std=gnu++11' '-fno-rtti' '-Wvla' '-c' '-Wall' '-Wextra' '-Wno-unused-parameter' '-Wno-missing-field-initializers' '-fmessage-length=0' '-fno-exceptions' '-fno-builtin' '-ffunction-sections' '-fdata-sections' '-funsigned-char' '-MMD' '-fno-delete-null-pointer-checks' '-fomit-frame-pointer' '-Os' '-mcpu=cortex-m7' '-mthumb' '-mfpu=fpv5-sp-d16' '-mfloat-abi=softfp'


CCFLAGS := ${CXXFLAGS}
CC := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}gcc
AR := $(CC_PREFIX) ${TARGET_TOOLCHAIN_PREFIX}ar
CFLAGS :=
LDOPTS :=
LDOPTS += -L/usr/local/lib -ldl
ARFLAGS := -r

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(GENDIR)
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

LIBS := \
-lstdc++ \
-lpthread \
-lm \
-ldl

# If we're on Linux, also link in the dl library.
ifeq ($(OS),LINUX)
	LIBS += -ldl -lpthread
endif

include $(MAKEFILE_DIR)/ios_makefile.inc

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a
LIB_PATH := $(LIBDIR)$(LIB_NAME)

# A small example program that shows how to link against the library.
BENCHMARK_PATH := $(BINDIR)benchmark_model

BENCHMARK_SRCS := \
tensorflow/contrib/lite/tools/test_cortex.cc
BENCHMARK_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_SRCS))))


PYTHON_CAMERA_PATH := $(BINDIR)tflite_c.so
PYTHON_CAMERA_SRCS := tensorflow/contrib/lite/examples/python_camera/tflite_c.cc
PYTHON_CAMERA_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(PYTHON_CAMERA_SRCS))))

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

# $(wildcard tensorflow/contrib/lite/kernels/*.cc) \
# $(wildcard tensorflow/contrib/lite/kernels/internal/*.cc) \
# $(wildcard tensorflow/contrib/lite/kernels/internal/optimized/*.cc) \
# $(wildcard tensorflow/contrib/lite/kernels/internal/reference/*.cc) \


CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/contrib/lite/*.cc) \
$(wildcard tensorflow/contrib/lite/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/optimized/*.c) \
$(wildcard tensorflow/contrib/lite/kernels/internal/reference/*.c) \
$(wildcard tensorflow/contrib/lite/downloads/farmhash/src/farmhash.cc)
# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard tensorflow/contrib/lite/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/*/*/*/*test.cc) \
$(wildcard tensorflow/contrib/lite/kernels/test_util.cc) \
$(BENCHMARK_SRCS)
# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))
# File names of the intermediate files target compilation generates.
TF_LITE_CC_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(TF_LITE_CC_SRCS))))
LIB_OBJS := $(TF_LITE_CC_OBJS)

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow C++ source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH) $(BENCHMARK_PATH) $(PYTHON_CAMERA_PATH)

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

$(BENCHMARK_PATH): $(BENCHMARK_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_PATH) $(BENCHMARK_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

#$(PYTHON_CAMERA_PATH): $(PYTHON_CAMERA_OBJS) $(LIB_PATH)
#	@mkdir -p $(dir $@)
#	$(CXX) $(CXXFLAGS) $(INCLUDES) \
#	-shared -o $(PYTHON_CAMERA_PATH) $(PYTHON_CAMERA_OBJS) \
#	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)



# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TF_CC_SRCS)))
