README.md                {#LREADME}
=========
# AVM Codec Library

## Contents
1. [Building the lib and applications](#building-the-library-and-applications)
    - [Prerequisites](#prerequisites)
    - [Get the code](#get-the-code)
    - [Basics](#basic-build)
    - [Configuration options](#configuration-options)
    - [Dylib builds](#dylib-builds)
    - [Debugging](#debugging)
    - [Cross compiling](#cross-compiling)
    - [Sanitizer support](#sanitizers)
    - [MSVC builds](#microsoft-visual-studio-builds)
    - [Xcode builds](#xcode-builds)
    - [Build with VMAF support](#build-with-vmaf)
2. [Testing the library](#testing-the-avm-codec)
    - [Basics](#testing-basics)
        - [Unit tests](#1_unit-tests)
        - [Example tests](#2_example-tests)
    - [Downloading test data](#downloading-the-test-data)
    - [Sharded testing](#sharded-testing)
3. [Coding style](#coding-style)
4. [Submitting patches](#submitting-patches)
    - [Testing your code](#testing-your-code)
5. [Support](#support)
6. [Bug reports](#bug-reports)

## Building the library and applications {#building-the-library-and-applications}

### Prerequisites {#prerequisites}

 1. [CMake](https://cmake.org) version 3.16 or higher.
 2. [Git](https://git-scm.com/).
 3. [Perl](https://www.perl.org/).
 4. For x86 targets, [yasm](http://yasm.tortall.net/), which is preferred, or a
    recent version of [nasm](http://www.nasm.us/). If you download yasm with
    the intention to work with Visual Studio, please download win32.exe or
    win64.exe and rename it into yasm.exe. DO NOT download or use vsyasm.exe.
 5. Building the documentation requires
   [doxygen version 1.8.10 or newer](http://doxygen.org).
 6. Building the unit tests requires [Python](https://www.python.org/).

### Get the code {#get-the-code}

The AVM project source code is stored in the Alliance of Open Media’s GitLab [repository](https://gitlab.com/AOMediaCodec/avm).
To get the code, 

~~~
    $ git clone https://gitlab.com/AOMediaCodec/avm.git
    # By default, the above command stores the source in the avm directory:
    $ cd avm
~~~

### Basic build {#basic-build}

CMake replaces the configure step typical of many projects. Running CMake will
produce configuration and build files for the currently selected CMake
generator. For most systems the default generator is Unix Makefiles. The basic
form of a makefile build is the following:

~~~
    $ cmake path/to/avm
    $ make
~~~

The above will generate a makefile build that produces the AVM library and
applications for the current host system after the make step completes
successfully. The compiler chosen varies by host platform, but a general rule
applies: On systems where cc and c++ are present in $PATH at the time CMake is
run the generated build will use cc and c++ by default.

### Configuration options {#configuration-options}

The AVM codec library has a great many configuration options. These come in two
varieties:

 1. Build system configuration options. These have the form `ENABLE_FEATURE`.
 2. AVM codec configuration options. These have the form `CONFIG_FEATURE`.

Both types of options are set at the time CMake is run. The following example
enables ccache and disables the AVM encoder:

~~~
    $ cmake path/to/avm -DENABLE_CCACHE=1 -DCONFIG_MULTITHREAD=0
    $ make
~~~

The available configuration options are too numerous to list here. Build system
configuration options can be found at the top of the CMakeLists.txt file found
in the root of the AVM repository, and AVM codec configuration options can
currently be found in the file `build/cmake/aom_config_defaults.cmake`.

### Dylib builds {#dylib-builds}

A dylib (shared object) build of the AVM codec library can be enabled via the
CMake built in variable `BUILD_SHARED_LIBS`:

~~~
    $ cmake path/to/avm -DBUILD_SHARED_LIBS=1
    $ make
~~~

This is currently only supported on non-Windows targets.

### Debugging {#debugging}

Depending on the generator used there are multiple ways of going about
debugging AVM components. For single configuration generators like the Unix
Makefiles generator, setting `CMAKE_BUILD_TYPE` to Debug is sufficient:

~~~
    $ cmake path/to/avm -DCMAKE_BUILD_TYPE=Debug
~~~

For Xcode, mainly because configuration controls for Xcode builds are buried two
configuration windows deep and must be set for each subproject within the Xcode
IDE individually, `CMAKE_CONFIGURATION_TYPES` should be set to Debug:

~~~
    $ cmake path/to/avm -G Xcode -DCMAKE_CONFIGURATION_TYPES=Debug
~~~

For Visual Studio the in-IDE configuration controls should be used. Simply set
the IDE project configuration to Debug to allow for stepping through the code.

In addition to the above it can sometimes be useful to debug only C and C++
code. To disable all assembly code and intrinsics set `AOM_TARGET_CPU` to
generic at generation time:

~~~
    $ cmake path/to/avm -DAOM_TARGET_CPU=generic
~~~

### Cross compiling {#cross-compiling}

For the purposes of building the AVM codec and applications, relative to the scope of this guide,
all builds for architectures differing from the native host architecture will be considered cross compiles.
The AVM CMake build handles cross compiling via the use of toolchain files included in the AVM repository.
The available toolchain files can be found at cmake folder in the AVM repository.
The following example demonstrates use of the x86-linux.cmake toolchain file on a x86_64 linux host:

~~~
    $ cmake path/to/avm \
      -DCMAKE_TOOLCHAIN_FILE=path/to/avm/build/cmake/toolchains/x86-linux.cmake
    $ make
~~~

To build for an unlisted target creation of a new toolchain file is the best
solution. The existing toolchain files can be used a starting point for a new
toolchain file since each one exposes the basic requirements for toolchain files
as used in the AVM codec build.

As a temporary work around an unoptimized AVM configuration that builds only C
and C++ sources can be produced using the following commands:

~~~
    $ cmake path/to/avm -DAOM_TARGET_CPU=generic
    $ make
~~~

In addition to the above it's important to note that the toolchain files
suffixed with gcc behave differently than the others. These toolchain files
attempt to obey the $CROSS environment variable.

### Sanitizers {#sanitizers}

Sanitizer integration is built-in to the CMake build system. To enable a
sanitizer, add `-DSANITIZE=<type>` to the CMake command line. For example, to
enable address sanitizer:

~~~
    $ cmake path/to/avm -DSANITIZE=address
    $ make
~~~

Sanitizers available vary by platform, target, and compiler. Consult your
compiler documentation to determine which, if any, are available.

### Microsoft Visual Studio builds {#microsoft-visual-studio-builds}

Building the AVM codec library in Microsoft Visual Studio is supported. Visual
Studio 2019 (16.7) or later is required. The following example demonstrates
generating projects and a solution for the Microsoft IDE:

~~~
    # This does not require a bash shell; Command Prompt (cmd.exe) is fine.
    # This assumes the build host is a Windows x64 computer.

    # To build with Visual Studio 2019 for the x64 target:
    $ cmake path/to/avm -G "Visual Studio 16 2019"
    $ cmake --build .

    # To build with Visual Studio 2019 for the 32-bit x86 target:
    $ cmake path/to/avm -G "Visual Studio 16 2019" -A Win32
    $ cmake --build .
~~~

NOTE: The build system targets Windows 7 or later by compiling files with
`-D_WIN32_WINNT=0x0601`.

### Xcode builds {#xcode-builds}

Building the AVM codec library in Xcode is supported. The following example
demonstrates generating an Xcode project:

~~~
    $ cmake path/to/avm -G Xcode
~~~

### Build with VMAF support {#build-with-vmaf}

After installing
[libvmaf.a](https://github.com/Netflix/vmaf/blob/master/libvmaf/README.md),
you can use it with the encoder:

~~~
    $ cmake path/to/avm -DCONFIG_TUNE_VMAF=1
~~~

Please note that the default VMAF model
will be used unless you set the following flag when running the encoder:

~~~
    # --vmaf-model-path=path/to/model
~~~

## Testing the AVM codec {#testing-the-avm-codec}

### Testing basics {#testing-basics}

There are several methods of testing the AVM codec. All of these methods require
the presence of the AVM source code and a working build of the AVM library and
applications.

#### 1. Unit tests: {#1_unit-tests}

The unit tests can be run at build time:

~~~
    # Before running the make command the LIBAOM_TEST_DATA_PATH environment
    # variable should be set to avoid downloading the test files to the
    # cmake build configuration directory.
    $ cmake path/to/avm
    # Note: The AVM CMake build creates many test targets. Running make
    # with multiple jobs will speed up the test run significantly.
    $ make runtests
~~~

#### 2. Example tests: {#2_example-tests}

The example tests require a bash shell and can be run in the following manner:

~~~
    # See the note above about LIBAOM_TEST_DATA_PATH above.
    $ cmake path/to/avm
    $ make
    # It's best to build the testdata target using many make jobs.
    # Running it like this will verify and download (if necessary)
    # one at a time, which takes a while.
    $ make testdata
    $ path/to/avm/test/examples.sh --bin-path examples
~~~

### Downloading the test data {#downloading-the-test-data}

The fastest and easiest way to obtain the test data is to use CMake to generate
a build using the Unix Makefiles generator, and then to build only the testdata
rule:

~~~
    $ cmake path/to/avm -G "Unix Makefiles"
    # 28 is used because there are 28 test files as of this writing.
    $ make -j28 testdata
~~~

The above make command will only download and verify the test data.

Additional input data for testing the encoder can be obtained from:
[AV2 - CTC](https://media.xiph.org/video/aomctc/test_set/)

### Sharded testing {#sharded-testing}

The AVM codec library unit tests are built upon gtest which supports sharding of test jobs.
Sharded test runs can be achieved in a couple of ways. Below is one example:

~~~
   # Set the environment variable GTEST_TOTAL_SHARDS to control the number of
   # shards.
   $ export GTEST_TOTAL_SHARDS=10
   # (GTEST shard indexing is 0 based).
   $ seq 0 $(( $GTEST_TOTAL_SHARDS - 1 )) \
       | xargs -n 1 -P 0 -I{} env GTEST_SHARD_INDEX={} ./test_libaom
~~~

To create a test shard for each CPU core available on the current system set
`GTEST_TOTAL_SHARDS` to the number of CPU cores on your system minus one.
The maximum number of test targets that can run concurrently is determined by
the number of CPUs on the system where the build is configured as detected by
CMake. A system with 24 cores can run 24 test shards using a value of 24 with
the `-j` parameter. When CMake is unable to detect the number of cores 10 shards
is the default maximum value.

## Coding style {#coding-style}

We are using the Google C Coding Style defined by the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

The coding style used by this project is enforced with clang-format using the
configuration contained in the
[.clang-format](https://gitlab.com/AOMediaCodec/avm/-/blob/main/.clang-format)
file in the root of the repository.

You can download clang-format using your system's package manager, or directly
from [llvm.org](http://llvm.org/releases/download.html). You can also view the
[documentation](https://clang.llvm.org/docs/ClangFormat.html) on llvm.org.
Output from clang-format varies by clang-format version,
for best results your version should match the one used by Gitlab CI. You can find the clang-format
version by reading the comment in the `.clang-format` file linked above.

Before pushing changes for review you can format your code with:

~~~
    # Apply clang-format to modified .c, .h and .cc files
    $ clang-format -i --style=file \
      $(git diff --name-only --diff-filter=ACMR '*.[hc]' '*.cc')
~~~

Check the .clang-format file for the version used to generate it if there is any
difference between your local formatting and the review system.

Some Git installations have clang-format integration. Here are some examples:

~~~
    # Apply clang-format to all staged changes:
    $ git clang-format

    # Clang format all staged and unstaged changes:
    $ git clang-format -f

    # Clang format all staged and unstaged changes interactively:
    $ git clang-format -f -p
~~~

## Submitting patches {#submitting-patches}

We manage the submission of patches using Gitlab's
[merge request](https://docs.gitlab.com/ee/user/project/merge_requests/) process.
This tool implements a workflow on top of the Git version control system to ensure that
all changes get peer reviewed and tested prior to their distribution. 

- Follow the one-time set-up steps as detailed [here](https://gitlab.com/AOMediaCodec/avm/-/wikis/AVM:-Software-Development-Workflow#1-one-time-setup).
- For pushing your code modifications, follow the steps detailed [here](https://gitlab.com/AOMediaCodec/avm/-/wikis/AVM:-Software-Development-Workflow#2-develop-a-tool-feature-bugfix-in-your-fork).
- Once the code is pushed into a branch, create a merge request as detailed [here](https://gitlab.com/AOMediaCodec/avm/-/wikis/AVM:-Software-Development-Workflow#3-create-a-merge-request-mr).
    - The code review, approval and CI/CD process will be initiated after the MR is created.
    - Once the MR is approved, the software co-ordinators will merge the MR to the main branch.
    
Note: If you are not able to submit an MR, please contact SW coordinators to make sure necessary contributor agreements are signed for the AOMedia Project.

Follow the Merge request page to check the status of the changes, review comments etc.


### Testing your code {#testing-your-code}

The testing basics are covered in the [testing section](#testing-the-avm-codec)
above.

In addition to the local tests, many more tests and/or configurations will run through Gitlab CI.

## Support {#support}

This library is an open source project supported by its community.
Please please email https://aomedia.org/contact/ for help.

## Bug reports {#bug-reports}

Bug reports can be filed in the Alliance for Open Media
[Gitlab issue tracker](https://gitlab.com/AOMediaCodec/avm/-/issues).
