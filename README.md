# Naive Bayes Classifier prototype
## *Naive Bayes Classifier on GPU using boost.compute*

### Setup project for ubuntu 20.04

#### install OpenCL

1 ``sudo apt install ocl-icd-libopencl1``

next install OpenCL for your system
 
for nvidia :

2 ``sudo apt install nvidia-opencl-dev `` 

#### install boost, cmake

``sudo apt install libboost-all-dev cmake``

it's possible to install subset of useb boost libraries - program_potions and unit_test_framework

#### install boost.compute

```
git clone git://github.com/kylelutz/compute.git

cd compute

cmake .

sudo make install
```

source : http://boostorg.github.io/compute/boost_compute/getting_started.html

##### Reference:

http://boostorg.github.io/compute/
https://www.boost.org/doc/libs/1_74_0/libs/compute/doc/html/index.html

### Program Usage

##### Options:

## ```TODO```

### Build instructions

- Project uses *boost* library for testing and parsing program parameters 
- When calling CMake for the first time, all needed compiler options must be
  specified on the command line.  After this initial call to CMake, the compiler
  definitions must not be included for further calls to CMake.  Other options
  can be specified on the command line multiple times including all definitions
  in the build options section below.
- Example of configuring, building, reconfiguring, rebuilding:

  ````
  # Initial configuration
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_CXX_COMPILER=clang++ ..  
    $ make
    ...
  # Second configuration
    $ make clean
    $ cmake -DCMAKE_BUILD_TYPE=Debug ..                               
    $ make
    ...
  # Third configuration
    $ rm -rf *
    $ cmake -DCMAKE_CXX_COMPILER=g++ ..        
    $ make


- CMake variables
    - **CMAKE_BUILD_TYPE** = ``Release|Debug``
      Build type can be ``Release``, ``Debug`` which chooses
      the optimization level and presence of debugging symbols.
    
    - **CMAKE_CXX_COMPILER** = <C++ compiler name>
      Specify the C++ compiler.
  
- For full documentation consult the CMake manual or execute
    ```
    cmake --help-variable VARIABLE_NAME 
  
- Project targets :

    - **build-all** - builds all available targets
    - **BGM-asm-test** - UnitTest made with *boost*
    - **BGM-asm-check** - dummy target for test run during library compilation
    - **BGM-asm-library** - library containing all classes excluding parameters
    - **BGM-asm-program** - program available for command line usage
