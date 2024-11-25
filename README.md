# Scikit++

This project has been developed by Lilian Schall and Guillaume Lalire,
Software Engineering students at EPITA.

## Notes
1. This project has been developed on a Linux system. No action have been
   explicitely taken to make it compile / run on Windows.

## What you need to do to run the project

This project is a CMake project with OpenBLAS, Google Benchmark and Google test as
required libraries.

The CMakeLists.txt located in the root folder checks if the two Google
libraries are already installed system-wide. If that is not the case, a local
copy is downloaded and stored into the `build/` folder.

In order to download the additional third-party libraries if
needed, your runtime environment should contain:
- git
- a cxx compiler (e.g. g++)
- cmake
- gnumake

OpenBLAS needs to be installed locally through your favorite package manager.
For example, in Debian based distributions, please execute the following
command:
```shell
sudo apt-get install libopenblas-dev
```

The setup of the build folder can simply be triggered using the following
command:
```shell
    make 
```

Afterwards, you need to actually compile the project.
This can be done through the following command:
```shell
    cd build;
    make -j4;
```

To run the test program, simply run the following command:
```shell
./test
```

