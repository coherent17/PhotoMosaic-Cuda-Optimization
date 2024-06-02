# PhotoMosaic-Cuda-Optimization
Using Cuda to perform data level parallelism

## Install third-party open source code
```bash=
$ make install
```

## Install NVCC
```bash=
$ sudo apt-get update
$ sudo apt install nvidia-cuda-toolkit
```

## Install libpng
```bash=
$ sudo apt-get update
$ sudo apt-get install libpng-dev
```

## Install cmake
```bash=
$ sudo apt-get update
$ sudo apt install cmake
```

## Compile
```bash=
$   make
$   make VERBOSE=1 # show all compile details
$   make -j        # parallel compiling
```

## Execute the program
```bash=
$   ./PhotoMosaic
```