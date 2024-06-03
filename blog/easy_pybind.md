---
layout: post
date: 2024-06-03
---

Start Writing Kernels Faster Before Writing Faster Kernels
==========================================================

[[Hieu's personal blog index](./index)]

In this blog post, I introduce a simple tool to *accelerate* and *simplify*the
process of writing and exeperimenting with kernels.

## Motivation

While Python is the de-facto language for AI/ML, Python programs can be painfully
slow.  As such, AI/ML practitioners who develop latency sensitive applications
often resort to writing *kernels* to accelerate their workloads.

Now, unless you work for Google, your kernels are probably written CUDA, exposed
via a C/C++ interface, and binded to Python as a package via PyBind11. This
process, despite being conceptually simple, usually results in an appalling
amount of *boilerplate complexity.* Here, "boiplerplate complexity" loosely
means the code, structure of files, libraries, etc. that you need to install
correctly *before* you can write the first line of code in your kernel. Let us
look at two examples.

- One popular kernel is
  [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main).
  The project has a
  [`setup.py`](https://github.com/Dao-AILab/flash-attention/blob/320fb59487658f033f56711efd3d61b7c7a6f8f3/setup.py#L19-L25)
  that compiles the kernels in
  [`csrc/flash_attn`](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/flash_attn),
  and binds them to
  [`flash_attn/flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/320fb59487658f033f56711efd3d61b7c7a6f8f3/flash_attn/flash_attn_interface.py#L1).
  The
  [`setup.py`](https://github.com/Dao-AILab/flash-attention/blob/320fb59487658f033f56711efd3d61b7c7a6f8f3/setup.py)
  file is more than 350 lines long. It depends on PyTorch, which
  makes the compilation of the kernels use many flags that are not exposed to
  users.

- A more canonical example is
  [PyTorch's official guide to write CPP extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html).
  While much simpler than FlashAttention, the guide still relies on a canned
  `setup.py`, which results in many inconveniences. When I followed the guide,
  my first annoyance is that I *must* have PyTorch installed on my system, even
  though my kernels are relatively framework-agnostic. Another, much more
  pestering annoyance, is that I have no control which compilation flags are
  used in the process.

This boilerplate complexity obscures the core logic of compiling, linking, and
binding kernels, turning a conceptually simple process into something opaque and
untracable. In turn, this creates a steep entrance barrier for anyone aspired to
develop or research kernels.

## EasyPybind -- a simple tool to kickstart writing kernels

Writing kernels is a complex business. But the complexity should go into
designing, implementing, optimizing, and testing the kernels, rather than into
the first step: to kickstart writing the kernels.

In fact, this first step is simple enough to be automated. And I have written a
simple, minimalistic tool called EasyPybind to do exactly that.

### Installation
EasyPybind is available [on GitHub](https://github.com/hyhieu/easy_pybind). It
can be installed via:
```bash
$ pip install easy_pybind
$ easy-pybind create --help
# usage: easy_pybind create [-h] --module-name MODULE_NAME [...]
```

### Usage example
Once installed, the command:
```bash
$ easy-pybind create --module-name="cu_example" --cuda
```
will generate a directory that looks like this:
```bash
cu_kernel/
├─ .gitignore            # ignore build artifacts
├─ build.sh              # build the module
├─ clean.sh              # clean up the build
├─ src/
│  ├─ cu_kernel.cc       # entry to the module
│  ├─ cu_kernel_impl.cu  # implementation of the module in CUDA
│  └─ cu_kernel_impl.h
├─ cu_kernel_test.py     # if you have --with-pytest
└─ main.py               # if you have --with-pymain
```
The file `cu_kernel_impl.cu` contains a rather simple function that adds two
integers.  The function is introduced in the header file `cu_kernel_impl.h`, and
is exposed through a C/C++ interface in `cu_kernel.cc`.

The `build.sh` script will compile the module and link it with the CUDA runtime.
If you look into the default `build.sh`, you will see it has 3 components:
- A call to `nvcc` that compiles the kernel `cu_kernel_impl.cu` into `cu_kernel_impl.o`.
- A call to `g++` that compiles the interface `cu_kernel.cc` into `cu_kernel.o`.
- A second call to `g++` that links `cu_kernel.o` and `cu_kernel_impl.o` into
  `cu_kernel.so`.

Running `./build.sh` results in the `cu_kernel.so` file in the same directory
with a `main.py` or a `cu_kernel_test.py` file. As long a you have this `cu_kernel.so`
file, you can do:
```python
import cu_kernel
cu_kernel.add(1, 2)
```
To write more complex kernels, you would start from changing its interface in
`cu_kernel.cc`, and the propagate the changes to `cu_kernel_impl.h` and
`cu_kernel_impl.cu`.

If you want to change your compilation process, for instance, to add some CUDA
libraries, you have all the raw skeleton in `build.sh`. If you want to add flags
like `-NDEBUG`, there goes the `nvcc` call in `build.sh`. I usually add:
```bash
  -I/path/to/my/cutlass/library/include
```
to use my beloved [CUTLASS](https://github.com/NVIDIA/cutlass) library by NVIDIA.

## Conclusion
I have been using EasyPybind in my own research to develop kernels. I like it
quite a lot, so I decided to share it with the world.

I do not intend to turn it into a large and untractable project which I don't
have time to develop and maintain. But if you have any suggestions, please let me
know. I am available at my email `hyhieu [at] gmail [dot] com`.
