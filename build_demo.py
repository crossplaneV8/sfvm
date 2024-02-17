
import glob
import os


_compile_flags = [
    "-O2",      # optimizing level 2
    "-mavx2",   # enable intel AVX2 instructions
    "-mfma",    # enable intel FMA instructions
    "-s",       # minimize file size
]

_openmp_flags = [
    "-fopenmp",
    "-lgomp",
]

_include_dirs = [
    "src",
    "src/onnx",
]

_source_files = [
    "src/base/*.c",
    "src/graph/*.c",
    "src/onnx/*.c",
    "src/optimizer/*.c",
    "src/backend/*.c",
    "src/compute_lib/*.c",
    "demo/demo.c"
]


if __name__ == "__main__":

    command = "gcc"
    out_file = "demo"
    use_openmp = False

    for flag in _compile_flags:
        command += " " + flag

    if use_openmp:
        for flag in _openmp_flags:
            command += " " + flag

    for dir in _include_dirs:
        command += " -I" + dir

    for src in _source_files:
        for file in glob.glob(src):
            command += " " + file

    command += " -o " + out_file

    print("command:")
    print(command)

    os.system(command)
    print("build complete")


