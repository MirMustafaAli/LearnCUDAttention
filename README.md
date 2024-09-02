# LearnCUDAttention

*Master CUDA Programming through Attention Mechanisms*

This repository is designed to help you learn CUDA programming with a unique focus on the attention mechanism as a foundational concept. Whether you’re a beginner or an experienced developer, you’ll find a range of resources, tutorials, and projects that guide you step-by-step through the process of understanding and implementing CUDA, with a special emphasis on leveraging attention models. Dive into the world of parallel computing and deep learning, and unlock the power of GPU acceleration in your projects.

## How to Run

To run the attention mechanism implemented in C and verified using Python, follow these steps:

1. **Compile the C code into a shared library**:
    - The following command compiles `main.c` into a shared library `libattention.so`. 
    - The `-fPIC` flag ensures that the compiled code is position-independent, which is required for creating shared libraries.
    - The `-lm` flag links the math library, which is necessary for any mathematical operations performed in the C code.

    ```bash
    gcc -shared -o libattention.so -fPIC main.c -lm
    ```

2. **Run the Python script**:
    - The Python script `attention.py` calls the C implementation of the attention mechanism via the `ctypes` library.
    - It also includes a NumPy implementation of the attention mechanism, which allows you to verify and debug the logic in Python before comparing it with the output from the C implementation.

    ```bash
    python attention.py
    ```

## Current Functionality

- The current script runs the attention mechanism on the CPU, which is based on the implementation from [Andrej Karpathy's LLM.c repository](https://github.com/karpathy/llm).

## Reference

- Original Attention Mechanism CPU Implementation: [Andrej Karpathy's LLM.c repository](https://github.com/karpathy/llm)

---

This project is a work in progress. Future updates will include CUDA implementations to harness GPU acceleration, further optimizing the attention mechanism for deep learning tasks.