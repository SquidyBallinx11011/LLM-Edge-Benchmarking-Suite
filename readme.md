# LLM-Edge-Benchmarking-Suite

A lightweight, repeatable benchmarking toolkit for measuring **LLM performance** across heterogeneous inference backends.

This repository code was used to gather results for the paper: 'CLOUD TO EDGE: BENCHMARKING LLM INFERENCE ON HARDWARE-ACCELERATED SINGLE-BOARD COMPUTERS'

This suite supports:

- ✅ Local CPU-based Ollama inference  
- ✅ Local GPU-based Ollama inference  
- ✅ External accelerators via custom endpoints  
- ✅ Dedicated low‑level accelerator benchmarking (AX630)  
- ✅ Multiple models  
- ✅ Multiple repeated runs  
- ✅ Time‑To‑First‑Token (TTFT)  
- ✅ Total generation latency  
- ✅ Tokens/sec throughput  
- ✅ Per‑run variability (delta vs average)  

The primary goal is to enable **reproducible latency benchmarking**, **model comparison**, and **hardware performance evaluation** across CPUs, GPUs, and purpose‑built AI accelerators.

---

## 🚀 Features

### ✅ CPU Benchmarking
Runs the selected model on the local Ollama **CPU backend**, including:

- Optional warmup run  
- TTFT measurement  
- Full streaming generation latency  
- Tokens/sec calculation  

This provides a CPU baseline for comparing other inference backends.

---

### ✅ Generic Ollama GPU Benchmarking
Identical to CPU benchmarking but explicitly configured to use local GPUs:

- Uses `num_gpu = 1`  
- Measures TTFT and total generation latency  
- Supports repeated runs for variance analysis  

This enables apples‑to‑apples **CPU vs GPU** comparisons.

---

### ✅ Remote / Custom Ollama‑Compatible Accelerators
Supports any **Ollama‑compatible HTTP endpoint**, allowing benchmarking of:

- Hailo NPU
- AX630c NPU

As long as the endpoint follows Ollama’s streaming API behaviour, it can be benchmarked without changes to the tooling. [ToDo - Make sure this full support is added.]

---

### ✅ AX630 Low‑Level Accelerator Benchmarking

The repository includes a **dedicated Python benchmarking script for AX630** that communicates directly with the device over its **native TCP socket interface**, bypassing Ollama entirely.

#### Why a dedicated script?
AX630 exposes a lower‑level streaming protocol. Measuring it via raw sockets avoids:

- HTTP overhead  
- Client buffering artefacts  
- Ollama scheduling effects  

This results in **device‑accurate latency and TTFT measurements**.

#### Key Characteristics
- Single persistent session (`setup` once, infer many times)
- Prefill phase excluded from timing
- True generation‑only TTFT
- Per‑run and averaged metrics
- Automatic CSV logging
- Graceful teardown
- Explicit failure on task‑full conditions

---

### ✅ Repeatable Runs
All benchmarks support multiple runs per model.

For each benchmark, the framework automatically computes:

- Average elapsed (generation) time  
- Average TTFT  
- Tokens/sec  
- Per‑run deviation from the mean  

This highlights run‑to‑run variance, jitter, and stability.

---

### ✅ CSV Logging

Benchmarks write results in a **consistent CSV format** suitable for plotting and statistical analysis.

#### `benchmark_results.csv`
Aggregated results per **model × device**, including:

- Number of runs  
- Token count  
- Average elapsed time  
- Average tokens/sec  
- Average TTFT  
- Example response text  

#### `benchmark_results_details.csv`
Per‑run detail logging, including:

- Run elapsed time  
- Run TTFT  
- Delta vs average elapsed time  
- Delta vs average TTFT  
- Full raw response  

This is ideal for:
- Variance analysis  
- Regression detection  
- Thermal / throttling studies  

---

## 📦 Installation

### Ollama‑based Benchmarks

```bash
pip install requests ollama