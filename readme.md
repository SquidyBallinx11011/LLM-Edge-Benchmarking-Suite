# Ollama Latency & Throughput Benchmarking Suite

A lightweight, repeatable benchmarking toolkit for measuring **LLM performance** across:

- ✅ Local CPU-based Ollama inference  
- ✅ Local GPU-based Ollama inference  
- ✅ External accelerators (e.g., Hailo, Helios) via custom endpoints  
- ✅ Multiple models  
- ✅ Multiple repeated runs  
- ✅ Time-To-First-Token (TTFT) and total generation time  
- ✅ Per-run variability (delta vs average)

This tool is designed for reproducible latency benchmarking, model comparisons, and hardware performance evaluation.

---

## 🚀 Features

### ✅ **CPU Benchmarking**
Runs the selected model on the local Ollama CPU backend with warmup, TTFT, and full streaming latency.

### ✅ **Generic Ollama GPU Benchmarking**
Identical to the CPU benchmark, but explicitly configured to use local GPUs (`num_gpu = 1`).

### ✅ **Hailo (Custom GPU Backend)**
Supports custom API endpoints for any remote Ollama-compatible accelerator.

### ✅ **Repeatable Runs**
Use `--repeat N` to run each model multiple times.

The framework automatically calculates:
- Average TTFT  
- Average total generation time  
- Tokens/sec  
- Per-run deltas vs the average  

### ✅ **CSV Logging**
Two output files:

#### 1. **benchmark_results.csv**
Contains aggregated averages for each model × device.

#### 2. **benchmark_results_details.csv**
Contains *every* run, including:
- Run elapsed time  
- Run TTFT  
- Delta from the average  
- Raw response text  

Perfect for plotting, statistical analysis, and performance drift checking.

---

## 📦 Installation

Install dependencies:

```bash
pip install requests ollama