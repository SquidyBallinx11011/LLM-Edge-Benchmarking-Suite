import argparse
import time
import csv
import json
import requests
from ollama import chat

# --------------------------------------------------------
#                   CSV LOGGING (AVERAGES)
# --------------------------------------------------------
def benchmark_csv(csv_file, model_name, device, num_runs, num_tokens, elapsed, response, ttft=None):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow([
                "model", "device", "num_runs", "num_tokens", "avg_time_s",
                "avg_tokens_per_sec", "avg_ttft_s", "response"
            ])

        writer.writerow([
            model_name,
            device,
            num_runs,
            num_tokens,
            f"{elapsed:.4f}",
            f"{num_tokens / elapsed:.2f}",
            f"{ttft:.4f}" if ttft else "",
            response["message"]["content"]
        ])


# --------------------------------------------------------
#                   PER RUN DETAILS CSV
# --------------------------------------------------------
def log_run_details(csv_file, model_name, device, num_tokens, deltas):
    details_file = csv_file.replace(".csv", "_details.csv")

    with open(details_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow([
                "model", "device", "num_tokens",
                "run_elapsed", "elapsed_delta",
                "run_ttft", "ttft_delta",
                "response"
            ])

        for d in deltas:
            writer.writerow([
                model_name,
                device,
                num_tokens,
                f"{d['elapsed']:.4f}",
                f"{d['elapsed_delta']:.4f}",
                f"{d['ttft']:.4f}",
                f"{d['ttft_delta']:.4f}",
                d["response"]["message"]["content"]
            ])


# --------------------------------------------------------
#                   AVERAGING HELPER
# --------------------------------------------------------
def average_runs(results):
    avg_elapsed = sum(r["elapsed"] for r in results) / len(results)
    avg_ttft = sum(r["ttft"] for r in results) / len(results)

    deltas = []
    for r in results:
        deltas.append({
            "elapsed": r["elapsed"],
            "elapsed_delta": r["elapsed"] - avg_elapsed,
            "ttft": r["ttft"],
            "ttft_delta": r["ttft"] - avg_ttft,
            "response": r["response"]
        })

    return {
        "elapsed": avg_elapsed,
        "ttft": avg_ttft,
        "response": results[-1]["response"],
        "deltas": deltas
    }


# --------------------------------------------------------
#                   CPU OLLAMA BENCHMARK
# --------------------------------------------------------
def benchmark_cpu(model_name, num_tokens):
    print(f"\n\n___CPU Ollama Benchmark: {model_name}___")

    # Warmup
    print("Starting warmup...")
    warm_start = time.perf_counter()

    warm_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Warmup"}],
        "stream": False,
        "options": {"num_gpu": 0}
    }

    warm_resp = requests.post("http://localhost:11434/api/chat", json=warm_payload)
    warm_end = time.perf_counter()
    warm_elapsed = warm_end - warm_start

    print(f"Warmup completed in {warm_elapsed:.4f}s")
    print("Warmup response:", warm_resp.text[:200], "...")

    # Main streaming benchmark
    print("Starting streamed generation...")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Explain why the sky is blue in one paragraph."}
        ],
        "options": {"num_predict": num_tokens, "num_gpu": 0},
        "stream": True
    }

    start = time.perf_counter()
    ttft = None
    full_text = ""

    r = requests.post("http://localhost:11434/api/chat", json=payload, stream=True)

    try:
        for line in r.iter_lines():
            if not line:
                continue

            if ttft is None:
                ttft = time.perf_counter() - start
                print(f"TTFT: {ttft:.4f}s")

            data = json.loads(line.decode("utf-8"))
            delta = data["message"]["content"]
            if delta:
                full_text += delta
    finally:
        r.close()

    end = time.perf_counter()
    elapsed = end - start

    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Tokens/sec: {num_tokens / elapsed:.2f}")
    print("Final response:", full_text)

    return {
        "elapsed": elapsed,
        "ttft": ttft,
        "response": {"message": {"content": full_text}}
    }

# --------------------------------------------------------
#                   GPU OLLAMA BENCHMARK
# --------------------------------------------------------
def benchmark_gpu(model_name, num_tokens):
    print(f"\n\n___CPU Ollama Benchmark: {model_name}___")

    # Warmup
    print("Starting warmup...")
    warm_start = time.perf_counter()

    warm_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Warmup"}],
        "stream": False,
        "options": {"num_gpu": 1}
    }

    warm_resp = requests.post("http://localhost:11434/api/chat", json=warm_payload)
    warm_end = time.perf_counter()
    warm_elapsed = warm_end - warm_start

    print(f"Warmup completed in {warm_elapsed:.4f}s")
    print("Warmup response:", warm_resp.text[:200], "...")

    # Main streaming benchmark
    print("Starting streamed generation...")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Explain why the sky is blue in one paragraph."}
        ],
        "options": {"num_predict": num_tokens, "num_gpu": 1},
        "stream": True
    }

    start = time.perf_counter()
    ttft = None
    full_text = ""

    r = requests.post("http://localhost:11434/api/chat", json=payload, stream=True)

    try:
        for line in r.iter_lines():
            if not line:
                continue

            if ttft is None:
                ttft = time.perf_counter() - start
                print(f"TTFT: {ttft:.4f}s")

            data = json.loads(line.decode("utf-8"))
            delta = data["message"]["content"]
            if delta:
                full_text += delta
    finally:
        r.close()

    end = time.perf_counter()
    elapsed = end - start

    print(f"Total time: {elapsed:.4f} seconds")
    print(f"Tokens/sec: {num_tokens / elapsed:.2f}")
    print("Final response:", full_text)

    return {
        "elapsed": elapsed,
        "ttft": ttft,
        "response": {"message": {"content": full_text}}
    }


# --------------------------------------------------------
#                 GPU hailo OLLAMA BENCHMARK
# --------------------------------------------------------
def benchmark_gpu_hailo(model_name, num_tokens):
    print(f"\n\n___GPU hailo-Ollama Benchmark: {model_name}___")
    print("Starting streamed generation (GPU)...")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Explain why the sky is blue in one paragraph."}
        ],
        "options": {"num_predict": num_tokens},
        "stream": True
    }

    start = time.perf_counter()
    ttft = None
    full_text = ""

    r = requests.post("http://localhost:8000/api/chat", json=payload, stream=True)

    try:
        for line in r.iter_lines():
            if not line:
                continue

            if ttft is None:
                ttft = time.perf_counter() - start
                print(f"TTFT: {ttft:.4f}s")

            data = json.loads(line.decode("utf-8"))
            delta = data["message"]["content"]
            if delta:
                full_text += delta
    finally:
        r.close()

    end = time.perf_counter()
    elapsed = end - start

    print(f"Total time: {elapsed:.4f}s")
    print(f"Tokens/sec: {num_tokens / elapsed:.2f}")
    print("Final response:", full_text)

    return {
        "elapsed": elapsed,
        "ttft": ttft,
        "response": {"message": {"content": full_text}},
    }


# --------------------------------------------------------
#                COMMAND-LINE INTERFACE
# --------------------------------------------------------
def load_models_from_args(args):
    models = []

    if args.models:
        models.extend(args.models)

    if args.models_file:
        with open(args.models_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    models.append(line)

    if not models:
        raise ValueError("No models provided. Use --models or --models-file.")

    return models


def main():
    parser = argparse.ArgumentParser(description="Ollama Benchmark with TTFT, repeat, CPU/GPU selection")
    parser.add_argument("--models", "-m", nargs="*", help="List of model names")
    parser.add_argument("--models-file", help="File with model names (one per line)")
    parser.add_argument("--num-tokens", type=int, default=20)
    parser.add_argument("--csv-file", default="benchmark_results.csv")
    parser.add_argument("--device", choices=["cpu", "gpu", "gpu-hailo", "both"], default="both",
                        help="Which device(s) to benchmark: cpu, gpu, or both")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of repeated runs for averaging")

    args = parser.parse_args()
    models = load_models_from_args(args)
    num_runs = args.repeat

    for model_name in models:

        # CPU Benchmarking
        if args.device in ("cpu", "both"):
            cpu_runs = [benchmark_cpu(model_name, args.num_tokens) for _ in range(num_runs)]
            cpu_avg = average_runs(cpu_runs)

            # Summary CSV
            benchmark_csv(
                args.csv_file, model_name, "CPU-Ollama", num_runs,
                args.num_tokens, cpu_avg["elapsed"], cpu_avg["response"], cpu_avg["ttft"]
            )

            # Detail CSV with per-run deltas
            log_run_details(
                args.csv_file, model_name, "CPU-Ollama",
                args.num_tokens, cpu_avg["deltas"]
            )

        # GPU Benchmarking
        if args.device in ("gpu", "both"):
            gpu_runs = [benchmark_gpu(model_name, args.num_tokens) for _ in range(num_runs)]
            gpu_avg = average_runs(gpu_runs)

            benchmark_csv(
                args.csv_file, model_name, "GPU-Ollama", num_runs,
                args.num_tokens, gpu_avg["elapsed"], gpu_avg["response"], gpu_avg["ttft"]
            )

            log_run_details(
                args.csv_file, model_name, "GPU-hailo-Ollama",
                args.num_tokens, gpu_avg["deltas"]
            )

        # GPU Hailo Benchmarking
        if args.device in ("gpu-hailo"):
            gpu_runs = [benchmark_gpu_hailo(model_name, args.num_tokens) for _ in range(num_runs)]
            gpu_avg = average_runs(gpu_runs)

            benchmark_csv(
                args.csv_file, model_name, "GPU-Hailo-Ollama", num_runs,
                args.num_tokens, gpu_avg["elapsed"], gpu_avg["response"], gpu_avg["ttft"]
            )

            log_run_details(
                args.csv_file, model_name, "GPU-hailo-Ollama",
                args.num_tokens, gpu_avg["deltas"]
            )


if __name__ == "__main__":
    main()