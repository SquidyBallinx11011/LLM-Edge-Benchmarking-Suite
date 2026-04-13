import socket
import json
import time
import argparse
import csv

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
            ttft_str = f"{d['ttft']:.4f}" if d["ttft"] is not None else ""
            ttft_delta_str = (
                f"{d['ttft_delta']:.4f}"
                if d["ttft_delta"] is not None
                else ""
            )

            writer.writerow([
                model_name,
                device,
                num_tokens,
                f"{d['elapsed']:.4f}",
                f"{d['elapsed_delta']:.4f}",
                ttft_str,
                ttft_delta_str,
                d["response"]["message"]["content"]
            ])


# --------------------------------------------------------
#                   AVERAGING HELPER
# --------------------------------------------------------
def average_runs(results):
    avg_elapsed = sum(r["elapsed"] for r in results) / len(results)

    # Only include runs where TTFT exists
    ttft_values = [r["ttft"] for r in results if r["ttft"] is not None]
    avg_ttft = (
        sum(ttft_values) / len(ttft_values)
        if ttft_values else None
    )

    deltas = []
    for r in results:
        deltas.append({
            "elapsed": r["elapsed"],
            "elapsed_delta": r["elapsed"] - avg_elapsed,
            "ttft": r["ttft"],
            "ttft_delta": (
                r["ttft"] - avg_ttft
                if (r["ttft"] is not None and avg_ttft is not None)
                else None
            ),
            "response": r["response"]
        })

    return {
        "elapsed": avg_elapsed,
        "ttft": avg_ttft,
        "response": results[-1]["response"],
        "deltas": deltas
    }


# ========================================================
#                AX630 LOW‑LEVEL HELPERS
# ========================================================
def ax_send(sock, data):
    sock.sendall((json.dumps(data) + "\n").encode("utf-8"))


def ax_recv(sock):
    buffer = ""
    while True:
        chunk = sock.recv(4096).decode("utf-8")
        if not chunk:
            raise RuntimeError("AX630 connection closed")
        buffer += chunk
        if "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            return json.loads(line)


# ========================================================
#                AX630 SESSION CONTROL
# ========================================================
def ax630_setup(host, port, model_name, max_tokens):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    setup_req = {
        "request_id": "llm_setup",
        "work_id": "llm",
        "action": "setup",
        "object": "llm.setup",
        "data": {
            "model": model_name,
            "response_format": "llm.utf-8.stream",
            "input": "llm.utf-8.stream",
            "enoutput": True,
            "max_token_len": max_tokens,
            "prompt": "You are a knowledgeable assistant."
        }
    }

    ax_send(sock, setup_req)
    resp = ax_recv(sock)

    error = resp.get("error", {}).get("code", 0)
    if error == -21:
        raise RuntimeError(
            "AX630 task full — device must be reset before benchmarking again"
        )
    if error != 0:
        raise RuntimeError(f"AX630 setup failed: {resp}")

    print("AX630 setup complete")
    return sock, resp["work_id"]


def ax630_infer(sock, work_id, prompt):
    # Prefill (not timed)
    ax_send(sock, {
        "request_id": "llm_prefill",
        "work_id": work_id,
        "action": "inference",
        "object": "llm.utf-8.stream",
        "data": {
            "delta": prompt,
            "index": 0,
            "finish": False
        }
    })

    # Generation (timed)
    start = time.perf_counter()
    ttft = None
    full_text = ""

    ax_send(sock, {
        "request_id": "llm_generate",
        "work_id": work_id,
        "action": "inference",
        "object": "llm.utf-8.stream",
        "data": {
            "delta": "",
            "index": 1,
            "finish": True
        }
    })

    while True:
        resp = ax_recv(sock)
        data = resp.get("data", {})

        delta = data.get("delta", "")
        finish = data.get("finish", False)

        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            full_text += delta

        if finish:
            break

    elapsed = time.perf_counter() - start

    return {
        "elapsed": elapsed,
        "ttft": ttft,
        "response": {"message": {"content": full_text}}
    }


def ax630_teardown(sock, work_id):
    ax_send(sock, {
        "request_id": "llm_exit",
        "work_id": work_id,
        "action": "exit"
    })
    ax_recv(sock)
    sock.close()
    time.sleep(0.5)


# ========================================================
#                     MAIN
# ========================================================
def main(host, port, model, num_runs):
    if args.device != "ax630":
        raise RuntimeError("This script currently only runs AX630 benchmarks")

    # Setup ONCE
    sock, work_id = ax630_setup(host, port, model, args.num_tokens)

    results = []

    try:
        # Optional warmup
        ax630_infer(sock, work_id, "Warmup prompt.")

        for i in range(num_runs):
            print(f"\nAX630 inference {i+1}/{num_runs}")
            res = ax630_infer(
                sock,
                work_id,
                "Explain why the sky is blue."
            )
            
            ttft_str = f"{res['ttft']:.4f}s" if res["ttft"] is not None else "N/A"

            print(
                f"TTFT: {ttft_str} | "
                f"Elapsed: {res['elapsed']:.4f}s | "
                f"Tokens/sec: {args.num_tokens / res['elapsed']:.2f}"
            )

            results.append(res)

    finally:
        ax630_teardown(sock, work_id)

    avg = average_runs(results)

    benchmark_csv(
        args.csv_file, model, "AX630",
        num_runs, args.num_tokens,
        avg["elapsed"], avg["response"], avg["ttft"]
    )

    log_run_details(
        args.csv_file, model, "AX630",
        args.num_tokens, avg["deltas"]
    )


# ========================================================
#                   ENTRY POINT
# ========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AX630 Benchmark (single-session)")
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=10001)
    parser.add_argument('--model', type=str, default='qwen2.5-0.5B-prefill-20e')
    parser.add_argument("--num-tokens", type=int, default=1023)
    parser.add_argument("--csv-file", default="benchmark_results.csv")
    parser.add_argument("--device", choices=["ax630"], default="ax630")
    parser.add_argument("--num_runs", type=int, default=1)

    args = parser.parse_args()
    main(args.host, args.port, args.model, args.num_runs)