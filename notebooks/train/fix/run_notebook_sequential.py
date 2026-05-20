import os
import json
import re
import ssl
import sys
import asyncio
import warnings
import signal
import threading
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

import requests
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from jupyter_client.kernelspec import KernelSpecManager
from urllib3.exceptions import InsecureRequestWarning


notebooks = [
    # "sc2_unet_mae.ipynb",
    # "sc2_gan.ipynb",
    # "sc1_convnext_mae.ipynb",
    # "sc1_resnet18.ipynb",
    # "sc2_covnext_mae.ipynb",
    # "sc2_resnet18_mae.ipynb",
    # "sc1_unet_preupsample.ipynb",
    # "sc2_unet_mae_preupsample.ipynb",
    # "sc1_convnext_mae_pixel.ipynb"
    # "sc1_unet.ipynb",
    # "sc2_unet_mae_preupsample.ipynb",
    # "sc1_resnet18_preupsample.ipynb",
    # "sc2_convnext_mae_preupsample.ipynb",
    # "sc2_resnet18_mae_preupsample.ipynb",
    # "sc1_gan.ipynb",
    # "sc1_convnext_mae_preupsample.ipynb"
    "sc1_gan_preupsample.ipynb",
    "sc2_gan_preupsample.ipynb",
]

LOOP_PER_NOTEBOOK = 2
KERNEL_SERVER_URL = "https://server-ta.zmarzuqi.dev"
USE_REMOTE_KERNEL = bool(KERNEL_SERVER_URL)

# Long training cells can run for many hours.
# None disables the per-cell execute-reply timeout.
CELL_TIMEOUT = None
# Keep IOPub wait tolerant for sparse logs (e.g., output every 10+ minutes).
IOPUB_TIMEOUT_SECONDS = 1200
# Periodically save in-flight notebook output so progress is visible on disk.
SAVE_INTERVAL_SECONDS = 60

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


shutdown_requested = threading.Event()
_shutdown_signal_count = 0


def _request_shutdown(signum, _frame):
    global _shutdown_signal_count
    _shutdown_signal_count += 1
    signal_name = signal.Signals(signum).name

    if _shutdown_signal_count == 1:
        print(f"\nReceived {signal_name}. Graceful shutdown requested; finishing current step...")
        shutdown_requested.set()
        return

    print(f"\nReceived {signal_name} again. Exiting immediately.")
    raise SystemExit(130)


signal.signal(signal.SIGINT, _request_shutdown)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _request_shutdown)


# ── Helpers ───────────────────────────────────────────────────────────────────

def yes_no_prompt(prompt: str, default_no: bool = True) -> bool:
    raw = input(prompt).strip().lower()
    if not raw:
        return not default_no
    return raw in {"y", "yes", "1", "true"}


def choose_kernel_interactive(kernel_names):
    if not kernel_names:
        return "python3"
    print("Available kernels:")
    for idx, name in enumerate(kernel_names, start=1):
        print(f"  {idx}. {name}")
    while True:
        raw = input(f"Choose kernel [1-{len(kernel_names)}] (default 1): ").strip()
        if not raw:
            return kernel_names[0]
        if raw.isdigit():
            pos = int(raw)
            if 1 <= pos <= len(kernel_names):
                return kernel_names[pos - 1]
        print("Invalid choice. Please enter a valid number.")


def patch_notebook_workdir(nb_json, notebook_dir: Path):
    replacement = f'os.chdir(r"{str(notebook_dir.resolve())}")'
    pattern = r"os\.chdir\((['\"])\/home\/jovyan\/work\1\)"
    for cell in nb_json.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        updated = re.sub(pattern, lambda _m: replacement, src)
        if updated != src:
            cell["source"] = updated


def fetch_local_kernel_names():
    ksm = KernelSpecManager()
    return sorted(ksm.find_kernel_specs().keys())


def write_notebook_atomic(nb_json, output_path: Path):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb_json, f)
    tmp_path.replace(output_path)


def write_progress_heartbeat(progress_path: Path, notebook_path: Path, run_idx: int, total_runs: int, nb_json):
    executed_cells = 0
    total_code_cells = 0
    for cell in nb_json.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        total_code_cells += 1
        if cell.get("execution_count") is not None:
            executed_cells += 1

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": str(notebook_path),
        "run_index": run_idx,
        "total_runs": total_runs,
        "executed_code_cells": executed_cells,
        "total_code_cells": total_code_cells,
        "shutdown_requested": shutdown_requested.is_set(),
    }
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── Auth + Session Setup ──────────────────────────────────────────────────────

ssl_ctx = ssl.create_default_context()
verify_ssl = yes_no_prompt("Verify SSL certificate? [Y/n]: ", default_no=False)

session = requests.Session()
session.verify = verify_ssl
if not verify_ssl:
    warnings.simplefilter("ignore", InsecureRequestWarning)

hub_user = input("JupyterHub username: ").strip()
jupyterhub_token = getpass("JupyterHub API token: ").strip()

gateway_url = f"{KERNEL_SERVER_URL.rstrip('/')}/user/{hub_user}"

os.environ["JUPYTER_GATEWAY_URL"] = gateway_url
os.environ["JUPYTER_GATEWAY_TOKEN"] = jupyterhub_token
os.environ["JUPYTER_GATEWAY_AUTH_TOKEN"] = jupyterhub_token

print(f"Gateway URL: {gateway_url}")

# ── Initialize session (IMPORTANT) ────────────────────────────────────────────

print("Initializing session via /lab ...")

session.headers.update({
    "Authorization": f"token {jupyterhub_token}"
})

resp = session.get(f"{gateway_url}/lab")
resp.raise_for_status()

xsrf_token = session.cookies.get("_xsrf", "")
if not xsrf_token:
    raise RuntimeError("Failed to obtain _xsrf cookie")

print(f"XSRF token obtained: {xsrf_token[:8]}...")

session.headers.update({
    "X-XSRFToken": xsrf_token
})


# ── Patch gateway_request ─────────────────────────────────────────────────────

from jupyter_server.gateway import gateway_client as _gwmod
from jupyter_server.gateway import managers as _mgmod

async def _patched_gateway_request(endpoint, **kwargs):
    method = kwargs.get("method", "GET").upper()
    data = kwargs.get("body", None)
    headers = kwargs.get("headers", {}) or {}

    url = endpoint if endpoint.startswith("http") else gateway_url + endpoint
    req_headers = {**session.headers, **headers}

    if method == "GET":
        r = session.get(url, headers=req_headers)
    elif method == "POST":
        r = session.post(url, headers=req_headers, data=data)
    elif method == "DELETE":
        r = session.delete(url, headers=req_headers)
    else:
        r = session.request(method, url, headers=req_headers, data=data)

    r.raise_for_status()

    class DummyResponse:
        def __init__(self, r):
            self.code = r.status_code
            self.body = r.content
            self.reason = r.reason

    return DummyResponse(r)

_gwmod.gateway_request = _patched_gateway_request
_mgmod.gateway_request = _patched_gateway_request

print("gateway_request patched with session")

import websocket
import ssl

websocket.setdefaulttimeout(60)

# disable SSL verification globally for websocket
sslopt = {"cert_reqs": ssl.CERT_NONE}

_original_create_connection = websocket.create_connection

def _patched_create_connection(*args, **kwargs):
    # 1. Apply SSL options
    kwargs["sslopt"] = sslopt
    
    # 2. Inject Authorization and XSRF headers from the session
    ws_headers = kwargs.get("header", [])
    if isinstance(ws_headers, dict):
        ws_headers = [f"{k}: {v}" for k, v in ws_headers.items()]
    else:
        ws_headers = list(ws_headers)
        
    if "Authorization" in session.headers:
        ws_headers.append(f"Authorization: {session.headers['Authorization']}")
    if "X-XSRFToken" in session.headers:
        ws_headers.append(f"X-XSRFToken: {session.headers['X-XSRFToken']}")
        
    kwargs["header"] = ws_headers

    # 3. Inject cookies from the session
    cookie_string = "; ".join([f"{c.name}={c.value}" for c in session.cookies])
    if cookie_string:
        existing_cookie = kwargs.get("cookie", "")
        kwargs["cookie"] = f"{existing_cookie}; {cookie_string}".strip("; ")

    return _original_create_connection(*args, **kwargs)

websocket.create_connection = _patched_create_connection


# ── Fetch kernels ─────────────────────────────────────────────────────────────

def fetch_gateway_kernel_names_session():
    url = f"{gateway_url}/api/kernelspecs"
    r = session.get(url)
    r.raise_for_status()
    payload = r.json()
    return sorted(payload.get("kernelspecs", {}).keys())


kernel_candidates = []
try:
    kernel_candidates = fetch_gateway_kernel_names_session()
except Exception as exc:
    print(f"Could not fetch kernels from gateway: {exc}")

if not kernel_candidates:
    use_local = yes_no_prompt("No remote kernels found. Fallback to local kernels? [y/N]: ")
    if use_local:
        kernel_candidates = fetch_local_kernel_names()
    else:
        raise SystemExit("Stop: no remote kernels available.")

KERNEL_NAME = choose_kernel_interactive(kernel_candidates)
print(f"Selected kernel: {KERNEL_NAME}")


# ── Run notebooks ─────────────────────────────────────────────────────────────

for notebook_name in notebooks:
    if shutdown_requested.is_set():
        print("Shutdown requested. Stopping before next notebook.")
        break

    notebook_path = Path(notebook_name)
    if not notebook_path.exists():
        print(f"Skip (not found): {notebook_path}")
        continue

    runs_dir = notebook_path.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(3, LOOP_PER_NOTEBOOK + 3):
        if shutdown_requested.is_set():
            print("Shutdown requested. Stopping before next run.")
            break

        with notebook_path.open("r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # patch_notebook_workdir(nb, notebook_path.parent)

        client = NotebookClient(
            nb,
            kernel_name=KERNEL_NAME,
            timeout=CELL_TIMEOUT,
            iopub_timeout=IOPUB_TIMEOUT_SECONDS,
            raise_on_iopub_timeout=False,
            resources={"metadata": {"path": str(notebook_path.parent.resolve())}},
            kernel_manager_class=(
                "jupyter_server.gateway.managers.GatewayKernelManager"
                if USE_REMOTE_KERNEL
                else "jupyter_client.manager.AsyncKernelManager"
            ),
        )

        output_path = runs_dir / f"{notebook_path.stem}_run{run_idx}_executed.ipynb"
        progress_path = runs_dir / f"{notebook_path.stem}_run{run_idx}_progress.json"
        progress_stop_event = threading.Event()

        def _autosave_loop():
            while not progress_stop_event.wait(SAVE_INTERVAL_SECONDS):
                try:
                    write_notebook_atomic(nb, output_path)
                    write_progress_heartbeat(progress_path, notebook_path, run_idx, LOOP_PER_NOTEBOOK, nb)
                    print(f"Checkpoint saved: {output_path.name}")
                except Exception as exc:
                    print(f"Checkpoint save failed for {output_path}: {type(exc).__name__}: {exc}")

        progress_thread = threading.Thread(target=_autosave_loop, name=f"autosave-{notebook_path.stem}-run{run_idx}", daemon=True)
        progress_thread.start()

        print(f"Running ({run_idx}/{LOOP_PER_NOTEBOOK}): {notebook_path}")
        run_ok = True
        interrupted = False
        try:
            client.execute()
        except KeyboardInterrupt:
            interrupted = True
            run_ok = False
            shutdown_requested.set()
            print(f"KeyboardInterrupt on {notebook_path} run {run_idx}. Requesting graceful shutdown...")
        except CellExecutionError as exc:
            run_ok = False
            print(f"CellExecutionError on {notebook_path} run {run_idx}: {exc}")
        except Exception as exc:
            run_ok = False
            print(f"Kernel/transport error on {notebook_path} run {run_idx}: {type(exc).__name__}: {exc}")
        finally:
            progress_stop_event.set()
            progress_thread.join(timeout=5)
            try:
                client.shutdown_kernel()
            except Exception:
                pass

        write_notebook_atomic(nb, output_path)
        write_progress_heartbeat(progress_path, notebook_path, run_idx, LOOP_PER_NOTEBOOK, nb)

        if run_ok:
            print(f"Saved: {output_path}")
        else:
            print(f"Saved partial output: {output_path}")
            print("Continuing to next run...")

        if interrupted or shutdown_requested.is_set():
            print("Graceful shutdown complete for current run.")
            break