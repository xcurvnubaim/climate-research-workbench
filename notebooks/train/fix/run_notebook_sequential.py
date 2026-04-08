import os
import json
import re
import ssl
import sys
import asyncio
import warnings
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
    "sc2_gan.ipynb",
    # "sc1_convnext_mae.ipynb",
    # "sc1_resnet18.ipynb",
    # "sc2_covnext_mae.ipynb",
    # "sc2_resnet18_mae.ipynb",
]

LOOP_PER_NOTEBOOK = 5
KERNEL_SERVER_URL = "https://10.28.76.90"
USE_REMOTE_KERNEL = bool(KERNEL_SERVER_URL)

# Long training cells can run for many hours.
# None disables the per-cell execute-reply timeout.
CELL_TIMEOUT = None
# Keep IOPub wait tolerant for sparse logs (e.g., output every 10+ minutes).
IOPUB_TIMEOUT_SECONDS = 1200

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


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
    notebook_path = Path(notebook_name)
    if not notebook_path.exists():
        print(f"Skip (not found): {notebook_path}")
        continue

    runs_dir = notebook_path.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(1, LOOP_PER_NOTEBOOK + 1):
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

        print(f"Running ({run_idx}/{LOOP_PER_NOTEBOOK}): {notebook_path}")
        run_ok = True
        try:
            client.execute()
        except CellExecutionError as exc:
            run_ok = False
            print(f"CellExecutionError on {notebook_path} run {run_idx}: {exc}")
        except Exception as exc:
            run_ok = False
            print(f"Kernel/transport error on {notebook_path} run {run_idx}: {type(exc).__name__}: {exc}")

        output_path = runs_dir / f"{notebook_path.stem}_run{run_idx}_executed.ipynb"
        with output_path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        if run_ok:
            print(f"Saved: {output_path}")
        else:
            print(f"Saved partial output: {output_path}")
            print("Continuing to next run...")