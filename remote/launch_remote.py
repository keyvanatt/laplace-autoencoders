"""
Launch train_ae.py on multiple remote hosts via SSH, then live-monitor logs.

Usage:
    Edit USERNAME, REMOTE_PROJECT_DIR, and JOB_CONFIGS below, then run:
        python remote/launch_remote.py
"""

import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

import paramiko
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

# ── Configuration ─────────────────────────────────────────────────────────────

USERNAME = "keyvan.attarian"
REMOTE_PROJECT_DIR = "diffusion_ae"
LOG_TAIL = 10
REFRESH_INTERVAL = 1  # seconds

DEFAULT_HOSTS = [
    "baudroie.polytechnique.fr",
    "rouloul.polytechnique.fr",
    "gymnote.polytechnique.fr",
    "kamiche.polytechnique.fr",
    "jabiru.polytechnique.fr",
    "barbeau.polytechnique.fr",
]

# One config dict per host — Hydra overrides for train_ae.py
# model=slae|llae selects configs/model/slae.yaml or llae.yaml
JOB_CONFIGS = {
    "rouloul.polytechnique.fr": {
        "model": "llae",
        "model.gamma_init": 0.0,
        "training.batch_size": 16,
        "+training.ckpt_path": "/Data/KAT/checkpoints/llae_ld64_K16_g0.0.ckpt",
    },
    "kamiche.polytechnique.fr": {
        "model": "llae",
        "model.gamma_init": 0.01,
        "training.batch_size": 16,
        "+training.ckpt_path": "/Data/KAT/checkpoints/llae_ld64_K16_g0.01.ckpt",
    },
    "jabiru.polytechnique.fr": {
        "model": "llae",
        "model.gamma_init": 0.0,
        "training.batch_size": 16,
        "model.learnable_laplace": True,
        "+training.ckpt_path": "/Data/KAT/checkpoints/llae_ld64_K16_g0.0_ll.ckpt",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

PYTHON = ".conda/bin/python"
SCRIPT = "scripts/train_ae.py"


def log_path(host: str) -> str:
    return f".log/train_ae_{host.replace('.', '_')}.log"


def session_name(host: str) -> str:
    return f"train_{host.split('.')[0]}"


def build_command(host: str) -> str:
    overrides = JOB_CONFIGS.get(host, {})
    override_str = " ".join(
        f"{k}={v}" if v is not None else str(k)
        for k, v in overrides.items()
    )
    cmd = f"{PYTHON} {SCRIPT} {override_str}".strip()
    log = log_path(host)
    session = session_name(host)
    # Run inside a named tmux session so `tmux attach -t <session>` shows the
    # full output even after a crash. `exec bash` keeps the pane open.
    inner = f"{cmd} 2>&1 | tee {log}; exec bash"
    return (
        f"mkdir -p .log && "
        f"tmux kill-session -t {session} 2>/dev/null; "
        f"tmux new-session -d -s {session} '{inner}' && "
        f"echo 'SESSION: {session}'"
    )


def launch_on_host(host: str, password: str) -> tuple[dict, paramiko.SSHClient | None]:
    short = host.split(".")[0]
    result = {"host": short, "status": "", "session": ""}

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=USERNAME, password=password, timeout=15)
    except Exception as e:
        result["status"] = "[red]✗ connect[/red]"
        result["session"] = str(e)
        return result, None

    _, stdout, stderr = client.exec_command(
        f"cd {REMOTE_PROJECT_DIR} && {build_command(host)}"
    )
    stdout.channel.settimeout(15)
    try:
        stdout.channel.recv_exit_status()
    except Exception:
        pass
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()

    if "SESSION:" in out:
        result["status"] = "[green]✓ launched[/green]"
        result["session"] = out.split("SESSION:")[-1].strip()
    else:
        result["status"] = "[yellow]? unknown[/yellow]"
        result["session"] = err or out

    return result, client


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console = Console()
    console.print()
    console.print(Rule("[bold cyan]train_ae remote launcher[/bold cyan]"))
    console.print()

    password = getpass.getpass(f"Password for {USERNAME}: ")
    console.print()

    results = []
    clients = {}

    console.print(f"  [dim]Launching {len(JOB_CONFIGS)} jobs in parallel...[/dim]")
    with ThreadPoolExecutor(max_workers=len(JOB_CONFIGS)) as pool:
        futures = {pool.submit(launch_on_host, host, password): host for host in JOB_CONFIGS}
        for future in as_completed(futures):
            host = futures[future]
            r, client = future.result()
            console.print(f"  {r['status']} [bold]{r['host']}[/bold]  [dim]tmux: {r['session']}[/dim]")
            results.append(r)
            if client is not None:
                clients[host] = client

    console.print()

    # Summary table
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("Host", style="cyan")
    table.add_column("Status")
    table.add_column("tmux session", style="yellow")
    for r in results:
        table.add_row(r["host"], r["status"], r["session"])
    console.print(table)

    for client in clients.values():
        client.close()
    console.print("\n[dim]Jobs launched. Use monitor_remote.py to follow logs.[/dim]")
