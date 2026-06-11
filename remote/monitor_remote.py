"""
Monitor AE and/or surrogate training logs on all remote hosts.
Derives the job list and log paths directly from launch_remote.py — no
duplication needed. Displays the last LOG_TAIL lines per job, refreshed
every REFRESH_INTERVAL seconds.

Usage:
    python remote/monitor_remote.py
"""

import getpass
import sys
import time
from pathlib import Path

import paramiko
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent))
from launch_remote import (
    AE_JOB_CONFIGS,
    LOG_TAIL,
    REFRESH_INTERVAL,
    REMOTE_PROJECT_DIR,
    SURROGATE_JOB_CONFIGS,
    USERNAME,
    log_path,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_TASK_COLOR = {"ae": "green", "surrogate": "yellow"}


def fetch_tail(client: paramiko.SSHClient, host: str, task: str) -> str:
    try:
        path = log_path(host, task)
        _, stdout, _ = client.exec_command(
            f"cd {REMOTE_PROJECT_DIR} && tail -n {LOG_TAIL} {path} 2>&1"
        )
        lines = stdout.read().decode().strip().splitlines()
        lines = [l for l in lines if "conda" not in l.lower() and "sqlite" not in l.lower()]
        return "\n".join(lines) if lines else "[dim]no output yet[/dim]"
    except Exception as e:
        return f"[red]error: {e}[/red]"


def make_panel(host: str, task: str, client: paramiko.SSHClient) -> Panel:
    short = host.split(".")[0]
    color = _TASK_COLOR.get(task, "white")
    content = fetch_tail(client, host, task)
    return Panel(
        Text.from_markup(content),
        title=f"[bold cyan]{short}[/bold cyan] [{color}]{task}[/{color}]",
        border_style="bright_black",
        expand=True,
    )


def build_display(jobs: list[tuple[str, str]], clients: dict[str, paramiko.SSHClient]) -> Columns:
    panels = [
        make_panel(host, task, clients[host])
        for host, task in jobs
        if host in clients
    ]
    return Columns(panels, equal=True, expand=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build job list from both configs (same order as launch_remote)
    jobs: list[tuple[str, str]] = []
    for host in AE_JOB_CONFIGS:
        jobs.append((host, "ae"))
    for host in SURROGATE_JOB_CONFIGS:
        jobs.append((host, "surrogate"))

    if not jobs:
        print("Both AE_JOB_CONFIGS and SURROGATE_JOB_CONFIGS are empty — nothing to monitor.")
        raise SystemExit(0)

    unique_hosts = list(dict.fromkeys(host for host, _ in jobs))

    console = Console()
    password = getpass.getpass(f"Password for {USERNAME}: ")
    console.print()

    clients: dict[str, paramiko.SSHClient] = {}
    console.print("[bold]Connecting to hosts...[/bold]")
    for host in unique_hosts:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(host, username=USERNAME, password=password, timeout=10)
            clients[host] = client
            console.print(f"  [green]✓[/green] {host}")
        except Exception as e:
            console.print(f"  [red]✗[/red] {host}: {e}")

    if not clients:
        console.print("[red]No hosts reachable, exiting.[/red]")
        raise SystemExit(1)

    console.print("\n[dim]Press Ctrl+C to stop.[/dim]\n")

    try:
        with Live(build_display(jobs, clients), console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(REFRESH_INTERVAL)
                live.update(build_display(jobs, clients))
    except KeyboardInterrupt:
        pass
    finally:
        for client in clients.values():
            client.close()
