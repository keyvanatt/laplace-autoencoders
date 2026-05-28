"""
Monitor train_ae logs on all remote hosts.
Displays the last 10 lines per host, refreshed every second.

Usage:
    python scripts/monitor_remote.py
"""

import getpass
import time

import paramiko
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ── Config (must match launch_remote.py) ─────────────────────────────────────

USERNAME = "keyvan.attarian"
REMOTE_PROJECT_DIR = "diffusion_ae"
REFRESH_INTERVAL = 1  # seconds

HOSTS = [
    "baudroie.polytechnique.fr",
    "rouloul.polytechnique.fr",
    "barbeau.polytechnique.fr",
    "kamiche.polytechnique.fr",
    "jabiru.polytechnique.fr",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

LOG_TAIL = 10


def log_path(host: str) -> str:
    slug = host.replace(".", "_")
    return f".log/train_ae_{slug}.log"


def fetch_tail(client: paramiko.SSHClient, host: str) -> str:
    try:
        _, stdout, stderr = client.exec_command(
            f"cd {REMOTE_PROJECT_DIR} && tail -{LOG_TAIL} {log_path(host)} 2>&1"
        )
        lines = stdout.read().decode().strip().splitlines()
        lines = [l for l in lines if "conda" not in l.lower() and "sqlite" not in l.lower()]
        return "\n".join(lines) if lines else "[dim]no output yet[/dim]"
    except Exception as e:
        return f"[red]error: {e}[/red]"


def make_panel(host: str, content: str) -> Panel:
    short = host.split(".")[0]
    return Panel(
        Text.from_markup(content),
        title=f"[bold cyan]{short}[/bold cyan]",
        border_style="bright_black",
        expand=True,
    )


def build_display(clients: dict) -> Columns:
    panels = [make_panel(host, fetch_tail(client, host)) for host, client in clients.items()]
    return Columns(panels, equal=True, expand=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    password = getpass.getpass(f"Password for {USERNAME}: ")

    console = Console()
    clients = {}

    console.print("[bold]Connecting to hosts...[/bold]")
    for host in HOSTS:
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
        with Live(build_display(clients), console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(REFRESH_INTERVAL)
                live.update(build_display(clients))
    except KeyboardInterrupt:
        pass
    finally:
        for client in clients.values():
            client.close()
