"""
List the contents of /Data/KAT/checkpoints on each host in DEFAULT_HOSTS.

Usage:
    python remote/list_checkpoints.py
"""

import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

import paramiko
from rich.console import Console
from rich.rule import Rule

from launch_remote import DEFAULT_HOSTS, USERNAME

CHECKPOINTS_DIR = "/Data/KAT/checkpoints"


def list_checkpoints(host: str, password: str) -> tuple[str, str, str]:
    """Returns (host_short, files_output, error)."""
    short = host.split(".")[0]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=USERNAME, password=password, timeout=15)
    except Exception as e:
        return short, "", f"connect error: {e}"

    try:
        _, stdout, stderr = client.exec_command(f"ls -lth {CHECKPOINTS_DIR} 2>&1")
        stdout.channel.settimeout(15)
        stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        return short, out, err
    finally:
        client.close()


if __name__ == "__main__":
    console = Console()
    console.print()
    console.print(Rule("[bold cyan]checkpoint listing[/bold cyan]"))
    console.print(f"  [dim]Directory: {CHECKPOINTS_DIR}[/dim]")
    console.print(f"  [dim]Hosts: {len(DEFAULT_HOSTS)}[/dim]")
    console.print()

    password = getpass.getpass(f"Password for {USERNAME}: ")
    console.print()

    with ThreadPoolExecutor(max_workers=len(DEFAULT_HOSTS)) as pool:
        futures = {
            pool.submit(list_checkpoints, host, password): host
            for host in DEFAULT_HOSTS
        }
        results = {}
        for future in as_completed(futures):
            short, out, err = future.result()
            results[short] = (out, err)

    for short in sorted(results):
        out, err = results[short]
        console.print(Rule(f"[bold cyan]{short}[/bold cyan]"))
        if err and not out:
            console.print(f"  [red]{err}[/red]")
        elif out:
            console.print(out)
        else:
            console.print("  [dim](empty or no output)[/dim]")
        console.print()
