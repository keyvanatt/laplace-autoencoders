"""
Launch train_ae.py, train_surrogate.py, and/or train_svd.py on multiple remote hosts via SSH.

Usage:
    Edit USERNAME, REMOTE_PROJECT_DIR, AE_JOB_CONFIGS, SURROGATE_JOB_CONFIGS,
    and SVD_JOB_CONFIGS below (leave any dict empty to skip that training type), then run:
        python remote/launch_remote.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AE JOB  —  scripts/train_ae.py  (model=slae|llae  training=ae)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  model                   slae | llae | lslae
  training                ae  (configs/training/ae.yaml)

  model.latent_dim        latent dimension               default 64
  model.K                 number of Laplace frequencies  default 16
  model.gamma_init        Bromwich contour damping γ     default 0.0

  [tag: ol]  model.optimal_laplace=true
             model.optimal_laplace_path=checkpoints/laplace_opti_K16.pt
                 → use pre-optimised poles from laplace_opti.py

  [tag: ll]  model.learnable_laplace=true   (LLAE/LSLAE only)
                 → learn poles s_k jointly during AE training

  model.beta              ridge on latent z              default 1e-2
  model.beta_latent       Laplace roundtrip MSE (LLAE)   default 0.1
  model.freq_L            FiLM conditioning levels (SLAE) default 8
  model.time_L            FiLM conditioning levels (LLAE) default 8

  training.epochs         default 300
  training.lr             default 5e-4
  training.batch_size     default 16
  training.ckpt_path      resume from checkpoint         default null

 !! For SLAE, Set batch size to 256. For LLAE, Set batch size to 16

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SURROGATE JOB  —  scripts/trainogate.py
                  (model=slae|llae|lslae  training=surrogate_slae|surrogate_llae|surrogate_lslae)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  model                   slae | llae | lslae
  training                surrogate_slae | surrogate_llae | surrogate_lslae

  training.ae_ckpt        path to frozen AE checkpoint   (required)
  training.ckpt_path      resume surrogate training       default null

  [tag: ll]  model.learnable_laplace=true
                 → also fine-tunes poles, uses training.lr_laplace

  Surrogate MLP arch (configs/training/surrogate_arch.yaml):
  training.hidden_dim     trunk hidden dim               default 512
  training.head_dim       per-freq head dim              default 256
  training.n_trunk        trunk depth                    default 4
  training.n_head         head depth                     default 2

  Loss weights:
  training.alpha_lat      latent supervision             default 1.0
  training.alpha_spat     spatial (physical-time) recon  default 1.0
  training.alpha_t        Tikhonov smoothness            default 7e-3
  training.lam            Tikhonov ridge                 default 3e-5

  LR:
  training.lrogate   surrogate MLP                  default 3e-4
  training.lr_decoder     frozen decoder fine-tune        default 5e-5
  training.lr_laplace     Laplace poles (ll tag only)    default 1e-5
  training.epochs         default 300
  training.batch_size     default 16

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SVD JOB  —  scripts/train_svd.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --tag   config tag to train, e.g. K16_ksvd64_g0.010
          (must match a key in checkpoints/svd_bases.pt)
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
    "ombrette.polytechnique.fr",
    "perdrix.polytechnique.fr",
    "quetzal.polytechnique.fr",
    "quiscale.polytechnique.fr",
    "sitelle.polytechnique.fr",
    "epervier.polytechnique.fr",
    "dindon.polytechnique.fr",
    "bengali.polytechnique.fr",
    "coucou.polytechnique.fr",
]

# One config dict per host — Hydra overrides for train_ae.py
# Leave empty to skip AE training.
AE_JOB_CONFIGS: dict[str, dict] = {}

# One config dict per host — Hydra overrides for train_surrogate.py
# Leave empty to skip surrogate training.
AE_CKPT_DIR = "checkpoints"
CKPT_DIR = "/Data/KAT/checkpoints"
SURROGATE_JOB_CONFIGS: dict[str, dict] = {}

# One config dict per host — argparse overrides for train_svd.py
# Each dict must contain exactly one key: "tag" (the SVD config tag to train).
# Hosts rouloul and gymnote are reserved for surrogate jobs above.
SVD_JOB_CONFIGS: dict[str, dict] = {

    "quetzal.polytechnique.fr":   {"tag": "K32_ksvd64_g0.010"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

PYTHON = ".conda/bin/python"

_TASK_META = {
    "ae":        {"script": "scripts/train_ae.py",        "log_prefix": "ae",        "session_prefix": "ae",   "arg_style": "hydra"},
    "surrogate": {"script": "scripts/train_surrogate.py", "log_prefix": "surrogate", "session_prefix": "surr", "arg_style": "hydra"},
    "svd":       {"script": "scripts/train_svd.py",       "log_prefix": "svd",       "session_prefix": "svd",  "arg_style": "argparse"},
}


def log_path(host: str, task: str) -> str:
    prefix = _TASK_META[task]["log_prefix"]
    return f".log/{prefix}_{host.split('.')[0]}.log"


def session_name(host: str, task: str) -> str:
    prefix = _TASK_META[task]["session_prefix"]
    return f"{prefix}_{host.split('.')[0]}"


def build_command(host: str, task: str, overrides: dict) -> str:
    script = _TASK_META[task]["script"]
    arg_style = _TASK_META[task]["arg_style"]
    if arg_style == "argparse":
        override_str = " ".join(
            f"--{k} {v}" for k, v in overrides.items()
        )
    else:
        override_str = " ".join(
            f"{k}={v}" if v is not None else str(k)
            for k, v in overrides.items()
        )
    cmd = f"{PYTHON} {script} {override_str}".strip()
    log = log_path(host, task)
    session = session_name(host, task)
    inner = f"{cmd} 2>&1 | tee {log}; exec bash"
    return (
        f"mkdir -p .log && "
        f"tmux kill-session -t {session} 2>/dev/null; "
        f"tmux new-session -d -s {session} '{inner}' && "
        f"echo 'SESSION: {session}'"
    )


def launch_on_host(host: str, task: str, overrides: dict, password: str) -> tuple[dict, paramiko.SSHClient | None]:
    short = host.split(".")[0]
    result = {"host": short, "task": task, "status": "", "session": ""}

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=USERNAME, password=password, timeout=15)
    except Exception as e:
        result["status"] = "[red]✗ connect[/red]"
        result["session"] = str(e)
        return result, None

    _, stdout, stderr = client.exec_command(
        f"cd {REMOTE_PROJECT_DIR} && {build_command(host, task, overrides)}"
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
    console.print(Rule("[bold cyan]remote launcher[/bold cyan]"))
    console.print()

    # Build the flat list of (host, task, overrides) jobs from all dicts
    all_jobs: list[tuple[str, str, dict]] = []
    for host, overrides in AE_JOB_CONFIGS.items():
        all_jobs.append((host, "ae", overrides))
    for host, overrides in SURROGATE_JOB_CONFIGS.items():
        all_jobs.append((host, "surrogate", overrides))
    for host, overrides in SVD_JOB_CONFIGS.items():
        all_jobs.append((host, "svd", overrides))

    if not all_jobs:
        console.print("[yellow]AE_JOB_CONFIGS, SURROGATE_JOB_CONFIGS and SVD_JOB_CONFIGS are all empty — nothing to launch.[/yellow]")
        raise SystemExit(0)

    ae_count = len(AE_JOB_CONFIGS)
    surr_count = len(SURROGATE_JOB_CONFIGS)
    svd_count = len(SVD_JOB_CONFIGS)
    console.print(f"  [dim]AE jobs: {ae_count}  |  Surrogate jobs: {surr_count}  |  SVD jobs: {svd_count}[/dim]")
    console.print()

    password = getpass.getpass(f"Password for {USERNAME}: ")
    console.print()

    results = []
    clients: dict[str, paramiko.SSHClient] = {}

    console.print(f"  [dim]Launching {len(all_jobs)} jobs in parallel...[/dim]")
    with ThreadPoolExecutor(max_workers=len(all_jobs)) as pool:
        futures = {
            pool.submit(launch_on_host, host, task, overrides, password): (host, task)
            for host, task, overrides in all_jobs
        }
        for future in as_completed(futures):
            r, client = future.result()
            console.print(
                f"  {r['status']} [bold]{r['host']}[/bold]"
                f"  [dim]{r['task']}[/dim]"
                f"  [dim]tmux: {r['session']}[/dim]"
            )
            results.append(r)
            if client is not None:
                clients[f"{r['host']}_{r['task']}"] = client

    console.print()

    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("Host", style="cyan")
    table.add_column("Task", style="blue")
    table.add_column("Status")
    table.add_column("tmux session", style="yellow")
    for r in results:
        table.add_row(r["host"], r["task"], r["status"], r["session"])
    console.print(table)

    for client in clients.values():
        client.close()
    console.print("\n[dim]Jobs launched. Use monitor_remote.py to follow logs.[/dim]")
