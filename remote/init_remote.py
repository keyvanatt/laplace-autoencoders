#!/usr/bin/env python3
"""
init_remote.py — Initialise un remote Polytechnique pour le traitement du dataset KAT.

Étapes :
  1. Connexion SSH à <pc>.polytechnique.fr  +  kamiche.polytechnique.fr
  2. Création de /Data/KAT/ sur la cible
  3. Transfert direct kamiche → cible via clé SSH temporaire :
       - génère une paire RSA éphémère
       - injecte la clé publique dans ~/.ssh/authorized_keys sur kamiche
       - écrit la clé privée dans /tmp sur la cible
       - lance scp depuis la cible
       - nettoie les deux machines
  4. Lancement de rotate.py depuis ~/diffusion_ae sur la cible
"""

import io
import sys
import time
import getpass
import threading

import paramiko
from tqdm import tqdm

DOMAIN       = "polytechnique.fr"
KAMICHE      = f"kamiche.{DOMAIN}"
USERNAME     = "keyvan.attarian"
REMOTE_DIR   = "/Data/KAT"
SOURCE_FILE  = f"{REMOTE_DIR}/CH4.npy"
DIFFUSION_AE = "$HOME/diffusion_ae"
PYTHON       = f"{DIFFUSION_AE}/.conda/bin/python"
ROTATE       = f"{DIFFUSION_AE}/src/laplace_surrogate/utils/rotate.py"
TMP_KEY      = "/tmp/.init_remote_tmpkey"
KEY_MARKER   = "init_remote_tmpkey"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ssh_connect(host: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=USERNAME, password=password, timeout=20)
    return client


def run_cmd(ssh: paramiko.SSHClient, cmd: str, label: str | None = None) -> int:
    """Exécute une commande SSH et streame stdout/stderr en temps réel."""
    if label:
        print(f"[→] {label}")
    _, stdout, stderr = ssh.exec_command(cmd)

    def drain(stream, prefix=""):
        for line in stream:
            print(f"{prefix}{line}", end="", flush=True)

    t_err = threading.Thread(target=drain, args=(stderr, "    [err] "), daemon=True)
    t_err.start()
    drain(stdout, "    ")
    rc = stdout.channel.recv_exit_status()
    t_err.join(timeout=2)
    return rc


def transfer_direct(ssh_target: paramiko.SSHClient, ssh_kamiche: paramiko.SSHClient) -> int:
    """
    Transfert direct kamiche → cible sans transit local.
    Utilise une paire RSA éphémère injectée temporairement dans authorized_keys.
    """
    # Génère une clé RSA temporaire
    print("[→] Génération de la clé SSH éphémère...")
    key = paramiko.RSAKey.generate(2048)
    pub = f"ssh-rsa {key.get_base64()} {KEY_MARKER}"

    buf = io.StringIO()
    key.write_private_key(buf)
    private_key_pem = buf.getvalue()

    # Injecte la clé publique sur kamiche
    print(f"[→] Injection de la clé publique dans authorized_keys sur {KAMICHE}...")
    run_cmd(ssh_kamiche,
        f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"echo '{pub}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    )
    print(f"[✓] Clé injectée")

    # Écrit la clé privée sur la cible dans /tmp
    sftp_t = ssh_target.open_sftp()
    with sftp_t.open(TMP_KEY, "w") as f:
        f.write(private_key_pem)
    sftp_t.chmod(TMP_KEY, 0o600)
    sftp_t.close()

    # Taille source pour la barre de progression
    sftp_k = ssh_kamiche.open_sftp()
    total_size = sftp_k.stat(SOURCE_FILE).st_size
    sftp_k.close()

    # Lance scp depuis la cible (non-bloquant)
    scp_cmd = (
        f"scp -i {TMP_KEY} "
        f"-o StrictHostKeyChecking=no "
        f"-o BatchMode=yes "
        f"{USERNAME}@{KAMICHE}:{SOURCE_FILE} {SOURCE_FILE}"
    )
    print(f"[→] Transfert direct {KAMICHE}:{SOURCE_FILE} → cible:{SOURCE_FILE}")
    _, scp_stdout, scp_stderr = ssh_target.exec_command(scp_cmd)

    # Barre de progression : poll la taille du fichier destination toutes les 500ms
    sftp_t = ssh_target.open_sftp()
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="    CH4.npy") as pbar:
        last = 0
        while not scp_stdout.channel.exit_status_ready():
            try:
                current = sftp_t.stat(SOURCE_FILE).st_size
                if current > last:
                    pbar.update(current - last)
                    last = current
            except FileNotFoundError:
                pass
            time.sleep(0.5)
        # Finalise la barre (scp peut écrire le dernier chunk après exit_status)
        try:
            current = sftp_t.stat(SOURCE_FILE).st_size
            if current > last:
                pbar.update(current - last)
        except FileNotFoundError:
            pass
    sftp_t.close()

    rc = scp_stdout.channel.recv_exit_status()
    if rc != 0:
        for line in scp_stderr:
            print(f"    [err] {line}", end="", flush=True)

    # Nettoyage : clé privée sur la cible
    run_cmd(ssh_target, f"rm -f {TMP_KEY}")

    # Nettoyage : clé publique sur kamiche
    run_cmd(ssh_kamiche,
        f"sed -i '/{KEY_MARKER}/d' ~/.ssh/authorized_keys"
    )
    print("[✓] Clé éphémère supprimée des deux machines")

    return rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pc = input("Nom du PC polytechnique (ex: ombrette) : ").strip() or "ombrette"
    target_host = f"{pc}.{DOMAIN}"
    password = getpass.getpass(f"Mot de passe pour {USERNAME} : ")

    # 1. Connexions SSH
    print(f"\n[→] Connexion à {target_host}...")
    ssh = ssh_connect(target_host, password)
    print(f"[✓] Connecté à {target_host}")

    print(f"[→] Connexion à {KAMICHE}...")
    ssh_k = ssh_connect(KAMICHE, password)
    print(f"[✓] Connecté à {KAMICHE}\n")

    # 2. Création /Data/KAT sur la cible
    rc = run_cmd(ssh, f"mkdir -p {REMOTE_DIR}", f"mkdir -p {REMOTE_DIR}")
    if rc != 0:
        print(f"[✗] mkdir échoué (rc={rc})", file=sys.stderr)
        sys.exit(rc)
    print(f"[✓] Dossier {REMOTE_DIR} prêt\n")

    # 3. Transfert direct kamiche → cible
    rc = transfer_direct(ssh, ssh_k)
    ssh_k.close()
    if rc != 0:
        print(f"[✗] scp échoué (rc={rc})", file=sys.stderr)
        sys.exit(rc)
    print(f"[✓] CH4.npy copié vers {target_host}:{SOURCE_FILE}\n")

    # 4. Lancement de rotate.py
    cmd = (
        f"cd {DIFFUSION_AE} && "
        f"PYTHONPATH=src {PYTHON} {ROTATE} "
        f"--input_dir {REMOTE_DIR} "
        f"--output_dir {REMOTE_DIR}"
    )
    print(f"[→] Lancement de rotate.py sur {target_host}")
    print(f"    {cmd}\n")
    rc = run_cmd(ssh, cmd)
    ssh.close()

    if rc == 0:
        print(f"\n[✓] rotate.py terminé avec succès")
    else:
        print(f"\n[✗] rotate.py terminé avec rc={rc}", file=sys.stderr)
        sys.exit(rc)


if __name__ == "__main__":
    main()
