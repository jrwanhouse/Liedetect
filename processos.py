"""Processos externos dos detectores (Py-Feat, OpenFace)."""
import subprocess
import sys

# Sem janela de console piscando no Windows
OPCOES_POPEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def encerrar(proc):
    """Encerra o processo e os filhos dele (precaucao: no Windows o python.exe de um venv e um
    lancador que abre o Python real como filho; o lancador ja o encerra junto, via job object,
    mas nao dependemos disso)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, **OPCOES_POPEN)
    else:
        proc.kill()
    proc.wait()
