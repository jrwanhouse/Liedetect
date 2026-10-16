"""Ajustes de compatibilidade; importar antes de qualquer outro modulo.

Python 3.15 (PEP 810) + PySide6 6.12: o shiboken6 troca o builtins.__import__ por uma funcao
em C e, depois disso, os imports adiados ("lazy") da biblioteca padrao deixam de ser
resolvidos. O MediaPipe recebe entao o objeto adiado no lugar da classe
(concurrent.futures.ThreadPoolExecutor -> "'lazy_import' object is not callable").
O filtro abaixo faz todo import ser imediato, como ate o Python 3.14.
"""
import sys

if hasattr(sys, "set_lazy_imports_filter"):
    sys.set_lazy_imports_filter(lambda *_: False)
