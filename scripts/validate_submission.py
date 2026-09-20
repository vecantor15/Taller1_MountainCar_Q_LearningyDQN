"""Validate the repository against the explicit deliverables of the workshop.

This script does not create any academic evidence. It only checks that the
files required for submission exist and that common packaging mistakes are
absent.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

required = [
    ROOT / "README.md",
    ROOT / "RESULTADOS.md",
    ROOT / "results" / "qlearning_evidence.png",
    ROOT / "results" / "dqn_evidence.png",
    ROOT / "results" / "qlearning_training.csv",
    ROOT / "results" / "dqn_training.csv",
    ROOT / "results" / "summary.json",
    ROOT / "docs" / "esquema_qlearning.png",
    ROOT / "docs" / "esquema_dqn.png",
]

missing = [p.relative_to(ROOT) for p in required if not p.exists()]
readme = (ROOT / "README.md").read_text(encoding="utf-8")
issues = []
if "jgcming-Kiro" in readme:
    issues.append("README todavía contiene una referencia al propietario de la plantilla.")
if "EXERCISE stubs" in readme:
    issues.append("README todavía contiene lenguaje de plantilla incompleta.")

print("Validación de entrega — Taller MountainCar")
print("=" * 44)
if missing:
    print("Archivos faltantes:")
    for p in missing:
        print(f"  - {p}")
else:
    print("Todos los archivos obligatorios están presentes.")

if issues:
    print("\nObservaciones:")
    for issue in issues:
        print(f"  - {issue}")

if missing or issues:
    print("\nEstado: INCOMPLETO")
    sys.exit(1)

print("\nEstado: LISTO PARA REVISIÓN FINAL")
