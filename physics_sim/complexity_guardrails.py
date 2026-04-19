"""Complexity guardrails for physics_sim.

Usage:
    python -m physics_sim.complexity_guardrails --root physics_sim
    python -m physics_sim.complexity_guardrails --root physics_sim --strict
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FILE_LINE_LIMIT = 300
FUNCTION_LINE_LIMIT = 80
NESTING_DEPTH_LIMIT = 4

_NESTING_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)


@dataclass
class FunctionMetric:
    file_path: str
    qualname: str
    line_count: int
    nesting_depth: int


@dataclass
class FileMetric:
    file_path: str
    line_count: int


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("."):
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def _compute_nesting_depth(node: ast.AST) -> int:
    max_depth = 0

    def visit(current: ast.AST, depth: int) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(current):
            next_depth = depth + 1 if isinstance(child, _NESTING_NODES) else depth
            visit(child, next_depth)

    visit(node, 0)
    return max_depth


def _collect_function_metrics(path: Path) -> list[FunctionMetric]:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    metrics: list[FunctionMetric] = []

    def walk(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_lineno = getattr(child, "end_lineno", child.lineno)
                qualname = ".".join([*parents, child.name]) if parents else child.name
                metrics.append(
                    FunctionMetric(
                        file_path=str(path),
                        qualname=qualname,
                        line_count=int(end_lineno - child.lineno + 1),
                        nesting_depth=_compute_nesting_depth(child),
                    )
                )
                walk(child, [*parents, child.name])
            elif isinstance(child, ast.ClassDef):
                walk(child, [*parents, child.name])
            else:
                walk(child, parents)

    walk(module, [])
    return metrics


def collect_metrics(root: Path) -> tuple[list[FileMetric], list[FunctionMetric]]:
    file_metrics: list[FileMetric] = []
    function_metrics: list[FunctionMetric] = []
    for path in _iter_python_files(root):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        file_metrics.append(FileMetric(file_path=str(path), line_count=line_count))
        function_metrics.extend(_collect_function_metrics(path))
    return file_metrics, function_metrics


def build_report(root: Path) -> dict:
    file_metrics, function_metrics = collect_metrics(root)
    file_violations = [
        fm for fm in file_metrics if fm.line_count > FILE_LINE_LIMIT
    ]
    function_length_violations = [
        fm for fm in function_metrics if fm.line_count > FUNCTION_LINE_LIMIT
    ]
    nesting_violations = [
        fm for fm in function_metrics if fm.nesting_depth > NESTING_DEPTH_LIMIT
    ]
    return {
        "root": str(root),
        "thresholds": {
            "file_lines": FILE_LINE_LIMIT,
            "function_lines": FUNCTION_LINE_LIMIT,
            "nesting_depth": NESTING_DEPTH_LIMIT,
        },
        "totals": {
            "files": len(file_metrics),
            "functions": len(function_metrics),
        },
        "violations": {
            "file_lines": [
                {"file": v.file_path, "lines": v.line_count}
                for v in sorted(file_violations, key=lambda x: x.line_count, reverse=True)
            ],
            "function_lines": [
                {
                    "file": v.file_path,
                    "function": v.qualname,
                    "lines": v.line_count,
                }
                for v in sorted(function_length_violations, key=lambda x: x.line_count, reverse=True)
            ],
            "nesting_depth": [
                {
                    "file": v.file_path,
                    "function": v.qualname,
                    "depth": v.nesting_depth,
                }
                for v in sorted(nesting_violations, key=lambda x: x.nesting_depth, reverse=True)
            ],
        },
    }


def _print_report(report: dict) -> None:
    print("Complexity guardrails report")
    print(f"- root: {report['root']}")
    print(
        f"- totals: files={report['totals']['files']}, "
        f"functions={report['totals']['functions']}"
    )
    print(
        "- thresholds: "
        f"file>{report['thresholds']['file_lines']}, "
        f"function>{report['thresholds']['function_lines']}, "
        f"nesting>{report['thresholds']['nesting_depth']}"
    )
    print("- file violations:", len(report["violations"]["file_lines"]))
    print("- function length violations:", len(report["violations"]["function_lines"]))
    print("- nesting violations:", len(report["violations"]["nesting_depth"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Complexity guardrails for physics_sim")
    parser.add_argument("--root", default="physics_sim", help="Root directory to scan")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any violation exists",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = build_report(root)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        _print_report(report)

    has_violations = any(
        report["violations"][key]
        for key in ("file_lines", "function_lines", "nesting_depth")
    )
    if args.strict and has_violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
