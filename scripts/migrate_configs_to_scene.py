"""Codemod: migrate ``experiments/*.py`` from flat ``SimConfig.objects`` to
the new scene-graph form ``SimConfig.scene=SceneConfig(parts=..., constraints=...)``.

Mapping rules (matches the `SimConfig.as_scene()` upgrade in models.py):

- ``ObjectConfig(role="collider_only", ...)`` → ``PartConfig(...)`` (no role arg)
  plus ``CollideOnly(part=name)`` appended to ``constraints``.
- ``ObjectConfig(role="render_only", ...)`` → ``PartConfig(...)`` (no role arg).
  (render_only was already encoded as "material is None" in most configs.)
- ``ObjectConfig(role="dynamic", ...)`` or no role → ``PartConfig(...)``.

Usage::

    python scripts/migrate_configs_to_scene.py experiments/
    python scripts/migrate_configs_to_scene.py --dry-run experiments/

For each ``SimConfig(..., objects=[...])`` call it rewrites:

    from physics_sim.config.models import (..., ObjectConfig, ...)
to:
    from physics_sim.config.models import (..., ...)   # ObjectConfig removed
    from physics_sim.config.scene import PartConfig, SceneConfig, CollideOnly

and::

    SimConfig(
        ...,
        objects=[
            ObjectConfig(name="a", ...),
            ObjectConfig(name="pot", role="collider_only", ...),
        ],
    )
becomes::

    SimConfig(
        ...,
        scene=SceneConfig(
            parts=[
                PartConfig(name="a", ...),
                PartConfig(name="pot", ...),
            ],
            constraints=[
                CollideOnly(part="pot"),
            ],
        ),
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import libcst as cst
from libcst import matchers as m


# ── Helpers ─────────────────────────────────────────────────────────

def _string_value(node: cst.BaseExpression) -> str | None:
    """If node is a SimpleString, return the unquoted value."""
    if isinstance(node, cst.SimpleString):
        return node.evaluated_value
    return None


def _is_objectconfig_call(call: cst.Call) -> bool:
    return m.matches(call.func, m.Name("ObjectConfig"))


def _pop_role(call: cst.Call) -> tuple[cst.Call, str | None]:
    """Return (new_call, role_value_or_None) with the ``role=`` kwarg removed."""
    new_args: list[cst.Arg] = []
    role: str | None = None
    for arg in call.args:
        if arg.keyword is not None and arg.keyword.value == "role":
            role = _string_value(arg.value)
            continue
        new_args.append(arg)
    # Clean trailing comma semantics after removal.
    if new_args:
        new_args = [
            a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in new_args[:-1]
        ] + [new_args[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)]
    return call.with_changes(args=new_args), role


def _object_name(call: cst.Call) -> str | None:
    for arg in call.args:
        if arg.keyword is not None and arg.keyword.value == "name":
            return _string_value(arg.value)
    return None


# ── Transformer ─────────────────────────────────────────────────────

class _Migrator(cst.CSTTransformer):
    def __init__(self) -> None:
        self.changed = False

    # Rewrite every ``SimConfig(..., objects=[...])`` call.
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        if not m.matches(updated_node.func, m.Name("SimConfig")):
            return updated_node

        new_args: list[cst.Arg] = []
        objects_arg: cst.Arg | None = None
        for arg in updated_node.args:
            if arg.keyword is not None and arg.keyword.value == "objects":
                objects_arg = arg
            else:
                new_args.append(arg)

        if objects_arg is None:
            return updated_node
        list_node = objects_arg.value
        if not isinstance(list_node, cst.List):
            return updated_node

        self.changed = True
        new_parts: list[cst.Element] = []
        new_constraints: list[cst.Element] = []

        for element in list_node.elements:
            if not isinstance(element.value, cst.Call) or not _is_objectconfig_call(element.value):
                # Leave unchanged (expression we don't recognize)
                new_parts.append(element)
                continue
            call = element.value
            name = _object_name(call)
            call_no_role, role = _pop_role(call)
            # Rename ObjectConfig -> PartConfig
            call_as_part = call_no_role.with_changes(func=cst.Name("PartConfig"))
            new_parts.append(cst.Element(value=call_as_part))
            if role == "collider_only" and name is not None:
                new_constraints.append(cst.Element(
                    value=cst.Call(
                        func=cst.Name("CollideOnly"),
                        args=[cst.Arg(
                            keyword=cst.Name("part"),
                            value=cst.SimpleString(f'"{name}"'),
                        )],
                    ),
                ))
            # role="render_only" / "dynamic" / absent → no constraint

        # Build SceneConfig(parts=[...], constraints=[...])
        parts_list = cst.List(elements=new_parts)
        scene_args = [cst.Arg(keyword=cst.Name("parts"), value=parts_list)]
        if new_constraints:
            scene_args.append(cst.Arg(
                keyword=cst.Name("constraints"),
                value=cst.List(elements=new_constraints),
            ))
        scene_call = cst.Call(func=cst.Name("SceneConfig"), args=scene_args)

        new_args.append(cst.Arg(keyword=cst.Name("scene"), value=scene_call))

        return updated_node.with_changes(args=new_args)


# ── Import rewrite ──────────────────────────────────────────────────

_SCENE_IMPORTS_NEEDED = {"PartConfig", "SceneConfig", "CollideOnly"}


class _ImportFixer(cst.CSTTransformer):
    """Remove ObjectConfig from models imports; add scene imports if needed."""

    def __init__(self, *, needs_scene_import: bool, needs_collide_only: bool) -> None:
        self.needs_scene_import = needs_scene_import
        self.needs_collide_only = needs_collide_only
        self.scene_import_inserted = False

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom,
    ) -> cst.BaseSmallStatement:
        # Match `from physics_sim.config.models import (...)`
        if not m.matches(
            updated_node.module,
            m.Attribute(
                value=m.Attribute(
                    value=m.Name("physics_sim"),
                    attr=m.Name("config"),
                ),
                attr=m.Name("models"),
            ),
        ):
            return updated_node

        # Remove ObjectConfig from names
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node
        kept: list[cst.ImportAlias] = []
        for alias in updated_node.names:
            nm = alias.name.value if isinstance(alias.name, cst.Name) else ""
            if nm == "ObjectConfig":
                continue
            kept.append(alias)
        if not kept:
            # fall back to keeping at least SimConfig to avoid empty imports
            return updated_node
        # Drop trailing comma on last kept alias
        kept = [a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in kept[:-1]] + [
            kept[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        ]
        return updated_node.with_changes(names=kept)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self.needs_scene_import:
            return updated_node
        # Build `from physics_sim.config.scene import PartConfig, SceneConfig[, CollideOnly]`
        names = [cst.ImportAlias(name=cst.Name("PartConfig")),
                 cst.ImportAlias(name=cst.Name("SceneConfig"))]
        if self.needs_collide_only:
            names.append(cst.ImportAlias(name=cst.Name("CollideOnly")))
        names = [n.with_changes(comma=cst.MaybeSentinel.DEFAULT) for n in names[:-1]] + [
            names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)
        ]
        new_import = cst.SimpleStatementLine(body=[cst.ImportFrom(
            module=cst.Attribute(
                value=cst.Attribute(
                    value=cst.Name("physics_sim"),
                    attr=cst.Name("config"),
                ),
                attr=cst.Name("scene"),
            ),
            names=names,
        )])
        # Insert after the last existing ImportFrom/Import statement.
        body = list(updated_node.body)
        insert_at = 0
        for i, stmt in enumerate(body):
            if isinstance(stmt, cst.SimpleStatementLine) and any(
                isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body
            ):
                insert_at = i + 1
        body.insert(insert_at, new_import)
        return updated_node.with_changes(body=body)


# ── Driver ──────────────────────────────────────────────────────────

def migrate_file(path: Path) -> tuple[bool, str]:
    """Return (changed, new_source)."""
    src = path.read_text(encoding="utf-8")
    tree = cst.parse_module(src)
    mig = _Migrator()
    tree2 = tree.visit(mig)
    if not mig.changed:
        return False, src

    # Also fix imports.  Detect whether CollideOnly is needed by scanning
    # the rewritten source for its name.
    new_src_mid = tree2.code
    needs_collide_only = "CollideOnly(" in new_src_mid
    tree3 = cst.parse_module(new_src_mid)
    imp_fixer = _ImportFixer(
        needs_scene_import=True, needs_collide_only=needs_collide_only,
    )
    tree4 = tree3.visit(imp_fixer)
    return True, tree4.code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+",
                        help="Files or directories to migrate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print diff instead of writing")
    args = parser.parse_args()

    targets: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            targets.extend(sorted(path.rglob("*.py")))
        else:
            targets.append(path)

    any_changed = False
    for path in targets:
        # Skip __init__ / dunder / tests
        if path.name.startswith("__"):
            continue
        try:
            changed, new_src = migrate_file(path)
        except Exception as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        if not changed:
            continue
        any_changed = True
        if args.dry_run:
            print(f"=== would rewrite {path} ===")
            print(new_src)
        else:
            path.write_text(new_src, encoding="utf-8")
            print(f"[rewrote] {path}")

    return 0 if any_changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
