from __future__ import annotations

import ast
from pathlib import Path

from .models import AlgorithmInspection

DISCARD_MISSING_IMPLEMENTATION = "missing_implementation"
DISCARD_INVALID_STUB = "invalid_stub"
DISCARD_UNRESOLVED_REDIRECT = "unresolved_redirect"
DISCARD_PLACEHOLDER_MODULE = "placeholder_module"

_REDIRECT_MARKERS = (
    "moved to",
    "moved into",
    "implementation moved",
    "see ",
    "redirect",
    "use ",
)
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "todo",
    "stub",
    "not implemented",
    "coming soon",
)


def discover_algorithm_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__":
            dirs.append(p)
    return dirs


def _parse_functions(py_file: Path) -> list[str]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out.append(node.name)
    return out


def _parse_tree(py_file: Path) -> ast.Module | None:
    try:
        return ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _infer_entrypoint(py_files: list[Path]) -> Path | None:
    if not py_files:
        return None
    for p in py_files:
        if p.name == "dga.py":
            return p
    preferred = [
        "main.py",
        "generator.py",
        "dgav2.py",
        "dgav3.py",
        "dga_a.py",
        "dga_b.py",
    ]
    for name in preferred:
        for p in py_files:
            if p.name == name:
                return p
    # File with a likely dga function
    for p in py_files:
        funcs = _parse_functions(p)
        if "dga" in funcs:
            return p
    return sorted(py_files)[0]


def _infer_callable_name(py_file: Path | None) -> str | None:
    if py_file is None:
        return None
    funcs = _parse_functions(py_file)
    for candidate in [
        "dga",
        "generate_domains",
        "get_domains",
        "create_domain",
        "next_domain",
    ]:
        if candidate in funcs:
            return candidate
    return funcs[0] if funcs else None


def _infer_tlds_from_examples(txt_files: list[Path]) -> list[str]:
    tlds: dict[str, int] = {}
    for txt in txt_files[:6]:
        try:
            lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines[:1000]:
            part = line.strip().lower().rstrip(".")
            if "." not in part:
                continue
            tld = "." + part.rsplit(".", 1)[-1]
            if 1 < len(tld) <= 10:
                tlds[tld] = tlds.get(tld, 0) + 1
    return [k for k, _ in sorted(tlds.items(), key=lambda kv: kv[1], reverse=True)[:10]]


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _stmt_list_without_docstring(stmts: list[ast.stmt]) -> list[ast.stmt]:
    out = list(stmts)
    if out and _is_docstring_expr(out[0]):
        out = out[1:]
    return out


def _is_stub_statement(node: ast.stmt) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if _is_docstring_expr(node):
        return True
    if isinstance(node, ast.Raise):
        exc = node.exc
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id in {"NotImplementedError"}:
            return True
        if isinstance(exc, ast.Name) and exc.id in {"NotImplementedError"}:
            return True
    return False


def _is_stub_body(stmts: list[ast.stmt]) -> bool:
    body = _stmt_list_without_docstring(stmts)
    if not body:
        return True
    return all(_is_stub_statement(n) for n in body)


def _is_name_main(node: ast.expr) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if not isinstance(node.left, ast.Name) or node.left.id != "__name__":
        return False
    if len(node.comparators) != 1 or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return False
    right = node.comparators[0]
    return isinstance(right, ast.Constant) and right.value == "__main__"


def _contains_redirect_text(raw_text: str) -> bool:
    lowered = raw_text.lower()
    if "http://" in lowered or "https://" in lowered:
        if "moved" in lowered or "redirect" in lowered or "see " in lowered:
            return True
    return any(marker in lowered for marker in _REDIRECT_MARKERS)


def _contains_placeholder_text(raw_text: str) -> bool:
    try:
        tree = ast.parse(raw_text)
    except (SyntaxError, ValueError):
        return False
    string_literals = "\n".join(
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return any(marker in string_literals for marker in _PLACEHOLDER_MARKERS)


def _is_executable_top_level_logic(node: ast.stmt) -> bool:
    if isinstance(node, ast.Expr):
        return not _is_docstring_expr(node)
    return isinstance(
        node,
        (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Try,
            ast.Match,
            ast.Return,
            ast.Raise,
            ast.Assert,
            ast.Delete,
        ),
    )


def _validate_entrypoint(entrypoint: Path, callable_name: str | None) -> tuple[bool, str | None]:
    raw_text = entrypoint.read_text(encoding="utf-8", errors="ignore")
    tree = _parse_tree(entrypoint)
    if tree is None:
        return False, DISCARD_MISSING_IMPLEMENTATION

    code_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not code_lines:
        if _contains_redirect_text(raw_text):
            return False, DISCARD_UNRESOLVED_REDIRECT
        return False, DISCARD_PLACEHOLDER_MODULE

    real_callable_names: set[str] = set()
    has_real_class = False
    has_real_main = False
    has_real_top_level_logic = False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_stub_body(node.body):
                real_callable_names.add(node.name)
            continue
        if isinstance(node, ast.ClassDef):
            if not _is_stub_body(node.body):
                has_real_class = True
            continue
        if isinstance(node, ast.If) and _is_name_main(node.test):
            if not _is_stub_body(node.body):
                has_real_main = True
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if _is_docstring_expr(node):
            continue
        if isinstance(node, ast.Pass):
            continue
        if _is_executable_top_level_logic(node):
            has_real_top_level_logic = True

    if callable_name and callable_name not in real_callable_names:
        if _contains_redirect_text(raw_text):
            return False, DISCARD_UNRESOLVED_REDIRECT
        if _contains_placeholder_text(raw_text):
            return False, DISCARD_INVALID_STUB
        return False, DISCARD_MISSING_IMPLEMENTATION

    if not (real_callable_names or has_real_class or has_real_main or has_real_top_level_logic):
        if _contains_redirect_text(raw_text):
            return False, DISCARD_UNRESOLVED_REDIRECT
        if _contains_placeholder_text(raw_text):
            return False, DISCARD_INVALID_STUB
        return False, DISCARD_MISSING_IMPLEMENTATION

    return True, None


def inspect_algorithm_dir(path: Path) -> AlgorithmInspection:
    py_files = sorted(path.glob("*.py"))
    txt_files = sorted(path.glob("*.txt"))
    entrypoint = _infer_entrypoint(py_files)
    callable_name = _infer_callable_name(entrypoint)

    required_params: list[str] = []
    requires_seed = False
    requires_date = False
    default_params: dict[str, object] = {}
    notes: list[str] = []
    discard_reason: str | None = None

    if entrypoint and callable_name:
        try:
            tree = ast.parse(entrypoint.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == callable_name:
                    for arg in node.args.args:
                        required_params.append(arg.arg)
                    break
        except Exception:
            pass

    params_lower = {p.lower() for p in required_params}
    requires_seed = "seed" in params_lower or "magic" in params_lower
    requires_date = bool(params_lower.intersection({"date", "d", "dt", "when", "time"}))

    for p in required_params:
        pl = p.lower()
        if pl in {"seed", "magic"}:
            default_params[p] = 1
        elif pl in {"date", "d", "dt", "when", "time"}:
            default_params[p] = "now"
        elif pl in {"nr", "n", "num", "num_domains", "count", "domain_nr", "sequence_nr"}:
            default_params[p] = 1
        elif pl in {"version"}:
            default_params[p] = "v1"

    strategy = "python_function" if entrypoint and callable_name else "cli_subprocess"
    status = "usable"

    if entrypoint is None:
        status = "discarded"
        discard_reason = DISCARD_MISSING_IMPLEMENTATION
        notes.append("validation:missing_entrypoint")
        strategy = "unknown"
    else:
        is_valid, reason = _validate_entrypoint(entrypoint, callable_name)
        if not is_valid:
            status = "discarded"
            discard_reason = reason
            notes.append(f"validation:{reason}")
            strategy = "unknown"
            callable_name = callable_name if callable_name and reason != DISCARD_PLACEHOLDER_MODULE else None

    return AlgorithmInspection(
        algorithm_code=path.name,
        path=str(path),
        python_files=[str(p) for p in py_files],
        txt_files=[str(p) for p in txt_files],
        strategy=strategy,
        entrypoint=str(entrypoint) if entrypoint else None,
        callable_name=callable_name,
        required_params=required_params,
        default_params=default_params,
        requires_seed=requires_seed,
        requires_date=requires_date,
        inferred_tlds=_infer_tlds_from_examples(txt_files),
        notes=notes,
        status=status,
        discard_reason=discard_reason,
    )


def discover_and_inspect(root: Path) -> list[AlgorithmInspection]:
    inspections: list[AlgorithmInspection] = []
    for algo_dir in discover_algorithm_dirs(root):
        inspections.append(inspect_algorithm_dir(algo_dir))
    return inspections
