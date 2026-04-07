from typing import Any


def _iter_parts(path: str) -> list[str]:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError("module_path cannot be empty")
    return parts


def get_at_path(root: Any, path: str) -> Any:
    current = root
    for part in _iter_parts(path):
        if part.isdigit():
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def set_at_path(root: Any, path: str, value: Any) -> None:
    parts = _iter_parts(path)
    if len(parts) == 1:
        parent = root
        final = parts[0]
    else:
        parent = get_at_path(root, ".".join(parts[:-1]))
        final = parts[-1]

    if final.isdigit():
        parent[int(final)] = value
    elif isinstance(parent, dict):
        parent[final] = value
    else:
        setattr(parent, final, value)
