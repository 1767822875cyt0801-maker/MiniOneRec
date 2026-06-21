import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


SID_TOKEN_RE = re.compile(r"<[A-Za-z]_\d+>")


def normalize_sid(sid: str) -> str:
    """Remove whitespace around and between SID tokens."""
    if sid is None:
        return ""
    return re.sub(r"\s+", "", str(sid).strip())


def parse_sid_tokens(sid: str) -> list[str]:
    """Parse a full SID into tokens, returning [] for malformed input."""
    normalized = normalize_sid(sid)
    if not normalized:
        return []

    tokens = SID_TOKEN_RE.findall(normalized)
    if not tokens or "".join(tokens) != normalized:
        return []
    return tokens


def sid_prefixes(sid: str) -> list[str]:
    """Return cumulative SID prefixes from level 1 to the full SID."""
    tokens = parse_sid_tokens(sid)
    return ["".join(tokens[: i + 1]) for i in range(len(tokens))]


def is_valid_sid(sid: str, valid_sid_set: set[str]) -> bool:
    return normalize_sid(sid) in valid_sid_set


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, set):
        data = sorted(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def load_json_set(path: str | os.PathLike[str]) -> set[str]:
    data = load_json(path)
    if isinstance(data, dict):
        return set(str(key) for key in data.keys())
    if isinstance(data, list):
        return set(str(value) for value in data)
    raise TypeError(f"Expected list or dict JSON for set loading: {path}")


def build_sid2items(item2sid: dict[Any, Any]) -> dict[str, list[str]]:
    sid2items: dict[str, list[str]] = {}
    for item_id, sid in item2sid.items():
        normalized_sid = normalize_sid(str(sid))
        sid2items.setdefault(normalized_sid, []).append(str(item_id))

    return {
        sid: sorted(items, key=_item_sort_key)
        for sid, items in sorted(sid2items.items())
    }


def build_item2sid_from_csv_or_info(
    csv_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
    info_path: str | os.PathLike[str] | None = None,
    index_path: str | os.PathLike[str] | None = None,
    source_priority: Sequence[str] = ("index", "info", "csv"),
    strict: bool = True,
) -> dict[str, str]:
    """Build item_id -> SID from index/info/csv sources.

    Priority means earlier sources win when lower-priority sources are missing.
    In strict mode, conflicting SID assignments across sources raise ValueError.
    """
    source_maps: dict[str, dict[str, str]] = {}

    if index_path is not None:
        source_maps["index"] = _build_item2sid_from_index(index_path, strict=strict)
    if info_path is not None:
        source_maps["info"] = _build_item2sid_from_info(info_path, strict=strict)
    normalized_csv_paths = _as_path_list(csv_paths)
    if normalized_csv_paths:
        source_maps["csv"] = _build_item2sid_from_csvs(normalized_csv_paths, strict=strict)

    conflicts = _find_cross_source_conflicts(source_maps)
    if strict and conflicts:
        examples = "; ".join(
            f"{item_id}: {assignments}" for item_id, assignments in conflicts[:10]
        )
        raise ValueError(f"Conflicting item SID assignments across sources: {examples}")

    item2sid: dict[str, str] = {}
    for source in source_priority:
        for item_id, sid in source_maps.get(source, {}).items():
            if item_id not in item2sid:
                item2sid[item_id] = sid

    for source, mapping in source_maps.items():
        if source in source_priority:
            continue
        for item_id, sid in mapping.items():
            if item_id not in item2sid:
                item2sid[item_id] = sid

    return {
        item_id: item2sid[item_id]
        for item_id in sorted(item2sid.keys(), key=_item_sort_key)
    }


def _build_item2sid_from_index(
    index_path: str | os.PathLike[str], strict: bool
) -> dict[str, str]:
    raw_index = load_json(index_path)
    if not isinstance(raw_index, dict):
        raise TypeError(f"Expected dict index JSON: {index_path}")

    mapping: dict[str, str] = {}
    for item_id, sid_value in raw_index.items():
        if isinstance(sid_value, list):
            sid = "".join(str(token).strip() for token in sid_value)
        elif isinstance(sid_value, str):
            sid = sid_value
        else:
            if strict:
                raise ValueError(f"Unsupported SID value for item {item_id}: {sid_value!r}")
            continue
        _add_mapping(mapping, item_id, sid, f"{index_path}", strict)
    return mapping


def _build_item2sid_from_info(
    info_path: str | os.PathLike[str], strict: bool
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(info_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                if strict:
                    raise ValueError(f"Malformed info row {info_path}:{line_no}: {line!r}")
                continue
            _add_mapping(mapping, parts[2], parts[0], f"{info_path}:{line_no}", strict)
    return mapping


def _build_item2sid_from_csvs(
    csv_paths: Sequence[Path], strict: bool
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_no, row in enumerate(reader, start=2):
                source = f"{csv_path}:{row_no}"
                if "item_id" in row and "item_sid" in row:
                    _add_mapping(mapping, row["item_id"], row["item_sid"], source, strict)

                if "history_item_id" not in row or "history_item_sid" not in row:
                    continue

                history_item_ids = _parse_list(row["history_item_id"], source, strict)
                history_item_sids = _parse_list(row["history_item_sid"], source, strict)
                if len(history_item_ids) != len(history_item_sids):
                    if strict:
                        raise ValueError(
                            f"history_item_id/history_item_sid length mismatch at {source}: "
                            f"{len(history_item_ids)} != {len(history_item_sids)}"
                        )
                    continue
                for item_id, sid in zip(history_item_ids, history_item_sids):
                    _add_mapping(mapping, item_id, sid, source, strict)
    return mapping


def _add_mapping(
    mapping: dict[str, str],
    item_id: Any,
    sid: Any,
    source: str,
    strict: bool,
) -> None:
    item_id_str = str(item_id).strip()
    normalized_sid = normalize_sid(str(sid))
    if not item_id_str or not parse_sid_tokens(normalized_sid):
        if strict:
            raise ValueError(f"Invalid item/SID mapping at {source}: {item_id!r} -> {sid!r}")
        return

    old_sid = mapping.get(item_id_str)
    if old_sid is not None and old_sid != normalized_sid:
        if strict:
            raise ValueError(
                f"Conflicting SID for item {item_id_str} at {source}: "
                f"{old_sid!r} != {normalized_sid!r}"
            )
        return
    mapping[item_id_str] = normalized_sid


def _find_cross_source_conflicts(
    source_maps: dict[str, dict[str, str]]
) -> list[tuple[str, dict[str, str]]]:
    by_item: dict[str, dict[str, str]] = {}
    for source, mapping in source_maps.items():
        for item_id, sid in mapping.items():
            by_item.setdefault(item_id, {})[source] = sid

    conflicts: list[tuple[str, dict[str, str]]] = []
    for item_id, assignments in by_item.items():
        if len(set(assignments.values())) > 1:
            conflicts.append((item_id, assignments))
    return sorted(conflicts, key=lambda item: _item_sort_key(item[0]))


def _parse_list(value: Any, source: str, strict: bool) -> list[Any]:
    if isinstance(value, list):
        return value

    value_str = "" if value is None else str(value).strip()
    if not value_str:
        return []

    try:
        parsed = ast.literal_eval(value_str)
    except (SyntaxError, ValueError) as exc:
        if strict:
            raise ValueError(f"Unable to parse list at {source}: {value!r}") from exc
        return []

    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def _as_path_list(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None,
) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _item_sort_key(item_id: Any) -> tuple[int, int | str]:
    item_id_str = str(item_id)
    try:
        return (0, int(item_id_str))
    except ValueError:
        return (1, item_id_str)
