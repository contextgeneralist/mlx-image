from __future__ import annotations

from mflux.models.common.training.state.training_spec import BlockRange, LoraTargetSpec


def expand_module_paths(target: LoraTargetSpec) -> list[str]:
    if target.blocks is None:
        return [target.module_path]
    if "{block}" not in target.module_path:
        raise ValueError(f"Target has blocks specified but module_path contains no '{{block}}': {target.module_path}")
    blocks: BlockRange = target.blocks
    return [target.module_path.format(block=b) for b in blocks.get_blocks()]


def expand_module_paths_from_targets(targets: list[LoraTargetSpec]) -> list[tuple[str, int]]:
    expanded: list[tuple[str, int]] = []
    for t in targets:
        expanded.extend((p, t.rank) for p in expand_module_paths(t))
    return expanded
