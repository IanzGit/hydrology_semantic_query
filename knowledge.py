from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


@dataclass(frozen=True, slots=True)
class BusinessPlaybook:
    name: str
    content: str


def load_business_playbooks(
    directory: Path = KNOWLEDGE_DIR,
) -> tuple[tuple[BusinessPlaybook, ...], list[str]]:
    if not directory.exists():
        return (), []
    playbooks: list[BusinessPlaybook] = []
    warnings: list[str] = []
    for path in sorted(directory.glob("*.md"), key=lambda item: item.name):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            warnings.append(f"业务知识文件 {path.name} 读取失败，已跳过：{str(exc)[:200]}")
            continue
        if not content:
            warnings.append(f"业务知识文件 {path.name} 为空，已跳过。")
            continue
        playbooks.append(BusinessPlaybook(name=path.name, content=content))
    return tuple(playbooks), warnings


def render_business_playbooks(playbooks: tuple[BusinessPlaybook, ...]) -> str:
    if not playbooks:
        return "无"
    return "\n\n".join(
        f'<business_playbook name="{playbook.name}">\n{playbook.content}\n</business_playbook>'
        for playbook in playbooks
    )


__all__ = [
    "BusinessPlaybook",
    "KNOWLEDGE_DIR",
    "load_business_playbooks",
    "render_business_playbooks",
]
