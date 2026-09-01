from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv.main import dotenv_values

_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SENSITIVE_KEY_PATTERN = re.compile(r"(?i)(PASS|PASSWORD|TOKEN|SECRET|API_KEY|APIKEY)")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:PASS|PASSWORD|TOKEN|SECRET|API_KEY|APIKEY)\b\s*[:=]\s*)([^\s,;]+)"
)


class ProfileConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CubeStartProfile:
    name: str
    profile_path: Path
    model_name: str
    model_path: Path
    cube_host: str
    cube_port: int
    dev_mode: str
    database_type: str
    database_host: str
    database_name: str
    sensitive_values: tuple[str, ...]

    def shell_lines(self) -> tuple[str, ...]:
        return (
            self.name,
            str(self.profile_path),
            self.model_name,
            str(self.model_path),
            self.cube_host,
            str(self.cube_port),
            self.dev_mode,
            self.database_type,
            self.database_host,
            self.database_name,
        )


def _load_values(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ProfileConfigError(f"缺少{label}：{path}")
    raw_values = dotenv_values(path)
    return {
        key: value.strip()
        for key, value in raw_values.items()
        if value is not None
    }


def _required(values: dict[str, str], key: str, label: str) -> str:
    value = values.get(key, "")
    if not value:
        raise ProfileConfigError(f"{label}缺少必填项 {key}")
    if any(character in value for character in ("\r", "\n", "\0")):
        raise ProfileConfigError(f"{label}中的 {key} 包含非法控制字符")
    return value


def _port(value: str, key: str, label: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ProfileConfigError(f"{label}中的 {key} 必须是有效端口") from exc
    if not 0 < port < 65536:
        raise ProfileConfigError(f"{label}中的 {key} 必须在1到65535之间")
    return port


def _name(value: str, key: str, label: str) -> str:
    if _NAME_PATTERN.fullmatch(value) is None:
        raise ProfileConfigError(
            f"{label}中的 {key} 只能包含字母、数字、下划线和连字符"
        )
    return value


def load_start_profile(
    selector_path: Path,
    *,
    profiles_root: Path | None = None,
    models_root: Path | None = None,
) -> CubeStartProfile:
    selector_path = selector_path.resolve()
    start_root = selector_path.parent
    profiles_root = (profiles_root or start_root / "profiles").resolve()
    models_root = (models_root or start_root.parent / "model").resolve()
    selector = _load_values(selector_path, "启动配置")
    profile_name = _name(
        _required(selector, "ACTIVE_PROFILE", "启动配置"),
        "ACTIVE_PROFILE",
        "启动配置",
    )
    cube_host = _required(selector, "CUBE_HOST", "启动配置")
    cube_port = _port(
        _required(selector, "CUBE_PORT", "启动配置"),
        "CUBE_PORT",
        "启动配置",
    )
    dev_mode = _required(selector, "CUBEJS_DEV_MODE", "启动配置").lower()
    if dev_mode not in {"true", "false"}:
        raise ProfileConfigError("启动配置中的 CUBEJS_DEV_MODE 只能是true或false")
    profile_path = profiles_root / f"{profile_name}.env"
    if not profile_path.is_file():
        example_path = profiles_root / f"{profile_name}.env.example"
        if example_path.is_file():
            raise ProfileConfigError(
                f"缺少Profile：{profile_path}，请复制 {example_path.name} 后填写数据库配置"
            )
        raise ProfileConfigError(f"缺少Profile：{profile_path}")
    profile = _load_values(profile_path, "Profile")
    model_name = _name(
        _required(profile, "MODEL_NAME", "Profile"),
        "MODEL_NAME",
        "Profile",
    )
    database_type = _required(profile, "CUBEJS_DB_TYPE", "Profile")
    database_host = _required(profile, "CUBEJS_DB_HOST", "Profile")
    _port(
        _required(profile, "CUBEJS_DB_PORT", "Profile"),
        "CUBEJS_DB_PORT",
        "Profile",
    )
    database_name = _required(profile, "CUBEJS_DB_NAME", "Profile")
    _required(profile, "CUBEJS_DB_USER", "Profile")
    _required(profile, "CUBEJS_DB_PASS", "Profile")
    model_path = models_root / model_name
    for directory_name in ("cubes", "views"):
        directory = model_path / directory_name
        if not directory.is_dir() or not any(directory.glob("*.yml")):
            raise ProfileConfigError(
                f"模型目录缺少有效的 {directory_name}：{directory}"
            )
    sensitive_values = tuple(
        dict.fromkeys(
            value
            for key, value in profile.items()
            if value and _SENSITIVE_KEY_PATTERN.search(key)
        )
    )
    return CubeStartProfile(
        name=profile_name,
        profile_path=profile_path.resolve(),
        model_name=model_name,
        model_path=model_path.resolve(),
        cube_host=cube_host,
        cube_port=cube_port,
        dev_mode=dev_mode,
        database_type=database_type,
        database_host=database_host,
        database_name=database_name,
        sensitive_values=sensitive_values,
    )


def redact_text(value: str, sensitive_values: tuple[str, ...]) -> str:
    redacted = value
    for sensitive_value in sensitive_values:
        redacted = redacted.replace(sensitive_value, "***")
    return _ASSIGNMENT_PATTERN.sub(r"\1***", redacted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析并校验Cube启动Profile")
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--redact-stdin", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        profile = load_start_profile(arguments.selector)
    except ProfileConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if arguments.redact_stdin:
        print(redact_text(sys.stdin.read(), profile.sensitive_values), end="")
        return 0
    print("\n".join(profile.shell_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
