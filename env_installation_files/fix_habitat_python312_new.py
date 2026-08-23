import re
import sys
from pathlib import Path

site_packages = next(p for p in sys.path if 'site-packages' in p)
PACKAGES = ["habitat", "habitat_baselines"]

MUTABLE_DEFAULT_RE = re.compile(
    r'(:\s*)([A-Z][A-Za-z0-9]*)\s*=\s*([A-Z][A-Za-z0-9]*)\(\)'
)


def find_candidates() -> list[Path]:
    """Scan for files that both define a dataclass and contain the
    'field: TypeName = TypeName()' mutable-default pattern."""
    candidates = []
    for pkg in PACKAGES:
        pkg_root = Path(site_packages) / pkg
        if not pkg_root.exists():
            continue
        for path in pkg_root.rglob("*.py"):
            if path.name.endswith(".bak"):
                continue
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if "@dataclass" in text and MUTABLE_DEFAULT_RE.search(text):
                candidates.append(path)
    return candidates


def patch(path: Path) -> bool:
    print(f"Patching: {path}")
    content = path.read_text()

    fixed = MUTABLE_DEFAULT_RE.sub(
        lambda m: m.group(0) if m.group(2) in ('List', 'Dict', 'Tuple', 'Set')
                  else f'{m.group(1)}{m.group(2)} = field(default_factory={m.group(3)})',
        content
    )

    if fixed == content:
        print("  (no changes needed -- already patched or pattern not present)")
        return True

    if 'from dataclasses import' in fixed and 'field' not in fixed.split('from dataclasses import')[1].split('\n')[0]:
        fixed = fixed.replace('from dataclasses import dataclass', 'from dataclasses import dataclass, field')

    path.write_text(fixed)
    print("  Patched.")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = [Path(site_packages) / p for p in sys.argv[1:]]
        missing = [p for p in targets if not p.exists()]
        for p in missing:
            print(f"SKIP (not found): {p}")
        targets = [p for p in targets if p.exists()]
    else:
        print(f"Scanning {', '.join(PACKAGES)} for affected files...")
        targets = find_candidates()
        if not targets:
            print("No affected files found.")
        else:
            print(f"Found {len(targets)} file(s):")
            for t in targets:
                print(f"  {t}")
            print()

    results = [patch(t) for t in targets]

    if targets and all(results):
        print("\nDone -- verify with: python -c \"import habitat; import habitat_baselines; print('OK')\"")
    elif not targets:
        sys.exit(0)
    else:
        print("\nSome files were not found -- check paths above.")
