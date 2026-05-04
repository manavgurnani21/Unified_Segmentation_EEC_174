"""Strip stray backslash-quote escapes from generated Stage 2 notebooks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
NEEDLE = chr(92) + chr(39)   # backslash + single-quote
REPLACE = chr(39)


def main() -> int:
    fixed = 0
    for p in sorted(NB_DIR.glob('0?_*.ipynb')):
        nb = json.loads(p.read_text(encoding='utf-8'))
        changed = False
        for cell in nb['cells']:
            src = cell['source']
            if not isinstance(src, list):
                continue
            new_lines = []
            for ln in src:
                if NEEDLE in ln:
                    ln = ln.replace(NEEDLE, REPLACE)
                    changed = True
                new_lines.append(ln)
            cell['source'] = new_lines
        if changed:
            p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
            fixed += 1
            print('fixed', p.name)
    print('done; fixed', fixed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
