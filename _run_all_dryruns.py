"""Run every _dryrun_p198*.py at the repo root and report a single
OK/FAIL line per script. Forces UTF-8 output so Unicode arrows in the
print statements don't crash on Windows cp1252 consoles.

Usage:
    python _run_all_dryruns.py            # run everything
    python _run_all_dryruns.py p198dp     # filter by substring
    python _run_all_dryruns.py p198dp p198dv  # multiple filters

Exit code is non-zero when any script fails or returns non-zero.
"""
import glob
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def discover():
    paths = sorted(glob.glob('_dryrun_p198*.py'))
    return paths


def summary_line(stdout):
    """Pull a one-line summary out of the script's stdout."""
    if not stdout:
        return '(no output)'
    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return '(empty)'
    # Patterns the dry-runs use for their final verdict
    for ln in reversed(lines[-15:]):
        if re.search(
            r'OVERALL[: ]|^Total:|scenarios? OK|/[0-9]+ .* OK|all .* pass'
            r'|0 hit|0 problem|truncation|0 files|GREEN|FAIL\b',
            ln, re.IGNORECASE):
            return ln.strip()[:160]
    return lines[-1].strip()[:160]


def main():
    args = sys.argv[1:]
    paths = discover()
    if args:
        paths = [p for p in paths if any(a.lower() in p.lower() for a in args)]

    if not paths:
        print('No dry-run scripts matched.')
        return 1

    print(f'Running {len(paths)} dry-run script(s)...')
    print('=' * 78)

    results = []
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    for p in paths:
        t0 = time.time()
        try:
            cp = subprocess.run(
                [sys.executable, p],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8',
                errors='replace',
            )
            ok = cp.returncode == 0
            line = summary_line(cp.stdout)
            results.append((p, ok, time.time() - t0, line))
        except subprocess.TimeoutExpired:
            results.append((p, False, time.time() - t0, '(timeout)'))
        except Exception as e:
            results.append((p, False, time.time() - t0, f'(error: {e})'))

    fail = sum(1 for _, ok, _, _ in results if not ok)
    for p, ok, dt, line in results:
        tag = 'OK  ' if ok else 'FAIL'
        print(f'[{tag}] {dt:5.1f}s  {os.path.basename(p):60s}  {line}')

    print('=' * 78)
    total = len(results)
    ok_n = total - fail
    print(f'Summary: {ok_n}/{total} OK' + (f', {fail} failed' if fail else ''))
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
