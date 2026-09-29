"""Execute the unchanged notebook in a fresh kernel with real Yahoo downloads.

Run from the project root: python tests/run_notebook.py
Requires network access; fails on cell errors or incomplete scanner results.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks' / 'Beurs.ipynb'
    before = path.read_bytes()
    notebook = json.loads(before)
    with tempfile.TemporaryDirectory(prefix='notebook-check-') as temp:
        manager = KernelManager(connection_file=str(Path(temp) / 'kernel.json'))
        manager._kernel_spec = KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher',
                                                '-f', '{connection_file}'],
                                          display_name='Project validation', language='python')
        env = dict(os.environ, MPLBACKEND='Agg', IPYTHONDIR=temp)
        started = time.monotonic()
        manager.start_kernel(cwd=str(root / 'notebooks'), env=env)
        client = manager.blocking_client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=60)
            print(f'Kernel ready in {time.monotonic()-started:.2f}s', flush=True)

            def execute(source):
                reply = client.execute_interactive(source, timeout=300)
                if reply['content']['status'] != 'ok':
                    raise RuntimeError(reply['content'])

            execute(f'import yfinance as yf; yf.set_tz_cache_location({temp!r})')
            count = 0
            for index, cell in enumerate(notebook['cells']):
                if cell['cell_type'] != 'code':
                    continue
                start = time.monotonic()
                execute(''.join(cell['source']))
                execute("import matplotlib.pyplot as plt\n"
                        "for number in plt.get_fignums(): plt.figure(number).canvas.draw()\n"
                        "plt.close('all')")
                if index in (18, 20):
                    execute("assert result.status.eq('OK').all(), result[['ticker', 'status', 'message']].to_string()")
                    execute("assert set(result.ticker) == {t.upper() for t in tickers}")
                count += 1
                print(f'Cell {index}: OK ({time.monotonic()-start:.2f}s)', flush=True)
            start = time.monotonic()
            manager.restart_kernel(now=True)
            client.wait_for_ready(timeout=60)
            execute('assert 1 + 1 == 2')
            print(f'Kernel restart: OK ({time.monotonic()-start:.2f}s)', flush=True)
            print(f'{count} code cells passed; SHA256 {hashlib.sha256(before).hexdigest()}')
        finally:
            client.stop_channels()
            manager.shutdown_kernel(now=True)
            assert path.read_bytes() == before, 'Notebook changed during validation'


if __name__ == '__main__':
    main()
