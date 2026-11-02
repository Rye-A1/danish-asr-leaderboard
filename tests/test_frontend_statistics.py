"""Check that cached browser statistics match an uncached calculation."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_bootstrap_cache_preserves_results():
    index = Path(__file__).resolve().parent.parent / "huggingface/space/index.html"
    script = r"""
const assert = require('assert');
const fs = require('fs');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('function addBordaRanks(');
const end = html.indexOf('/* ── Dynamic column sets', start);
assert(start >= 0 && end > start);
eval(html.slice(start, end));

const cols = ['a', 'b', 'c', 'd', 'e'];
const rows = Array.from({length: 12}, (_, i) => Object.fromEntries(
  cols.map((col, j) => [col, (i + j) % 11 === 0 ? null : (i * j + j * 3) % 17])
));
let calls = 0;
const originalBt = _btFn;
_btFn = (...args) => { calls++; return originalBt(...args); };
function run(rows, disableCache) {
  let seed = 12345;
  const originalRandom = Math.random, originalGet = Map.prototype.get;
  Math.random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2**32; };
  if (disableCache) Map.prototype.get = () => undefined;
  calls = 0;
  try { addBootstrapCI(rows, cols, 'mean_wer'); }
  finally { Math.random = originalRandom; Map.prototype.get = originalGet; }
  return calls;
}
const uncached = structuredClone(rows), cached = structuredClone(rows);
assert.strictEqual(run(uncached, true), 1000);
assert(run(cached, false) <= 126);
assert.deepStrictEqual(cached, uncached);
"""
    subprocess.run(["node", "-e", script, str(index)], check=True, timeout=30)


def test_snapshot_refresh_omits_stale_compare_data(tmp_path):
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parent.parent / ".github/workflows/snapshot-hf-downloads.yml"
    steps = yaml.safe_load(workflow.read_text())["jobs"]["snapshot"]["steps"]
    command = next(step["run"] for step in steps if step.get("name") == "Refresh Space")
    space = tmp_path / "huggingface/space"
    space.mkdir(parents=True)
    (space / "compare_data.json").write_text("old")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text("#!/bin/sh\ntest ! -e huggingface/space/compare_data.json\n")
    uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(["bash", "-e", "-c", command], cwd=tmp_path, env=env, check=True)
