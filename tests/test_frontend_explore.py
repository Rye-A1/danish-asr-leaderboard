"""Exercise Explore ranking, missing history, cards and sample links."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_explore_lists_and_sample_links():
    index = Path(__file__).resolve().parent.parent / "huggingface/space/index.html"
    script = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const part = (start, end) => {
  const a = html.indexOf(start), b = html.indexOf(end, a);
  assert(a >= 0 && b > a, `missing ${start}`);
  return html.slice(a, b);
};
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.classList = {add: () => {}}; this.listeners = {}; }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(event, callback) { this.listeners[event] = callback; }
  click() { this.listeners.click(); }
}
const elements = Object.fromEntries(['explore-trending', 'explore-recent', 'explore-samples', 'explore-time-graph', 'explore-time-toggle', 'tab-time']
  .map(id => [id, new Element('div')]));
elements['tab-time'].hidden = true;
let redraws = 0;
const context = {
  URL, URLSearchParams, Intl,
  window: {location: {href: 'https://example.test/?view=explore&model=old&clip=old'}},
  document: {getElementById: id => elements[id], createElement: tag => new Element(tag),
    createElementNS: (_, tag) => new Element(tag)},
  chartWerRows: (rows, cutoff) => rows.filter(r => r.mean_wer == null || r.mean_wer <= cutoff),
  requestAnimationFrame: callback => callback(),
  redraw: () => redraws++,
};
const code = part('let sparkId = 0;', 'function downloadCell(')
  + part('function modelAvatar(', 'function showModelPreview(')
  + part('function chartScoreKey(', 'function sizeChartPoints(')
  + part('function parseISODate(', 'const TIME_CSV_FIELDS')
  + part('let openSamples = null;', "document.addEventListener('click', event => {");
vm.runInNewContext(code + `
  setupExplore({wer: [
    {name:'new&model', hf_downloads:100, hf_download_history:[10, 12, 15, 20, 30, 40, 60, 100], submitted:'2026-11-02', released:'2026-02-01', mean_wer:20, rank:1},
    {name:'popular', logo:'https://example.test/logo.png', hf_downloads:200, hf_download_history:[100, 110, 120, 130, 140, 150, 180, 200], submitted:'2026-10-01', released:'2026-01-01', mean_wer:25, rank:9},
    {name:'missing', hf_downloads:null, submitted:'2026-10-02', mean_wer:null},
    {name:'zero', hf_downloads:2, hf_download_history:[0, 0, 0, 0, 1, 1, 1, 2], submitted:'2026-11-01', mean_wer:10},
    {name:'incomplete', hf_downloads:1000, hf_download_history:[1, 2, 3], submitted:'2026-10-03', mean_wer:22},
    {name:'declining', hf_downloads:5, hf_download_history:[10, 9, 8, 7, 6, 5, 5, 5], submitted:'2026-10-04', mean_wer:23},
  ]}, redraw);
  window.sample = samplesURL('new&model', 'cv17_da');
  window.issue = referenceIssueURL('cv17_da', 'clip 2.wav', ['new&model'],
    'https://example.test/?token=private-value&model=old#other');
  window.localIssue = referenceIssueURL('cv17_da', 'clip.wav', ['model'], 'http://127.0.0.1:7882/');
`, context);
const trending = elements['explore-trending'].children;
const recent = elements['explore-recent'].children;
const link = row => row.children[0].children[1].children[0];
assert.deepStrictEqual(trending.map(row => link(row).textContent), ['new&model', 'popular'], 'growth, not total downloads');
assert.deepStrictEqual(recent.map(row => link(row).textContent), ['new&model', 'zero', 'declining', 'incomplete', 'missing']);
assert.strictEqual(trending[1].children[0].children[0].children[0].src, 'https://example.test/logo.png');
const trend = trending[0].children[1].children[0];
assert.strictEqual(trend.children[0].children[0].tag, 'svg');
assert.strictEqual(trend.children[0].children[1].children[0].textContent, '100');
assert.strictEqual(trend.children[0].children[1].children[1].textContent, '+900.0%');
assert.strictEqual(recent[0].children[0].children[1].children[1].textContent, 'Submitted 2026-11-02');
assert.strictEqual(recent[0].children[1].children[0].children[0].textContent, '#2');
assert.strictEqual(recent[0].children[1].children[1].children[0].textContent, '20.00%');
assert.strictEqual(recent[4].children[1].children[0].children[0].textContent, '—');
assert.strictEqual(recent[4].children[1].children[1].children[0].textContent, '—');
const historyGraph = elements['explore-time-graph'].children[0];
assert.strictEqual(historyGraph.tag, 'svg');
const line = historyGraph.children.find(child => child.tag === 'path' && child.stroke === 'currentColor');
const yValues = [...line.d.matchAll(/[ML][\d.]+,([\d.]+)/g)].map(match => Number(match[1]));
assert(yValues[0] > yValues.at(-1), 'improving WER rises on the history graph');
assert.strictEqual(elements['tab-time'].hidden, true);
elements['explore-time-toggle'].click();
assert.strictEqual(elements['tab-time'].hidden, false);
assert.strictEqual(elements['explore-time-toggle']['aria-expanded'], 'true');
assert.strictEqual(redraws, 1);
elements['explore-time-toggle'].click();
assert.strictEqual(elements['tab-time'].hidden, true);
const url = new URL(context.window.sample);
assert.strictEqual(url.searchParams.get('model'), 'new&model');
assert.strictEqual(url.searchParams.get('dataset'), 'cv17_da');
assert.strictEqual(url.searchParams.get('view'), 'compare');
assert(!url.searchParams.has('clip'));
assert.strictEqual(url.hash, '#compare-transcripts');
const issue = new URL(context.window.issue);
const body = issue.searchParams.get('body');
assert.strictEqual(issue.origin, 'https://github.com');
assert(body.includes('Clip ID: clip 2.wav'));
assert(body.includes('Optional or uncertain words'));
assert(!body.includes('private-value'));
const sampleURL = new URL(body.split('\n').find(line => line.startsWith('Sample: ')).slice(8));
assert.strictEqual(sampleURL.searchParams.get('model'), 'new&model');
assert.strictEqual(sampleURL.searchParams.get('clip'), 'clip 2.wav');
assert(!sampleURL.searchParams.has('token'));
assert(!new URL(context.window.localIssue).searchParams.get('body').includes('Sample: '));
"""
    subprocess.run(["node", "-e", script, str(index)], check=True, timeout=30)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_highlights_follow_displayed_text_and_refresh_stale_manifest():
    index = Path(__file__).resolve().parent.parent / "huggingface/space/index.html"
    script = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const part = (start, end) => {
  const a = html.indexOf(start), b = html.indexOf(end, a);
  assert(a >= 0 && b > a, `missing ${start}`);
  return html.slice(a, b);
};
const calls = [];
let newManifest = false;
const manifests = [
  {references: {first: 'old-ref-1', second: 'old-ref-2'}, models: {model: {paths: {first: 'old-out-1', second: 'old-out-2'}}}},
  {references: {second: 'new-ref'}, models: {model: {paths: {second: 'new-out'}}}},
];
const fetch = async path => {
  calls.push(path);
  if (path === 'compare-all/manifest.json') return {ok: true, json: async () => manifests[+newManifest]};
  if (path.startsWith('old-') && newManifest) return {ok: false, status: 404};
  if (path === 'new-ref') return {ok: true, json: async () => ['new-reference']};
  if (path === 'new-out') return {ok: true, json: async () => ({clip: 'new-output'})};
  return {ok: true, json: async () => []};
};
const context = {fetch, browseCache: new Map(), result: null, Error};
const code = part('  function compareWords(value)', '  function compareWordChanges(')
  + part('  function browseFetch(path)', '  function loadBrowse(')
  + part('  function highlightExamples(', '  function renderExample()');
vm.runInNewContext(code + `
  const candidates = [
    {id:'punctuation', hypotheses:{one:'C Matador', two:'C-matador'},
      normalized_hypotheses:{one:'c matador', two:'cmatador'}},
    {id:'number', hypotheses:{one:'3', two:'tre'},
      normalized_hypotheses:{one:'tre', two:'tre'}},
    {id:'missing', hypotheses:{one:'et'}, normalized_hypotheses:{one:'et'}}
  ];
  result = {
    raw: highlightExamples(candidates, ['one', 'two'], 'raw').map(item => item.id),
    normalized: highlightExamples(candidates, ['one', 'two'], 'normalized').map(item => item.id),
    fallback: highlightExamples([candidates[2]], ['one', 'two'], 'normalized').map(item => item.id),
    browseDataset
  };
`, context);
assert.deepStrictEqual(Array.from(context.result.raw), ['number']);
assert.deepStrictEqual(Array.from(context.result.normalized), ['punctuation']);
assert.deepStrictEqual(Array.from(context.result.fallback), ['missing']);
(async () => {
  await context.result.browseDataset('first', ['model']);
  newManifest = true;
  const data = await context.result.browseDataset('second', ['model']);
  assert.deepStrictEqual(Array.from(data.references), ['new-reference']);
  assert.strictEqual(data.outputs[0].clip, 'new-output');
  assert.strictEqual(calls.filter(path => path === 'compare-all/manifest.json').length, 2);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    subprocess.run(["node", "-e", script, str(index)], check=True, timeout=30)
