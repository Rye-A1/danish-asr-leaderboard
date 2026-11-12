"""Check that CoRal audio is offered only for matching clips after consent."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_coral_audio_requires_consent_and_matching_reference():
    index = Path(__file__).resolve().parents[1] / "huggingface/space/index.html"
    script = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('  async function renderExampleAudio(');
const end = html.indexOf('  function render() {', start);
assert(start >= 0 && end > start);
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.isConnected = true; }
  append(...items) { for (const item of items) { if (typeof item !== 'string') item.parent = this; this.children.push(item); } }
  appendChild(item) { this.append(item); }
  insertBefore(item) { this.append(item); }
  querySelector() { return null; }
  addEventListener(name, cb) { this[name] = cb; }
  setAttribute(name, value) { this[name] = value; }
  replaceWith(item) { const i = this.parent.children.indexOf(this); this.parent.children[i] = item; item.parent = this.parent; }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
}
function find(root, tag) {
  if (root.tag === tag) return root;
  for (const child of root.children || []) if (typeof child !== 'string') { const found = find(child, tag); if (found) return found; }
  return null;
}
const storage = {};
const fetches = [];
const item = {id:'coral_conversation_00001.wav', reference:'a reference'};
const context = {URL, document:{createElement: tag => new Element(tag)},
  localStorage:{setItem:(key, value) => storage[key] = value},
  fetch: async url => { fetches.push(url); return {ok:true, json: async () => ({examples:{[item.id]:'a reference'}})}; },
  window:{}};
vm.runInNewContext(`let fleursManifestPromise = null;
const coralLicenseKey = 'test-license';
let coralLicenseAccepted = false;
const coralManifests = {};
${html.slice(start, end)}
window.renderAudio = renderExampleAudio;`, context);
(async () => {
  const ref = new Element('div');
  const head = new Element('div');
  await context.window.renderAudio(ref, head, 'coral_conversation', item);
  const base = 'https://huggingface.co/datasets/RyeAI/coral-v3-test-playback/resolve/main/audio/coral_conversation';
  assert.strictEqual(fetches[0], base + '/manifest.json');
  assert.strictEqual(find(ref, 'a').href, base + '/LICENSE');
  assert.strictEqual(find(ref, 'audio'), null);
  const button = find(ref, 'button');
  assert(button && button.textContent.includes('Agree'));
  button.click();
  assert.strictEqual(storage['test-license'], 'accepted');
  assert.strictEqual(find(ref, 'audio').src, base + '/' + item.id);
  const second = new Element('div');
  await context.window.renderAudio(second, new Element('div'), 'coral_conversation', item);
  assert(find(second, 'audio'));
  const mismatch = new Element('div');
  await context.window.renderAudio(mismatch, new Element('div'), 'coral_conversation', {...item, reference:'changed'});
  assert.strictEqual(find(mismatch, 'audio'), null);
  const invalid = new Element('div');
  await context.window.renderAudio(invalid, new Element('div'), 'coral_conversation', {...item, id:'../unsafe.wav'});
  assert.strictEqual(find(invalid, 'audio'), null);
})().catch(e => { console.error(e); process.exitCode = 1; });
"""
    subprocess.run(["node", "-e", script, str(index)], check=True, timeout=30)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_correction_button_toggles_the_form():
    index = Path(__file__).resolve().parents[1] / "huggingface/space/index.html"
    script = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf("    const report = document.createElement('div');", html.indexOf('  function renderExample('));
const end = html.indexOf('    target.append(reference, columns);', start);
assert(start >= 0 && end > start);
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.listeners = {}; }
  append(...items) { this.children.push(...items); }
  appendChild(item) { this.append(item); }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, cb) { this.listeners[name] = cb; }
}
const reference = new Element('div');
const context = {
  document: {createElement: tag => new Element(tag)},
  _icon: (_, cls) => ({tag:'svg', cls}),
  item: {reference:'original text', id:'clip.wav'}, dataset:'coral_conversation',
  selected:[], window:{location:{href:'https://example.test/'}},
  referenceIssueURL: () => 'https://example.test/issue', CORRECTION_ENDPOINT:'', reference,
};
vm.runInNewContext(html.slice(start, end), context);
const [button, panel] = reference.children[0].children;
assert.strictEqual(button.tag, 'button');
assert.strictEqual(button.attributes['aria-expanded'], 'false');
assert.strictEqual(button.children[0].cls, 'correction-edit');
assert.strictEqual(button.children[2].cls, 'correction-chevron');
assert.strictEqual(panel.hidden, true);
button.listeners.click();
assert.strictEqual(panel.hidden, false);
assert.strictEqual(button.attributes['aria-expanded'], 'true');
button.listeners.click();
assert.strictEqual(panel.hidden, true);
"""
    subprocess.run(["node", "-e", script, str(index)], check=True, timeout=30)
