"""Exercise the Apps Script inbox without making external requests."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is required')
def test_correction_inbox():
    script_path = Path(__file__).resolve().parents[1] / 'scripts/correction_inbox/Code.gs'
    test = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const cache = new Map(), properties = new Map([
  ['CORRECTIONS_SHEET_ID', 'private_sheet_identifier_12345'],
  ['SERVICE_ACCOUNT_EMAIL', 'inbox@project.iam.gserviceaccount.com'],
  ['WIF_PROVIDER', '//iam.googleapis.com/projects/123/locations/global/workloadIdentityPools/only-inbox/providers/apps-script'],
]);
const rows = [['Submitted (UTC)', 'Dataset', 'Clip ID', 'Original reference',
  'Suggested reference', 'Sample link', 'Review status', 'Submission ID']];
const requests = [];
const context = {
  CacheService: {getScriptCache: () => ({get: key => cache.get(key), put: (key, value) => cache.set(key, value)})},
  PropertiesService: {getScriptProperties: () => ({getProperty: key => properties.get(key), setProperty: (key, val) => properties.set(key, val),
    setProperties: values => Object.entries(values).forEach(([key, val]) => properties.set(key, val))})},
  LockService: {getScriptLock: () => ({tryLock: () => true, releaseLock: () => {}})},
  ScriptApp: {getIdentityToken: () => 'identity-token'},
  UrlFetchApp: {fetch: (url, options) => {
    requests.push({url, options});
    let body;
    if (url === 'https://sts.googleapis.com/v1/token') {
      assert.strictEqual(options.headers, undefined);
      const payload = JSON.parse(options.payload);
      assert.strictEqual(payload.audience, properties.get('WIF_PROVIDER'));
      assert.strictEqual(payload.subjectToken, 'identity-token');
      assert.strictEqual(payload.subjectTokenType, 'urn:ietf:params:oauth:token-type:jwt');
      body = {access_token:'federated-token'};
    } else if (url === 'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/inbox%40project.iam.gserviceaccount.com:generateAccessToken') {
      assert.strictEqual(options.headers.Authorization, 'Bearer federated-token');
      assert.deepStrictEqual(JSON.parse(options.payload).scope, ['https://www.googleapis.com/auth/spreadsheets']);
      body = {accessToken:'service-token-value', expireTime:new Date(Date.now()+3600000).toISOString()};
    }
    else {
      assert(url.startsWith('https://sheets.googleapis.com/v4/spreadsheets/private_sheet_identifier_12345/values/'));
      assert.strictEqual(options.headers.Authorization, 'Bearer service-token-value');
      if (url.includes('A1%3AH1')) body = {values:[rows[0]]};
      else if (url.includes('H2%3AH')) body = {values:rows.slice(1).map(row => [row[7]])};
      else { assert(url.includes('valueInputOption=RAW')); rows.push(JSON.parse(options.payload).values[0]); body = {}; }
    }
    return {getResponseCode: () => 200, getContentText: () => JSON.stringify(body)};
  }},
  Utilities: {formatDate: () => '2026-01-01',
    base64EncodeWebSafe: value => Buffer.from(value).toString('base64url'),
    computeRsaSha256Signature: () => 'signature'},
  ContentService: {MimeType: {JAVASCRIPT:'js'}, createTextOutput: text => ({text, setMimeType() { return this; }})},
  console,
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
context.verifySetup();
assert.strictEqual(requests.length, 3);
const data = {id:'a'.repeat(32), dataset:'coral_conversation', clip:'coral_conversation_00001.wav',
  model:'RyeAI/ekko-v1-tiny', original:'A reference', suggested:'=A suggested correction', website:''};
const post = data => context.doPost({postData:{contents:JSON.stringify(data)}});
post(data);
assert.strictEqual(rows.length, 2);
assert.strictEqual(rows[1][4], "'=A suggested correction");
assert.strictEqual(rows[1][6], 'Pending');
assert(rows[1][5].includes('model=RyeAI%2Fekko-v1-tiny'));
assert.strictEqual(cache.get('correction:'+data.id), 'ok');
const callback = '__correction_' + data.id;
assert.strictEqual(context.doGet({parameter:{id:data.id, callback}}).text, callback + '("ok");');
assert.strictEqual(context.doGet({parameter:{id:data.id, callback:'alert(1)'}}).text, 'Invalid request');
post(data);
cache.delete('correction:'+data.id);
post(data);
assert.strictEqual(rows.length, 2, 'duplicate retries must not insert another row');
const bad = {...data, id:'b'.repeat(32), website:'spam'};
post(bad);
assert.strictEqual(rows.length, 2);
assert.strictEqual(cache.get('correction:'+bad.id), 'error');
const badClip = {...data, id:'c'.repeat(32), clip:'../sensitive.wav'};
post(badClip);
assert.strictEqual(rows.length, 2);
properties.set('corrections_count', '2000');
const limited = {...data, id:'d'.repeat(32)};
post(limited);
assert.strictEqual(cache.get('correction:'+limited.id), 'error');
assert.strictEqual(rows.length, 2);
// Failed federation must not append a row or reveal its tokens via status.
properties.set('corrections_count', '0');
cache.delete('service-token');
context.UrlFetchApp.fetch = () => ({getResponseCode: () => 403, getContentText: () => 'credential details'});
const failed = {...data, id:'e'.repeat(32)};
post(failed);
assert.strictEqual(cache.get('correction:'+failed.id), 'error');
assert.strictEqual(rows.length, 2);
assert.strictEqual(context.doGet({parameter:{id:failed.id, callback:'__correction_'+failed.id}}).text,
  '__correction_' + failed.id + '("error");');
"""
    subprocess.run(['node', '-e', test, str(script_path)], check=True, timeout=30)


def test_inbox_has_no_personal_drive_permission():
    import json

    manifest = Path(__file__).resolve().parents[1] / 'scripts/correction_inbox/appsscript.json'
    assert json.loads(manifest.read_text())['oauthScopes'] == ['openid', 'https://www.googleapis.com/auth/script.external_request']


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is required')
def test_browser_confirms_save_before_reporting_success():
    index = Path(__file__).resolve().parents[1] / 'huggingface/space/index.html'
    test = r"""
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const start = html.indexOf('function correctionStatus(');
const end = html.indexOf('function referenceIssueURL(', start);
assert(start >= 0 && end > start);
let statuses = [], posts = [];
const listeners = new Set();
const window = {
  addEventListener: (type, fn) => { assert.strictEqual(type, 'message'); listeners.add(fn); },
  removeEventListener: (type, fn) => { assert.strictEqual(type, 'message'); listeners.delete(fn); },
};
const context = {
  URL, window, clearTimeout,
  setTimeout: (cb, ms) => ms === 1000 ? queueMicrotask(cb) : setTimeout(cb, ms),
  fetch: async (url, options) => { posts.push({url, options}); },
  document: {createElement: tag => {
    assert.strictEqual(tag, 'iframe');
    return {contentWindow:{}, setAttribute(name, value) { this[name] = value; }, remove() {}};
  }, body: {appendChild: frame => {
    assert.strictEqual(frame.sandbox, 'allow-scripts');
    assert(frame.srcdoc.includes('parent.postMessage'));
    queueMicrotask(() => {
      for (const listener of listeners) listener({source:{}, data:{type:'correction-status', status:'ok'}});
      for (const listener of listeners) listener({source:frame.contentWindow, data:{type:'correction-status', status:statuses.shift()}});
    });
  }}},
};
vm.runInNewContext(html.slice(start, end) + '\nwindow.submit = submitCorrection;', context);
(async () => {
  const payload = {id:'a'.repeat(32), suggested:'changed'};
  statuses = ['pending', 'ok'];
  await window.submit('https://example.test/exec', payload);
  assert.strictEqual(posts[0].options.mode, 'no-cors');
  assert.strictEqual(JSON.parse(posts[0].options.body).id, payload.id);
  assert.strictEqual(listeners.size, 0);
  statuses = ['error'];
  await assert.rejects(window.submit('https://example.test/exec', payload));
  statuses = Array(10).fill('pending');
  await assert.rejects(window.submit('https://example.test/exec', payload));
})().catch(e => { console.error(e); process.exitCode = 1; });
"""
    subprocess.run(['node', '-e', test, str(index)], check=True, timeout=30)
