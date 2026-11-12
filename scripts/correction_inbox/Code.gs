// The service account is granted access only to the review spreadsheet.
const CORRECTION_SHEET = 'Suggestions';
const DAILY_LIMIT = 2000;
const DATASET_IDS = {
  coral_conversation: /^coral_conversation_\d{5}\.wav$/,
  coral_read_aloud: /^coral_read_aloud_\d{5}\.wav$/,
  cv17_da: /^common_voice_da_\d+\.wav$/,
  fleurs_da: /^fleurs_da_\d{5}\.wav$/,
  ftspeech: /^ftspeech_\d{5}\.wav$/,
};

function configuration() {
  const properties = PropertiesService.getScriptProperties();
  const sheetId = properties.getProperty('CORRECTIONS_SHEET_ID') || '';
  const serviceAccount = properties.getProperty('SERVICE_ACCOUNT_EMAIL') || '';
  const provider = properties.getProperty('WIF_PROVIDER') || '';
  if (!/^[a-zA-Z0-9_-]{20,100}$/.test(sheetId) ||
      !/^[a-zA-Z0-9._-]+@[a-zA-Z0-9-]+\.iam\.gserviceaccount\.com$/.test(serviceAccount) ||
      !/^\/\/iam\.googleapis\.com\/projects\/\d+\/locations\/global\/workloadIdentityPools\/[a-z0-9-]+\/providers\/[a-z0-9-]+$/.test(provider)) {
    throw new Error('Inbox not configured');
  }
  return {sheetId, serviceAccount, provider};
}

function accessToken(config) {
  const cache = CacheService.getScriptCache();
  const cached = cache.get('service-token');
  if (cached) return cached;
  const identityToken = ScriptApp.getIdentityToken();
  if (!identityToken) throw new Error('Script identity unavailable');
  const exchange = UrlFetchApp.fetch('https://sts.googleapis.com/v1/token', {
    method: 'post', contentType: 'application/json', muteHttpExceptions: true,
    payload: JSON.stringify({audience: config.provider,
      grantType: 'urn:ietf:params:oauth:grant-type:token-exchange',
      requestedTokenType: 'urn:ietf:params:oauth:token-type:access_token',
      subjectTokenType: 'urn:ietf:params:oauth:token-type:jwt',
      scope: 'https://www.googleapis.com/auth/cloud-platform', subjectToken: identityToken}),
  });
  if (exchange.getResponseCode() !== 200) throw new Error('Identity federation failed');
  const federated = JSON.parse(exchange.getContentText());
  if (typeof federated.access_token !== 'string' || !federated.access_token) throw new Error('Identity federation failed');
  const serviceUrl = `https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${encodeURIComponent(config.serviceAccount)}:generateAccessToken`;
  const response = UrlFetchApp.fetch(serviceUrl, {
    method: 'post', contentType: 'application/json', muteHttpExceptions: true,
    headers: {Authorization: `Bearer ${federated.access_token}`},
    payload: JSON.stringify({scope: ['https://www.googleapis.com/auth/spreadsheets'], lifetime: '3600s'}),
  });
  if (response.getResponseCode() !== 200) throw new Error('Service account authorization failed');
  const result = JSON.parse(response.getContentText());
  const seconds = Math.floor((Date.parse(result.expireTime) - Date.now()) / 1000);
  if (typeof result.accessToken !== 'string' || !result.accessToken || !(seconds > 60)) {
    throw new Error('Service account authorization failed');
  }
  cache.put('service-token', result.accessToken, Math.min(3000, seconds - 60));
  return result.accessToken;
}

// Run only from the editor to configure the provider. Never log the token itself.
function identityClaimsForSetup() {
  const token = ScriptApp.getIdentityToken();
  if (!token) throw new Error('Identity token unavailable; check openid authorization');
  const encoded = token.split('.')[1] || '';
  const claim = JSON.parse(Utilities.newBlob(Utilities.base64DecodeWebSafe(encoded.padEnd(Math.ceil(encoded.length / 4) * 4, '='))).getDataAsString());
  if (claim.iss !== 'https://accounts.google.com' || !/^[0-9]+$/.test(claim.sub || '') ||
      typeof claim.aud !== 'string' || !claim.aud.endsWith('.apps.googleusercontent.com')) {
    throw new Error('Unexpected identity claims');
  }
  console.log(`Issuer: ${claim.iss}\nAudience: ${claim.aud}\nSubject: ${claim.sub}`);
}

function sheetValues(config, range, row) {
  const url = `https://sheets.googleapis.com/v4/spreadsheets/${encodeURIComponent(config.sheetId)}/values/${encodeURIComponent(range)}`;
  const options = {headers: {Authorization: `Bearer ${accessToken(config)}`}, muteHttpExceptions: true};
  if (row) {
    options.method = 'post';
    options.contentType = 'application/json';
    options.payload = JSON.stringify({values: [row]});
  }
  const response = UrlFetchApp.fetch(row ? `${url}:append?valueInputOption=RAW&insertDataOption=INSERT_ROWS` : url, options);
  if (response.getResponseCode() < 200 || response.getResponseCode() >= 300) throw new Error('Sheet request failed');
  return JSON.parse(response.getContentText());
}

function verifySetup() {
  const config = configuration();
  const headers = ['Submitted (UTC)', 'Dataset', 'Clip ID', 'Original reference',
    'Suggested reference', 'Sample link', 'Review status', 'Submission ID'];
  const actual = sheetValues(config, `${CORRECTION_SHEET}!A1:H1`).values?.[0] || [];
  if (headers.some((value, index) => actual[index] !== value)) throw new Error('Review sheet headers do not match');
  console.log('Review sheet is ready.');
}

function statusOutput(callback, status) {
  return ContentService.createTextOutput(`${callback}(${JSON.stringify(status)});`)
    .setMimeType(ContentService.MimeType.JAVASCRIPT);
}

function doGet(e) {
  const id = String(e.parameter.id || '');
  const callback = String(e.parameter.callback || '');
  if (!/^[a-f0-9]{32}$/.test(id) || callback !== `__correction_${id}`) {
    return ContentService.createTextOutput('Invalid request');
  }
  const status = CacheService.getScriptCache().get(`correction:${id}`) || 'pending';
  return statusOutput(callback, status);
}

function safeCell(value) {
  return /^[\s]*[=+\-@]/.test(value) ? `'${value}` : value;
}

function validateCorrection(data) {
  if (!data || typeof data !== 'object' || !/^[a-f0-9]{32}$/.test(data.id || '')) throw new Error('invalid');
  if (!Object.prototype.hasOwnProperty.call(DATASET_IDS, data.dataset) ||
      !DATASET_IDS[data.dataset].test(data.clip || '')) throw new Error('invalid');
  if (typeof data.model !== 'string' || !/^[a-zA-Z0-9._/-]{1,160}$/.test(data.model)) throw new Error('invalid');
  if (data.website) throw new Error('invalid'); // honeypot, not strong spam protection
  const original = typeof data.original === 'string' ? data.original.trim() : '';
  const suggested = typeof data.suggested === 'string' ? data.suggested.trim() : '';
  if (!original || !suggested || original.length > 2000 || suggested.length > 2000 || original === suggested) {
    throw new Error('invalid');
  }
  const sample = `https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard?view=compare&dataset=${encodeURIComponent(data.dataset)}&clip=${encodeURIComponent(data.clip)}&model=${encodeURIComponent(data.model)}#compare-transcripts`;
  return [new Date().toISOString(), data.dataset, data.clip, safeCell(original), safeCell(suggested), sample, 'Pending', data.id];
}

function doPost(e) {
  let id = '';
  const cache = CacheService.getScriptCache();
  try {
    if (!e || !e.postData || typeof e.postData.contents !== 'string' || e.postData.contents.length > 9000) throw new Error('invalid');
    const data = JSON.parse(e.postData.contents);
    id = typeof data.id === 'string' && /^[a-f0-9]{32}$/.test(data.id) ? data.id : '';
    const row = validateCorrection(data);
    const lock = LockService.getScriptLock();
    if (!lock.tryLock(5000)) throw new Error('busy');
    try {
      if (cache.get(`correction:${id}`) !== 'ok') {
        const properties = PropertiesService.getScriptProperties();
        const day = Utilities.formatDate(new Date(), 'UTC', 'yyyy-MM-dd');
        const count = properties.getProperty('corrections_date') === day
          ? Number(properties.getProperty('corrections_count') || 0) : 0;
        const config = configuration();
        const ids = sheetValues(config, `${CORRECTION_SHEET}!H2:H`).values || [];
        const exists = ids.some(row => row[0] === id);
        if (!exists) {
          if (count >= DAILY_LIMIT) throw new Error('limit');
          sheetValues(config, `${CORRECTION_SHEET}!A:H`, row);
          properties.setProperties({'corrections_date': day, 'corrections_count': String(count + 1)});
        }
        cache.put(`correction:${id}`, 'ok', 21600);
      }
    } finally { lock.releaseLock(); }
  } catch (error) {
    if (id && cache.get(`correction:${id}`) !== 'ok') cache.put(`correction:${id}`, 'error', 21600);
  }
  return ContentService.createTextOutput('');
}
