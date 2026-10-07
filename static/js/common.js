// ============================================================================
// Shared by the camera and voice pages: helpers, the fused-emotion store, the
// mood-music player, manual emotion trigger and robot controls. Page-specific
// pipelines live in camera.js and voice.js; neither touches the other's
// hardware.
// ============================================================================

const EMOTION_EMOJI = {
  happy: '😊', angry: '😠', sad: '😢', neutral: '😐',
  surprise: '😮', fear: '😨', disgust: '🤢', contempt: '😒',
};

function text(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function show(id, visible) {
  const el = document.getElementById(id);
  if (el) el.hidden = !visible;
}

function showRobotOutput(data) {
  const el = document.getElementById('robotOutput');
  if (el) el.textContent = JSON.stringify(data, null, 2);
}

// Fire-and-forget POST that survives the page being unloaded (pagehide).
function beacon(url) {
  try {
    if (navigator.sendBeacon && navigator.sendBeacon(url)) return;
  } catch (e) { /* fall through */ }
  fetch(url, { method: 'POST', keepalive: true }).catch(() => {});
}

// ---- network error surfacing (never fail silently) -------------------------
let consecutiveNetworkErrors = 0;

function noteNetworkOk() {
  consecutiveNetworkErrors = 0;
  show('connectionError', false);
}

function noteNetworkError(what, err) {
  consecutiveNetworkErrors++;
  console.log('[NET]', what, err);
  if (consecutiveNetworkErrors >= 3) {
    text('connectionError', 'Lost contact with the server (' + what + '). Is the Python app still running?');
    show('connectionError', true);
  }
}

async function getJSON(url, options) {
  const res = await fetch(url, options);
  const body = await res.text();
  let data;
  try {
    data = JSON.parse(body);
  } catch (e) {
    throw new Error('Server sent a malformed response (HTTP ' + res.status + ')');
  }
  if (!res.ok && data.error === undefined) {
    throw new Error('Server error (HTTP ' + res.status + ')');
  }
  return data;
}

function postJSON(url, payload) {
  return getJSON(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
}

// ============================================================================
// MOOD MUSIC - one player, one audible tab
// ----------------------------------------------------------------------------
// Only one browser tab is allowed to make sound (localStorage leader lock), so
// opening two pages cannot double the music. Playback is only started for a
// reason WE caused (a new song, or just becoming leader); a user's own pause
// on the native controls is respected and never auto-resumed.
// ============================================================================
const Music = (function () {
  const el = document.getElementById('musicPlayer');
  const TAB_ID = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random());
  const LEADER_KEY = 'emotionAppAudioLeader';
  const LEADER_TTL_MS = 3000;
  let wasLeader = false;

  function claimLeaderIfFree() {
    try {
      const raw = localStorage.getItem(LEADER_KEY);
      const leader = raw ? JSON.parse(raw) : null;
      if (!leader || (Date.now() - leader.ts) > LEADER_TTL_MS || leader.id === TAB_ID) {
        localStorage.setItem(LEADER_KEY, JSON.stringify({ id: TAB_ID, ts: Date.now() }));
      }
    } catch (e) { /* private mode: every tab behaves as leader */ }
  }

  function isLeader() {
    try {
      const raw = localStorage.getItem(LEADER_KEY);
      return raw ? JSON.parse(raw).id === TAB_ID : true;
    } catch (e) { return true; }
  }

  function play(src) {
    if (!el) return;
    claimLeaderIfFree();
    const leader = isLeader();
    const justBecameLeader = leader && !wasLeader;
    wasLeader = leader;
    if (!leader) { el.pause(); return; }
    const changed = src && el.getAttribute('src') !== src;
    if (changed) el.setAttribute('src', src);
    if (changed || justBecameLeader) {
      el.play().catch(() => { /* autoplay blocked until the user interacts */ });
    }
  }

  function stop() {
    if (el) el.pause();
  }

  if (el) {
    el.addEventListener('play', () => { if (!isLeader()) el.pause(); });
    el.addEventListener('error', () => {
      const src = el.getAttribute('src');
      text('musicError', src ? 'Could not play ' + src : '');
    });
    claimLeaderIfFree();
    setInterval(() => { claimLeaderIfFree(); if (!isLeader() && !el.paused) el.pause(); }, 1000);
    window.addEventListener('pagehide', () => {
      try { if (isLeader()) localStorage.removeItem(LEADER_KEY); } catch (e) {}
      el.pause();
    });
  }

  return { play, stop };
})();

// ============================================================================
// EMOTION STORE - the fused ("final") emotion shown in the output card.
// Responses carry the server's state_seq; an older snapshot never overwrites
// a newer one, whatever order responses arrive in.
// ============================================================================
const EmotionState = {
  emotion: null,
  source: null,
  confidence: 0,
  stateSeq: -1,
  poseKey: '-',
  background: '#ffffff',
  song: '',
  lastAction: null,
};

function applyEmotionState(payload) {
  const seq = Number(payload.state_seq);
  if (!Number.isFinite(seq)) return false;
  if (seq < EmotionState.stateSeq) return false;
  EmotionState.stateSeq = seq;
  EmotionState.emotion = payload.emotion || null;
  EmotionState.source = payload.source || null;
  EmotionState.confidence = Number(payload.confidence) || 0;
  if (payload.pose_key !== undefined) EmotionState.poseKey = payload.pose_key;
  if (payload.background !== undefined) EmotionState.background = payload.background;
  if (payload.song !== undefined) EmotionState.song = payload.song;
  if (payload.last_action !== undefined) EmotionState.lastAction = payload.last_action;
  renderEmotion();
  return true;
}

function renderEmotion() {
  const e = EmotionState.emotion;
  const label = e ? e.toUpperCase() : null;
  const isDefault = EmotionState.source === 'default';
  const confText = (!e || isDefault)
    ? (e ? 'n/a (nothing detected yet)' : '-')
    : Math.round(EmotionState.confidence * 100) + '%';

  text('finalEmotionStatus', 'Final Emotion: ' + (label || '-'));
  text('finalSource', EmotionState.source || '-');
  text('finalConfidence', confText);
  text('poseKey', 'Pose: ' + (EmotionState.poseKey || '-'));
  text('finalLastAction', EmotionState.lastAction ? EmotionState.lastAction.action : '-');

  if (e && !isDefault) {
    document.body.style.backgroundColor = EmotionState.background || '';
    if (EmotionState.song) Music.play(EmotionState.song); else Music.stop();
  } else {
    document.body.style.backgroundColor = '';
  }
}

function renderRobotPills(connected, busy) {
  text('robotConnected', 'Robot: ' + (connected ? 'Connected' : 'Not connected'));
  text('robotBusy', 'Busy: ' + (busy ? 'Yes' : 'No'));
}

// Each page sets this to its own status poll so a button press shows at once.
let refreshPageStatus = () => {};

// ============================================================================
// MANUAL EMOTION TRIGGER
// ============================================================================
async function setEmotion(emotion) {
  try {
    showRobotOutput(await getJSON('/set_emotion/' + encodeURIComponent(emotion)));
    refreshPageStatus();
  } catch (e) { noteNetworkError('/set_emotion', e); }
}

async function clearManual() {
  try { showRobotOutput(await getJSON('/clear_manual')); refreshPageStatus(); }
  catch (e) { noteNetworkError('/clear_manual', e); }
}

// ============================================================================
// ROBOT
// ============================================================================
async function scanRobot() {
  try {
    const data = await getJSON('/robot/scan');
    showRobotOutput(data);
    const select = document.getElementById('robotDevice');
    select.innerHTML = '';
    (data.devices || []).forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.address;
      opt.textContent = `${d.name} | ${d.address} | RSSI:${d.rssi}`;
      select.appendChild(opt);
    });
  } catch (e) { noteNetworkError('/robot/scan', e); }
}

async function connectRobot() {
  const address = document.getElementById('robotDevice').value;
  const writeIndex = parseInt(document.getElementById('writeIndex').value || '0');
  try {
    showRobotOutput(await postJSON('/robot/connect', { address: address, write_index: writeIndex }));
  } catch (e) { noteNetworkError('/robot/connect', e); }
}

async function disconnectRobot() {
  try { showRobotOutput(await getJSON('/robot/disconnect')); }
  catch (e) { noteNetworkError('/robot/disconnect', e); }
}

async function robotHome() {
  try { showRobotOutput(await getJSON('/robot/home')); }
  catch (e) { noteNetworkError('/robot/home', e); }
}

async function robotStop() {
  try { showRobotOutput(await getJSON('/robot/stop')); }
  catch (e) { noteNetworkError('/robot/stop', e); }
}

async function runRobot(emotion) {
  try { showRobotOutput(await getJSON('/robot/run/' + encodeURIComponent(emotion))); }
  catch (e) { noteNetworkError('/robot/run', e); }
}
