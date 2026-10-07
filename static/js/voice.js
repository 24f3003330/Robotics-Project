// ============================================================================
// VOICE PAGE - live microphone emotion (tone of voice, analysed in Python),
// plus speaker gender from the same microphone stream.
// Never calls any camera endpoint: the webcam is untouched by this page.
//
// The server captures the microphone with sounddevice and classifies every
// 2-4 s of speech with HuBERT in its own worker thread. This page only starts
// and stops that pipeline and polls /voice/status every 500 ms, rendering
// every new prediction. Leaving the page stops the microphone (pagehide
// beacon, with a server-side watchdog as the fallback).
// ============================================================================
const VOICE_EMOTIONS = ['happy', 'angry', 'sad', 'neutral'];
const MIC_UNAVAILABLE = 'Microphone access is unavailable. Please grant microphone permission and try again.';
const MIC_STATE = {
  stopped:       { label: 'Stopped',       cls: 'pill-idle' },
  starting:      { label: 'Starting',      cls: 'pill-warn' },
  loading_model: { label: 'Loading model', cls: 'pill-warn' },
  listening:     { label: 'Listening',     cls: 'pill-ok' },
  error:         { label: 'Error',         cls: 'pill-bad' },
};

let lastPredictionIndex = 0;
let startError = null;       // a failed /voice/start, shown until the next attempt

function setVoiceButtons(running) {
  show('btnStartVoice', !running);
  show('btnStopVoice', running);
}

function renderVoiceError(s) {
  // A running stream that delivers only silence is a blocked microphone
  // (macOS permission for the app running Python), not a working one.
  const detail = startError || s.error || (s.running && s.mic_silent ? s.warning : '') || '';
  if (!detail) { show('voiceErrorBox', false); return; }
  const modelProblem = /model|pytorch|torch/i.test(detail) && !/microphone/i.test(detail);
  text('voiceErrorText', modelProblem ? 'The voice emotion model could not be loaded.' : MIC_UNAVAILABLE);
  text('voiceErrorDetail', detail);
  show('voiceErrorBox', true);
}

async function loadVoiceDevices() {
  const select = document.getElementById('voiceDevice');
  const previous = select.value;
  try {
    const data = await getJSON('/voice/devices');
    select.innerHTML = '<option value="">System default</option>';
    (data.devices || []).forEach(d => {
      const opt = document.createElement('option');
      opt.value = String(d.index);
      opt.textContent = `[${d.index}] ${d.name} (${d.hostapi})${d.is_default ? ' - default' : ''}`;
      select.appendChild(opt);
    });
    const wanted = (data.selected !== null && data.selected !== undefined) ? String(data.selected) : previous;
    if ([...select.options].some(o => o.value === wanted)) select.value = wanted;
    if (!data.ok) {
      startError = data.error || 'No microphone could be listed.';
      renderVoiceError({});
    }
  } catch (e) { noteNetworkError('/voice/devices', e); }
}

async function startVoice() {
  startError = null;
  show('voiceErrorBox', false);
  document.getElementById('btnStartVoice').disabled = true;
  text('voiceMic', 'Starting...');
  try {
    const data = await postJSON('/voice/start', { device: document.getElementById('voiceDevice').value });
    if (!data.ok) startError = data.error || data.message || 'Could not start the microphone.';
  } catch (e) {
    startError = 'Could not reach the server: ' + e.message;
  } finally {
    document.getElementById('btnStartVoice').disabled = false;
  }
  pollVoiceStatus();
}

async function stopVoice() {
  try {
    await postJSON('/voice/stop');
  } catch (e) {
    noteNetworkError('/voice/stop', e);
  }
  pollVoiceStatus();
}

function renderEnsemble(s) {
  const info = s.ensemble || {};
  // A single model reports itself as one member: nothing to compare, stay hidden.
  if (!info.members || info.members.length < 2) { show('voiceEnsembleBox', false); return; }
  show('voiceEnsembleBox', true);
  const agree = (s.member_agreement === null || s.member_agreement === undefined)
    ? '-' : Math.round(s.member_agreement * 100) + '% of models agree';
  text('voiceEnsembleSummary', `${info.members.length} models, ${info.combiner} - ${agree}`);
  const members = s.member_probabilities || {};
  const top = s.member_top || {};
  const dissenting = s.member_dissenting || [];
  const container = document.getElementById('voiceEnsembleMembers');
  container.textContent = '';
  Object.keys(members).sort().forEach(name => {
    const probs = members[name] || {};
    const best = top[name] || Object.keys(probs).sort((a, b) => probs[b] - probs[a])[0];
    const all = VOICE_EMOTIONS.map(e => `${e.slice(0, 3)} ${Math.round((probs[e] || 0) * 100)}`).join('  ');
    const row = document.createElement('div');
    row.textContent = `${name}: ${String(best || '-').toUpperCase()} ` +
      `${Math.round((probs[best] || 0) * 100)}%  [${all}]` +
      (dissenting.indexOf(name) !== -1 ? '  ⚠ disagrees' : '');
    container.appendChild(row);
  });
}

// Speaker gender shares the microphone and VAD with the emotion pipeline
// (see voice_gender.py). It is display-only: it never feeds the final emotion.
const GENDER_STATE = {
  stopped:       { label: 'Stopped',       cls: 'pill-idle' },
  loading_model: { label: 'Loading model', cls: 'pill-warn' },
  listening:     { label: 'Listening',     cls: 'pill-ok' },
  unavailable:   { label: 'Unavailable',   cls: 'pill-bad' },
};

function renderGender(g, micRunning) {
  g = g || {};
  const running = micRunning && !!g.running;
  const state = running ? (g.model_error ? 'unavailable' : g.state) : 'stopped';
  const display = GENDER_STATE[state] || { label: state || '-', cls: '' };
  const pill = document.getElementById('genderPill');
  pill.textContent = '● ' + display.label;
  pill.className = 'pill ' + display.cls;

  let value;
  if (running && g.gender) value = g.gender.toUpperCase();
  else if (running && g.model_error) value = 'Unavailable';
  else if (running && state === 'loading_model') value = 'Loading model...';
  else if (running) value = 'Listening... speak to detect';
  else value = '-';
  text('genderValue', value);
  text('genderEmoji', running && g.gender === 'male' ? '👨' : running && g.gender === 'female' ? '👩' : '🗣️');
  text('genderConfidence', running && g.gender ? (g.confidence * 100).toFixed(1) + '%' : '-');
  text('genderSegments', running ? (g.segments_classified || 0) : 0);
  text('genderBuffered', running ? (g.speech_seconds_buffered || 0).toFixed(1) + 's' : '-');

  const probs = (running && g.gender && g.probabilities) || {};
  ['male', 'female'].forEach(k => {
    const pct = Math.round((probs[k] || 0) * 100);
    document.getElementById('gbar-' + k).style.width = pct + '%';
    text('gpct-' + k, pct + '%');
  });
  document.getElementById('genderBars').classList.toggle('idle', !(running && g.gender));

  const warning = running ? (g.model_error || g.error || '') : '';
  text('genderWarning', warning);
  show('genderWarning', !!warning);
}

// Estimated age GROUP (voice_age/): a separate model fed by the same microphone
// stream. Display-only; never feeds the final emotion, music or the robot.
const AGE_STATE = {
  stopped:             { label: 'Stopped',             cls: 'pill-idle' },
  model_not_trained:   { label: 'Not trained',         cls: 'pill-idle' },
  loading_model:       { label: 'Loading model',       cls: 'pill-warn' },
  listening:           { label: 'Listening',           cls: 'pill-ok' },
  insufficient_speech: { label: 'Insufficient speech', cls: 'pill-warn' },
  estimated:           { label: 'Estimated',           cls: 'pill-ok' },
  unavailable:         { label: 'Unavailable',         cls: 'pill-bad' },
};

function renderAge(a, micRunning) {
  a = a || {};
  const running = micRunning && !!a.running;
  let state = a.state || 'stopped';
  if (state !== 'model_not_trained' && state !== 'unavailable' && !running) state = 'stopped';
  const display = AGE_STATE[state] || { label: state, cls: '' };
  const pill = document.getElementById('agePill');
  pill.textContent = '● ' + display.label;
  pill.className = 'pill ' + display.cls;

  const has = running && !!a.age_group;
  let value;
  if (state === 'model_not_trained') value = 'Voice age model not trained yet.';
  else if (state === 'unavailable') value = 'Age estimation unavailable';
  else if (has) value = a.age_group;
  else if (running && state === 'insufficient_speech') value = 'Insufficient speech';
  else if (running) value = 'Listening... speak for a few seconds';
  else value = '-';
  text('ageValue', value);
  text('ageConfidence', has ? Math.round(a.confidence * 100) + '%' : '-');
  text('ageWindows', running ? (a.windows_classified || 0) : 0);
  text('ageBuffered', running ? (a.speech_seconds_buffered || 0).toFixed(1) + 's' : '-');

  const bars = document.getElementById('ageBars');
  const probs = has ? (a.probabilities || {}) : {};
  const names = Object.keys(a.probabilities || {});
  if (bars.dataset.names !== names.join('|')) {          // build rows once per class list
    bars.innerHTML = '';
    names.forEach((n, i) => {
      const row = document.createElement('div');
      row.className = 'voice-bar-row age-bar-row';
      const label = document.createElement('span');
      label.textContent = n;
      const track = document.createElement('div');
      track.className = 'voice-bar-track';
      const fill = document.createElement('div');
      fill.className = 'voice-bar-fill age-fill';
      fill.dataset.age = n;
      track.appendChild(fill);
      const pct = document.createElement('span');
      pct.className = 'voice-bar-pct';
      pct.dataset.agePct = n;
      row.append(label, track, pct);
      bars.appendChild(row);
    });
    bars.dataset.names = names.join('|');
  }
  bars.querySelectorAll('[data-age]').forEach(el => {
    el.style.width = Math.round((probs[el.dataset.age] || 0) * 100) + '%';
  });
  bars.querySelectorAll('[data-age-pct]').forEach(el => {
    el.textContent = Math.round((probs[el.dataset.agePct] || 0) * 100) + '%';
  });
  bars.classList.toggle('idle', !has);

  const warning = state === 'model_not_trained' ? (a.message || '')
    : state === 'unavailable' ? (a.message || a.error || '') : (running ? (a.error || '') : '');
  text('ageWarning', warning);
  show('ageWarning', !!warning);
}

function renderVoiceStatus(s) {
  const running = !!s.running;
  const probs = s.probabilities || {};
  VOICE_EMOTIONS.forEach(e => {
    const pct = Math.round((probs[e] || 0) * 100);
    const bar = document.getElementById('bar-' + e);
    if (bar) bar.style.width = pct + '%';
    text('pct-' + e, pct + '%');
  });
  document.getElementById('voiceBars').classList.toggle('idle', !(running && s.speech_detected));

  // Current emotion = the smoothed result of the latest speech windows. It
  // changes with every prediction, and clears after a long silence.
  let current;
  if (s.emotion) current = s.emotion.toUpperCase();
  else if (running && s.status === 'listening') current = 'Listening... speak to detect';
  else if (running) current = 'Starting...';
  else current = '-';
  text('voiceEmotion', current);
  text('voiceEmoji', EMOTION_EMOJI[s.emotion] || '🎙️');
  text('voiceConfidence', s.emotion ? Math.round(s.confidence * 100) + '%' : '-');
  text('voiceSpeech', !running ? '-' : s.speech_detected ? 'detected' : 'silence');

  // Flash the readout on every NEW prediction so continuous updating is visible.
  const index = s.prediction_index || 0;
  if (index > lastPredictionIndex) {
    const box = document.getElementById('voiceDisplay');
    box.classList.remove('flash');
    void box.offsetWidth;                // restart the animation
    box.classList.add('flash');
  }
  lastPredictionIndex = index;

  text('voiceStable', s.voice_stable_emotion ? s.voice_stable_emotion.toUpperCase() : '-');
  text('voiceStreak',
       `${(s.stability_count || 0).toFixed(1)}s/${(s.stability_required || 2).toFixed(1)}s`);
  text('voicePredictions', s.inference_count || 0);
  if (s.prediction_timestamp && s.server_time && running) {
    const age = Math.max(0, s.server_time - s.prediction_timestamp);
    text('voicePredictionAge', age < 1 ? 'just now' : `${age.toFixed(age < 10 ? 1 : 0)}s ago`);
  } else {
    text('voicePredictionAge', running ? 'waiting for speech' : '-');
  }
  text('voiceBuffered', running ? `${(s.buffered_speech_seconds || 0).toFixed(1)}s speech / ` +
                                  `${(s.buffered_seconds || 0).toFixed(1)}s audio` : '-');

  const state = (running && s.mic_silent) ? { label: 'No audio', cls: 'pill-bad' }
              : MIC_STATE[s.status] || { label: s.status || '-', cls: '' };
  const pill = document.getElementById('voiceMicPill');
  pill.textContent = '● ' + state.label;
  pill.className = 'pill ' + state.cls;
  let mic = state.label;
  if (running && s.input_device) mic += ' - ' + s.input_device;
  text('voiceMic', mic);
  text('voiceModelState', s.model_loading ? 'Loading emotion model (first time can take a while)...'
                          : s.model_loaded ? 'Emotion model ready' : '');

  text('voiceRobotMode', s.robot_trigger_enabled ? 'ENABLED (robot will move)' : 'Simulation only');
  text('voiceWarning', s.warning || '');
  show('voiceWarning', !!s.warning && !s.mic_silent);
  if (running) startError = null;
  renderVoiceError(s);
  renderEnsemble(s);
  setVoiceButtons(running);
  renderRobotPills(s.robot_connected, s.robot_busy);

  applyEmotionState({
    state_seq: s.state_seq,
    emotion: s.final_emotion,
    source: s.final_source,
    confidence: s.final_confidence,
    pose_key: s.robot_pose_key,
    background: s.background,
    song: s.song,
    last_action: s.last_robot_action,
  });

  // Last, and guarded: a gender display problem must never stop emotion updates.
  try { renderGender(s.gender, running); } catch (e) { console.log('[GENDER] render failed', e); }
  try { renderAge(s.age, running); } catch (e) { console.log('[AGE] render failed', e); }
}

let voiceBusy = false;
let voiceSeq = 0;
let voiceRendered = 0;

async function pollVoiceStatus() {
  if (voiceBusy) return;
  voiceBusy = true;
  const seq = ++voiceSeq;
  try {
    const data = await getJSON('/voice/status');
    noteNetworkOk();
    if (seq <= voiceRendered) return;    // a newer response already rendered
    voiceRendered = seq;
    renderVoiceStatus(data);
  } catch (e) {
    noteNetworkError('/voice/status', e);
  } finally {
    voiceBusy = false;
  }
}
refreshPageStatus = pollVoiceStatus;

// 500 ms while visible. While hidden the microphone keeps running, and a slow
// poll doubles as the heartbeat that keeps the server-side watchdog satisfied.
let voiceTimer = null;
function schedulePolling(intervalMs) {
  if (voiceTimer) clearInterval(voiceTimer);
  voiceTimer = setInterval(pollVoiceStatus, intervalMs);
}

document.addEventListener('visibilitychange', () => {
  schedulePolling(document.hidden ? 5000 : 500);
  if (!document.hidden) pollVoiceStatus();
});

// Leaving the page stops the microphone and the processing thread.
window.addEventListener('pagehide', () => {
  if (voiceTimer) { clearInterval(voiceTimer); voiceTimer = null; }
  beacon('/voice/stop');
});
window.addEventListener('pageshow', (event) => {
  if (event.persisted) { schedulePolling(500); pollVoiceStatus(); }
});

schedulePolling(500);
pollVoiceStatus();
loadVoiceDevices();
