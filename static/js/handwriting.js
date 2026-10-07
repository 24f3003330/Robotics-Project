// ============================================================================
// HANDWRITING PAGE - canvas stroke capture and on-demand analysis.
// Never calls any camera or voice endpoint.
//
// Pointer Events cover mouse, trackpad, touch and stylus. Every sample is
// stored as canvas-relative CSS pixels (x right, y down) with its event time
// in ms; pressure/tilt are sent only when the device really reports them
// (a mouse's fixed 0.5 "pressure" is sent as null). In-air movement of a
// mouse or hovering stylus between strokes is kept as pen_down=false points;
// the server collapses them to the lift point if the model was trained on a
// pen that records nothing in the air.
// POST /handwriting/analyze {points: [...]} runs the Transformer on the server
// and returns {predictions, probabilities} for Anxiety / Depression / Stress.
// The separate 4-class PROTOTYPE (Happy / Neutral / Sad / Angry, seed data) is
// selected with the model switch and uses /handwriting/4class/*; its result is
// always shown with the prototype notice.
// On the 4-class tab a handwriting IMAGE can be uploaded instead of writing; it is
// sent to /handwriting/4class/analyze-image (never to the EMOTHAW model).
// Collect mode saves participant samples (anonymous writer id + selected label)
// via /handwriting/4class/collect; it never tells anyone how to write a label.
// ============================================================================
const Handwriting = (function () {
  const canvas = document.getElementById('hwCanvas');
  const ctx = canvas.getContext('2d');
  let points = [];            // {x, y, t, pen_down, pressure, altitude, azimuth}
  let activePointer = null;   // one writing pointer at a time (ignore extra fingers)
  let lastDown = null;        // last pen-down point, for drawing line segments
  const pointerTypes = new Set();

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#101828';
    redraw();
  }

  function segment(a, b) {
    const p = (b.pressure === null || b.pressure === undefined) ? 0.5 : b.pressure;
    ctx.lineWidth = 1.5 + 3 * p;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let prev = null;
    points.forEach(p => {
      if (p.pen_down && prev && prev.pen_down) segment(prev, p);
      else if (p.pen_down) segment(p, p);
      prev = p;
    });
  }

  function sample(e, penDown) {
    const rect = canvas.getBoundingClientRect();
    const isPen = e.pointerType === 'pen';
    const tilt = isPen && typeof e.altitudeAngle === 'number';
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      t: e.timeStamp,
      pen_down: penDown,
      // Mouse pressure is a fixed 0.5 while pressed - not a measurement.
      pressure: e.pointerType === 'mouse' ? null : (penDown ? e.pressure : 0),
      altitude: tilt ? e.altitudeAngle * 180 / Math.PI : null,
      azimuth: tilt ? e.azimuthAngle * 180 / Math.PI : null,
    };
  }

  function add(p) {
    if (points.length && p.t < points[points.length - 1].t) p.t = points[points.length - 1].t;
    if (p.pen_down) {
      segment(lastDown && points.length && points[points.length - 1].pen_down ? lastDown : p, p);
      lastDown = p;
    }
    points.push(p);
  }

  canvas.addEventListener('pointerdown', e => {
    if (activePointer !== null) return;
    activePointer = e.pointerId;
    pointerTypes.add(e.pointerType);
    canvas.setPointerCapture(e.pointerId);
    lastDown = null;
    add(sample(e, true));
    show('hwPlaceholder', false);
    e.preventDefault();
    Handwriting.onChange();
  });

  canvas.addEventListener('pointermove', e => {
    const writing = activePointer === e.pointerId;
    // Hover (in-air) samples: mouse or stylus only, and only once writing has started.
    if (!writing && (activePointer !== null || e.pointerType === 'touch' || !points.length)) return;
    const events = (typeof e.getCoalescedEvents === 'function' && e.getCoalescedEvents().length)
      ? e.getCoalescedEvents() : [e];
    events.forEach(ev => add(sample(ev, writing)));
    if (writing) e.preventDefault();
    Handwriting.onChange();
  });

  function end(e) {
    if (activePointer !== e.pointerId) return;
    add(sample(e, false));          // pen lifted here
    activePointer = null;
    lastDown = null;
    Handwriting.onChange();
  }
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  window.addEventListener('resize', resize);
  resize();

  return {
    points: () => points,
    pointerTypes: () => [...pointerTypes],
    clear() {
      points = []; lastDown = null; activePointer = null; pointerTypes.clear();
      redraw();
      show('hwPlaceholder', true);
      this.onChange();
    },
    onChange() {},
  };
})();

// ---- capture summary ---------------------------------------------------------
Handwriting.onChange = function () {
  const pts = Handwriting.points();
  let strokes = 0;
  pts.forEach((p, i) => { if (p.pen_down && (i === 0 || !pts[i - 1].pen_down)) strokes++; });
  const dur = pts.length > 1 ? (pts[pts.length - 1].t - pts[0].t) / 1000 : 0;
  const pressures = pts.filter(p => p.pen_down && typeof p.pressure === 'number').map(p => p.pressure);
  const varies = pressures.length > 5 && Math.max(...pressures) - Math.min(...pressures) > 1e-3;
  text('hwPoints', pts.length);
  text('hwStrokes', strokes);
  text('hwDuration', dur.toFixed(1) + 's');
  text('hwDevice', Handwriting.pointerTypes().join(', ') || '-');
  text('hwPressure', !pts.length ? '-' : varies ? 'yes' : 'not available');
};

function clearCanvas() {
  Handwriting.clear();
  show('hwError', false);
  show('hwDefaulted', false);
  resetTargets();
  resetFourClass();
  text('hwSummary', '-');
  text('hwMeta', '');
  text('hwEmoji', '✍️');
}

// ---- model switch: two separate experiments, two endpoints -----------------------
const HW_ENDPOINTS = {
  emothaw: { status: '/handwriting/status', analyze: '/handwriting/analyze' },
  '4class': { status: '/handwriting/4class/status', analyze: '/handwriting/4class/analyze' },
};
let hwMode = null;                               // set by the initial setHwMode() at the bottom
const HW_MODES = ['emothaw', '4class', 'collect'];

function setHwMode(mode) {
  if (!HW_MODES.includes(mode) || mode === hwMode) return;
  hwMode = mode;
  document.querySelectorAll('.hw-mode').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
    b.setAttribute('aria-selected', b.dataset.mode === mode ? 'true' : 'false');
  });
  show('hwEmothawPanel', mode === 'emothaw');
  show('hw4Panel', mode === '4class');
  show('hwCollectPanel', mode === 'collect');
  if (mode !== '4class' && hwImageFile) removeImage();       // images are for the 4-class prototype only
  show('hwUploadBar', mode !== 'collect');                   // collection needs real canvas strokes
  text('hwUploadNote', mode === '4class' ? 'PNG, JPG or WEBP · max 10 MB'
       : 'PNG, JPG or WEBP · images are analysed with the Happy · Neutral · Sad · Angry prototype');
  text('hwInputTitle', mode === 'collect' ? 'Write your sample below'
                                          : 'Write on the canvas — or upload a handwriting image');
  show('hwAboutEmothaw', mode === 'emothaw');
  show('hwAbout4', mode === '4class');
  show('hwAboutCollect', mode === 'collect');
  text('hwAboutTitle', mode === 'collect' ? 'About sample collection' : 'About this model');
  try { history.replaceState(null, '', '#' + mode); } catch (e) { /* not essential */ }
  show('btnHwAnalyze', mode !== 'collect');
  show('btnHwSave', mode === 'collect');
  show('hwSaved', false);
  show('hwError', false);
  show('hwDefaulted', false);
  resetTargets();
  resetFourClass();
  text('hwSummary', '-');
  if (mode === 'collect') {
    show('hwModelBanner', false);
    const pill = document.getElementById('hwModelPill');
    pill.textContent = '● Collecting samples';
    pill.className = 'pill pill-idle';
    text('hwModelInfo', '');
    refreshCollectStatus();
  } else {
    refreshModelStatus();
  }
}

// ---- model status ---------------------------------------------------------------
const HW_STATE = {
  ready:         { label: 'Model ready',   cls: 'pill-ok' },
  loading:       { label: 'Loading model', cls: 'pill-warn' },
  not_loaded:    { label: 'Loading model', cls: 'pill-warn' },
  not_installed: { label: 'No model',      cls: 'pill-bad' },
  error:         { label: 'Model error',   cls: 'pill-bad' },
};

function renderModelStatus(s) {
  const st = HW_STATE[s.state] || { label: s.state, cls: '' };
  const pill = document.getElementById('hwModelPill');
  pill.textContent = '● ' + st.label;
  pill.className = 'pill ' + st.cls;
  const problem = s.state === 'not_installed' || s.state === 'error';
  renderModelBanner(s, problem);
  const m = s.model;
  if (m && hwMode === '4class') {
    const counts = m.split_counts || {};
    const groups = ['train', 'val', 'test'].map(k => counts[k] ? `${k} ${counts[k].writers}` : null)
      .filter(Boolean).join(' / ');
    const pct = v => (v === null || v === undefined) ? '-' : Math.round(v * 100) + '%';
    const h = m.heads[0];
    const scoreLabel = m.prototype
      ? 'Seed-data pipeline check on held-out seed samples (NOT emotion-recognition accuracy)'
      : 'Held-out test';
    text('hwModelInfo', `Loaded: ${h.classes.join(' / ')}${m.prototype ? ' - PROTOTYPE / SEED MODEL' : ''}, ` +
         `trained on ${m.dataset || '?'}, split groups ${groups}, features "${m.feature_set}", ` +
         `${m.trained_at || ''}. ${scoreLabel}: balanced accuracy ` +
         `${pct((m.test_balanced_accuracy || {})[h.name])}, macro F1 ${pct((m.test_macro_f1 || {})[h.name])}.`);
  } else if (m) {
    const counts = m.split_counts || {};
    const writers = ['train', 'val', 'test'].map(k => counts[k] ? `${k} ${counts[k].writers}` : null)
      .filter(Boolean).join(' / ');
    const pct = v => (v === null || v === undefined) ? '-' : Math.round(v * 100) + '%';
    const scores = m.heads.map(h =>
      `${h.display} balanced accuracy ${pct((m.test_balanced_accuracy || {})[h.name])}, ` +
      `F1 ${pct((m.test_f1 || {})[h.name])}`).join('; ');
    const targets = m.heads.map(h => h.display).join(' / ');
    text('hwModelInfo', `Loaded: ${targets} (normal / elevated), trained on ${m.dataset || '?'}` +
         `${m.tasks ? ' (tasks ' + m.tasks.join(', ') + ')' : ''}, writers ${writers}` +
         `${m.writer_independent === false ? ' (NOT fully writer-independent)' : ''}, ` +
         `features "${m.feature_set}", ${m.trained_at || ''}. Held-out test writers: ${scores}.`);
  } else {
    text('hwModelInfo', !problem ? 'Loading...'
         : hwMode === 'emothaw' ? 'Not installed on this computer (needs the EMOTHAW dataset).'
         : 'No trained model is loaded, so no prediction can be made.');
  }
  return s.state;
}

// A missing model is explained in plain words with a way forward, not shown as an error.
function renderModelBanner(s, problem) {
  const banner = document.getElementById('hwModelBanner');
  banner.textContent = '';
  banner.className = 'alert ' + (s.state === 'error' ? 'alert-warn' : 'alert-info');
  if (problem) {
    const msg = document.createElement('span');
    if (s.state === 'not_installed' && hwMode === 'emothaw') {
      msg.textContent = 'The Anxiety · Depression · Stress model is not installed on this computer: it has to be ' +
        'trained on the EMOTHAW research dataset, which is not included in the project. You can still write ' +
        'or upload handwriting with the Happy · Neutral · Sad · Angry prototype. ';
      banner.appendChild(msg);
      const go = document.createElement('button');
      go.className = 'small-btn';
      go.textContent = 'Use Happy · Neutral · Sad · Angry';
      go.onclick = () => setHwMode('4class');
      banner.appendChild(go);
    } else if (s.state === 'not_installed') {
      msg.textContent = 'The Happy · Neutral · Sad · Angry prototype is not trained yet. Train it with ' +
        '"python -m handwriting.generate_seed_data" and "python -m handwriting.train --dataset seed_4class".';
      banner.appendChild(msg);
    } else {
      msg.textContent = s.message || 'The model could not be loaded.';
      banner.appendChild(msg);
    }
  }
  show('hwModelBanner', problem);
}

async function refreshModelStatus() {
  const mode = hwMode;
  if (!HW_ENDPOINTS[mode]) return;              // collect mode has no model
  const url = HW_ENDPOINTS[mode].status;
  try {
    const s = await getJSON(url);
    noteNetworkOk();
    if (mode !== hwMode) return;                 // the user switched model meanwhile
    if (renderModelStatus(s) === 'loading' || s.state === 'not_loaded') setTimeout(refreshModelStatus, 1000);
  } catch (e) { noteNetworkError(url, e); }
}

// ---- results ------------------------------------------------------------------
// One fixed row per EMOTHAW target (templates/handwriting.html): the prediction
// (Normal / Elevated) and the probability of "elevated".
function resetTargets() {
  document.querySelectorAll('#hwHeads .hw-head').forEach(row => {
    row.querySelector('.hw-head-label').textContent = '-';
    row.querySelector('.voice-bar-fill').style.width = '0%';
    row.querySelector('.voice-bar-pct').textContent = '-';
  });
  text('hwMeta', '');
  text('hwEmoji', '✍️');
}

function renderResult(r) {
  document.querySelectorAll('#hwHeads .hw-head').forEach(row => {
    const target = row.dataset.target;
    const label = r.predictions[target];
    const p = r.probabilities[target];
    if (label === undefined || typeof p !== 'number') return;
    const elevated = label === 'Elevated';
    const shown = elevated ? p : 1 - p;          // confidence of the predicted class
    row.querySelector('.hw-head-label').textContent = `${label} · ${(shown * 100).toFixed(1)}%`;
    const fill = row.querySelector('.voice-bar-fill');
    fill.style.width = Math.round(p * 100) + '%';
    fill.style.background = elevated ? '#d97706' : '#2563eb';
    row.querySelector('.voice-bar-pct').textContent = Math.round(p * 100) + '%';
  });
  text('hwSummary', r.elevated && r.elevated.length ? r.elevated.join(', ') : 'None predicted');
  text('hwEmoji', r.elevated && r.elevated.length ? '⚠️' : '✅');
  const i = r.input;
  text('hwMeta', `${i.points_received} points · ${i.strokes} strokes · ${i.duration_s}s · ` +
                 `${i.windows} window(s) · ${r.inference_ms} ms`);
  const d = i.defaulted_features || [];
  text('hwDefaulted', d.length ? `Your device did not report ${d.join(' / ')}; the training ` +
       `average was used for ${d.length > 1 ? 'these' : 'this'}. Predictions from a real stylus are more reliable.` : '');
  show('hwDefaulted', d.length > 0);
}

// 4-class prototype: one bar per class, the predicted class highlighted.
const HW4_EMOJI = { Happy: '😊', Neutral: '😐', Sad: '😢', Angry: '😠' };

function resetFourClass() {
  document.querySelectorAll('#hw4Bars .voice-bar-row').forEach(row => {
    row.querySelector('.voice-bar-fill').style.width = '0%';
    row.querySelector('.voice-bar-pct').textContent = '-';
  });
  text('hw4Summary', '-');
  const conf = document.getElementById('hw4Confidence');
  if (conf) conf.innerHTML = '<b>Confidence:</b> -';
  text('hw4Meta', '');
  text('hw4Emoji', '✍️');
}

// ---- image upload (4-class prototype) ----------------------------------------------
const HW_UPLOAD_MAX = 10 * 1024 * 1024;
const HW_UPLOAD_EXT = /\.(png|jpe?g|webp)$/i;
let hwImageFile = null;
let hwImageUrl = null;

function showImageMode(on) {
  show('hwImageWrap', on);
  show('hwCanvasWrap', !on);
  show('btnHwClear', !on);
  show('hwCaptured', !on);
}

function onImageChosen(input) {
  const f = input.files && input.files[0];
  input.value = '';                                  // choosing the same file again still fires
  if (!f) return;
  if (hwMode !== '4class') setHwMode('4class');       // only the 4-class prototype analyses images
  show('hwError', false);
  let problem = null;
  if (!HW_UPLOAD_EXT.test(f.name) && !['image/png', 'image/jpeg', 'image/webp'].includes(f.type))
    problem = 'Unsupported file type - please choose a PNG, JPG or WEBP image.';
  else if (f.size === 0) problem = 'The chosen file is empty.';
  else if (f.size > HW_UPLOAD_MAX) problem = 'The image is too large (max 10 MB).';
  if (problem) { text('hwError', problem); show('hwError', true); return; }
  if (hwImageUrl) URL.revokeObjectURL(hwImageUrl);
  hwImageFile = f;
  hwImageUrl = URL.createObjectURL(f);
  document.getElementById('hwImagePreview').src = hwImageUrl;
  text('hwImageName', f.name);
  showImageMode(true);
  resetFourClass();
}

function removeImage() {
  hwImageFile = null;
  if (hwImageUrl) URL.revokeObjectURL(hwImageUrl);
  hwImageUrl = null;
  document.getElementById('hwImagePreview').removeAttribute('src');
  showImageMode(false);
  resetFourClass();
  show('hwError', false);
}

async function analyzeImage() {
  const btn = document.getElementById('btnHwAnalyze');
  btn.disabled = true;
  text('hw4Summary', 'Analyzing...');
  const form = new FormData();
  form.append('image', hwImageFile, hwImageFile.name);
  try {
    const r = await getJSON('/handwriting/4class/analyze-image', { method: 'POST', body: form });
    noteNetworkOk();
    if (hwMode !== '4class') return;
    if (!r.ok) {
      resetFourClass();
      text('hwError', r.error || 'Analysis failed.');
      show('hwError', true);
      if (r.status) renderModelStatus(r.status);
      return;
    }
    renderFourClass(r);
  } catch (e) {
    noteNetworkError('/handwriting/4class/analyze-image', e);
    text('hwError', 'Could not reach the server: ' + e.message);
    show('hwError', true);
    text('hw4Summary', '-');
  } finally {
    btn.disabled = false;
  }
}

function renderFourClass(r) {
  document.querySelectorAll('#hw4Bars .voice-bar-row').forEach(row => {
    const c = row.dataset.class;
    const p = r.probabilities[c];
    if (typeof p !== 'number') return;
    const fill = row.querySelector('.voice-bar-fill');
    fill.style.width = Math.round(p * 100) + '%';
    fill.style.background = c === r.emotion ? '#2563eb' : '#98a2b3';
    row.querySelector('.voice-bar-pct').textContent = Math.round(p * 100) + '%';
  });
  text('hw4Summary', r.emotion);
  document.getElementById('hw4Confidence').innerHTML = '<b>Confidence:</b> ' + (r.confidence * 100).toFixed(1) + '%';
  text('hw4Emoji', HW4_EMOJI[r.emotion] || '✍️');
  if (r.notice) text('hw4Notice', r.notice);
  const i = r.input;
  text('hw4Meta', i.source_type === 'uploaded_image'
    ? `${r.prototype ? 'Prototype · ' : ''}uploaded image ${i.image_size[0]}×${i.image_size[1]} · ` +
      `${i.strokes} strokes reconstructed (order estimated; no timing or pressure) · ${r.inference_ms} ms`
    : `${r.prototype ? 'Prototype · ' : ''}canvas · ${i.points_received} points · ` +
      `${i.strokes} strokes · ${i.windows} window(s) · ${r.inference_ms} ms`);
  show('hwDefaulted', false);                    // the 4-class profile may use no pen channels at all
}

async function analyzeHandwriting() {
  show('hwError', false);
  if (hwMode === '4class' && hwImageFile) return analyzeImage();
  const pts = Handwriting.points();
  if (!pts.some(p => p.pen_down)) {
    text('hwError', 'Please write something before analyzing.');
    show('hwError', true);
    return;
  }
  const btn = document.getElementById('btnHwAnalyze');
  btn.disabled = true;
  const mode = hwMode;
  const url = HW_ENDPOINTS[mode].analyze;
  text(mode === '4class' ? 'hw4Summary' : 'hwSummary', 'Analyzing...');
  try {
    const r = await postJSON(url, { points: pts, pointer_types: Handwriting.pointerTypes() });
    noteNetworkOk();
    if (mode !== hwMode) return;                 // switched model while waiting
    if (!r.ok) {
      resetTargets();
      resetFourClass();
      text('hwSummary', '-');
      let msg = r.error || 'Analysis failed.';
      if (r.kind === 'model_unavailable' && r.input) {
        msg += ` (Your handwriting was received and preprocessed: ${r.input.points_received} points, ` +
               `${r.input.strokes} strokes.)`;
      }
      text('hwError', msg);
      show('hwError', true);
      if (r.status) renderModelStatus(r.status);
      return;
    }
    if (mode === '4class') renderFourClass(r); else renderResult(r);
  } catch (e) {
    noteNetworkError(url, e);
    text('hwError', 'Could not reach the server: ' + e.message);
    show('hwError', true);
    text('hwSummary', '-');
    text('hw4Summary', '-');
  } finally {
    btn.disabled = false;
  }
}

// ---- collect mode ---------------------------------------------------------------
const WRITER_RE = /^writer_\d{3,6}$/;
let hwLabel = null;

function writerInput() { return document.getElementById('hwWriterId'); }
function currentWriter() { return writerInput().value.trim().toLowerCase(); }

function selectLabel(label) {
  hwLabel = label;
  document.querySelectorAll('#hwLabels .hw-label').forEach(b =>
    b.classList.toggle('active', b.dataset.label === label));
  show('hwSaved', false);
}

function renderCounts(s) {
  Object.entries(s.totals || {}).forEach(([c, n]) => text('hwCountAll' + c, n));
  Object.keys(s.totals || {}).forEach(c =>
    text('hwCountMine' + c, s.writer ? s.writer[c] : '-'));
  text('hwCollectMeta', `${s.writers} writer(s), ${s.samples} sample(s) collected so far.`);
}

async function refreshCollectStatus() {
  const w = currentWriter();
  const url = '/handwriting/4class/collect/status' + (WRITER_RE.test(w) ? '?writer_id=' + encodeURIComponent(w) : '');
  try {
    const s = await getJSON(url);
    noteNetworkOk();
    if (s.ok) renderCounts(s);
    return s;
  } catch (e) { noteNetworkError(url, e); }
}

function onWriterIdChange() {
  try { localStorage.setItem('hwWriterId', currentWriter()); } catch (e) { /* storage unavailable */ }
  show('hwSaved', false);
  refreshCollectStatus();
}

async function useNextWriterId() {
  const s = await refreshCollectStatus();
  if (s && s.next_writer_id) {
    writerInput().value = s.next_writer_id;
    onWriterIdChange();
  }
}

async function saveSample() {
  show('hwError', false);
  show('hwSaved', false);
  const w = currentWriter();
  let problem = null;
  if (!WRITER_RE.test(w)) problem = 'Enter an anonymous writer ID like writer_001 (or press New ID).';
  else if (!hwLabel) problem = 'Select a label first.';
  else if (!Handwriting.points().some(p => p.pen_down)) problem = 'Please write something before saving.';
  if (problem) { text('hwError', problem); show('hwError', true); return; }
  const btn = document.getElementById('btnHwSave');
  btn.disabled = true;
  try {
    const r = await postJSON('/handwriting/4class/collect',
      { writer_id: w, label: hwLabel, points: Handwriting.points(), pointer_types: Handwriting.pointerTypes() });
    noteNetworkOk();
    if (!r.ok) { text('hwError', r.error || 'Could not save the sample.'); show('hwError', true); return; }
    Handwriting.clear();
    selectLabel(null);
    text('hwSaved', `Sample saved: ${r.label} #${r.number}`);
    show('hwSaved', true);
    renderCounts(r.counts);
  } catch (e) {
    noteNetworkError('/handwriting/4class/collect', e);
    text('hwError', 'Could not reach the server: ' + e.message);
    show('hwError', true);
  } finally {
    btn.disabled = false;
  }
}

try { writerInput().value = localStorage.getItem('hwWriterId') || ''; } catch (e) { /* storage unavailable */ }
// Open on the tab named in the URL (#4class, #collect, #emothaw), else the one the server chose.
setHwMode(HW_MODES.includes(location.hash.slice(1)) ? location.hash.slice(1) : (window.HW_INITIAL_MODE || 'emothaw'));
