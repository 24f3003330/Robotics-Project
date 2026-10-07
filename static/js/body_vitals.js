// ============================================================================
// Body Vitals page (/body-vitals). Independent of camera / voice / handwriting.
// Demo buttons -> POST /body-vitals/demo -> render the SIMULATED inputs and the
// prediction of the WESAD-trained Random Forest (the only prediction mechanism).
// ============================================================================

const BV_CLASSES = ['baseline', 'stress', 'amusement'];
const BV_DISPLAY = { baseline: 'Baseline', stress: 'Stress', amusement: 'Amusement' };
const BV_EMOJI = { Baseline: '🟦', Stress: '🟥', Amusement: '🟨' };
const BV_MODEL_NAMES = { RandomForest: 'Random Forest', XGBoost: 'XGBoost', SVM: 'SVM (RBF kernel)', HistGradientBoosting: 'Gradient boosting', LogisticRegression: 'Logistic regression' };

function bvText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function bvShow(id, visible) {
  const el = document.getElementById(id);
  if (el) el.hidden = !visible;
}

function bvError(message) {
  bvText('bvError', message || '');
  bvShow('bvError', !!message);
}

function pct(x) {
  return Math.round(x * 100) + '%';
}

async function loadStatus() {
  try {
    const res = await fetch('/body-vitals/status');
    const st = await res.json();
    const pill = document.getElementById('bvModelPill');
    if (st.ready) {
      const m = st.model;
      pill.className = 'pill pill-ok';
      pill.textContent = '● model ready';
      bvText('bvModelName', (BV_MODEL_NAMES[m.model_type] || m.model_type) + ' · trained on ' + m.dataset);
      const rec = m.test_recall || {};
      bvText('bvModelInfo',
        (BV_MODEL_NAMES[m.model_type] || m.model_type) + ' trained on WESAD chest ECG / EDA / respiration (' +
        m.n_features + ' features, ' + m.window_size_sec + ' s windows' +
        (m.normalization === 'rest_calibration' ? ', compared with the person\'s own resting reference' : '') +
        '). On held-out test subjects ' + (m.test_subjects || []).join(', ') +
        ': accuracy ' + pct(m.test_accuracy) + ', balanced accuracy ' + pct(m.test_balanced_accuracy) +
        ', macro F1 ' + m.test_macro_f1.toFixed(2) + '; recall Baseline ' + pct(rec.baseline || 0) +
        ', Stress ' + pct(rec.stress || 0) + ', Amusement ' + pct(rec.amusement || 0) + '. ' +
        'Classes are WESAD study conditions, not self-reported emotions and not a diagnosis.');
      bvShow('bvModelBanner', false);
    } else {
      pill.className = 'pill pill-bad';
      pill.textContent = '● model not installed';
      bvText('bvModelBanner', st.error || 'The Body Vitals model is not available.');
      bvShow('bvModelBanner', true);
    }
  } catch (e) {
    bvError('Cannot reach the server: ' + e);
  }
}

async function runDemo(scenario) {
  bvError('');
  document.querySelectorAll('.bv-demo-buttons button').forEach(b => { b.disabled = true; });
  try {
    const res = await fetch('/body-vitals/demo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: scenario }),
    });
    const data = await res.json();
    if (!data.success) {
      bvError(data.error || 'Prediction failed.');
      if (data.kind === 'model_unavailable') loadStatus();
      return;
    }
    render(data);
  } catch (e) {
    bvError('Request failed: ' + e);
  } finally {
    document.querySelectorAll('.bv-demo-buttons button').forEach(b => { b.disabled = false; });
  }
}

function render(data) {
  const input = data.input;
  const simulated = input.is_simulated === true;

  // where the numbers came from - never shown as sensor data unless they are
  const pill = document.getElementById('bvSourcePill');
  pill.className = 'pill ' + (simulated ? 'pill-warn' : 'pill-idle');
  pill.textContent = '● ' + input.label;
  bvShow('bvSimBanner', simulated);
  if (simulated) bvText('bvSimNotice', input.notice);
  const tag = document.getElementById('bvSourceTag');
  tag.hidden = false;
  tag.textContent = simulated ? 'SIMULATED' : input.label;
  document.getElementById('bvVitals').classList.toggle('bv-simulated', simulated);

  const v = input.vitals;
  bvText('bvHr', v.heart_rate_bpm.toFixed(0));
  bvText('bvSdnn', v.hrv_sdnn_ms.toFixed(0));
  bvText('bvRmssd', v.hrv_rmssd_ms.toFixed(0));
  bvText('bvGsr', v.gsr_us.toFixed(2));
  bvText('bvResp', v.respiration_rate_bpm.toFixed(0));
  const q = input.signal_quality;
  bvText('bvQuality', q ? ('Measured from a ' + (input.window_sec || 60) + ' s window: ' + q.r_peaks +
    ' R-peaks (' + q.rr_valid + ' clean RR intervals), ' + q.breaths + ' breaths, ' + q.scr_peaks +
    ' skin conductance responses.') : '');

  const cal = data.calibration || {};
  bvShow('bvCalibration', !!cal.used);
  if (cal.used) {
    bvText('bvCalibration', 'Personal resting reference: ' + cal.windows + ' ' +
      (cal.is_simulated ? 'SIMULATED resting windows of the same simulated person' : 'resting windows') +
      (cal.seconds ? ' (first ' + Math.round(cal.seconds / 60) + ' min of seated rest)' : '') +
      '. The model compares this window with that reference.');
  }

  const tbody = document.querySelector('#bvFeatures tbody');
  tbody.innerHTML = '';
  Object.entries(input.features).forEach(([name, value]) => {
    const tr = document.createElement('tr');
    const a = document.createElement('td');
    const b = document.createElement('td');
    a.textContent = name;
    b.textContent = Number(value).toPrecision(4);
    tr.append(a, b);
    tbody.appendChild(tr);
  });

  bvText('bvState', data.predicted_state.toUpperCase());
  bvText('bvEmoji', BV_EMOJI[data.predicted_state] || '💓');
  bvText('bvConfidence', pct(data.confidence));
  bvText('bvModelName', (BV_MODEL_NAMES[data.model.type] || data.model.type) + ' · trained on ' + data.model.dataset);
  bvText('bvScenario', input.scenario_requested
    ? ' · demo scenario requested: ' + BV_DISPLAY[input.scenario_requested] : '');
  BV_CLASSES.forEach(c => {
    const p = data.probabilities[BV_DISPLAY[c]];
    document.getElementById('bvBar-' + c).style.width = pct(p);
    bvText('bvPct-' + c, pct(p));
  });
  document.getElementById('bvBars').classList.remove('idle');
  const box = document.getElementById('bvStateBox');
  box.classList.remove('flash');
  void box.offsetWidth;
  box.classList.add('flash');
}

loadStatus();
