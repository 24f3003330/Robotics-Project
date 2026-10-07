// ============================================================================
// CAMERA PAGE - webcam lifecycle, frame polling and face-emotion display.
// Never calls any /voice endpoint: the microphone and the voice model are
// untouched by this page.
//
// Lifecycle: page load -> POST /camera/start -> poll /video_frame (~10 fps)
// and /status (1 s). Hidden tab / leaving the page -> /camera/stop, so the
// webcam is released; coming back re-opens it.
// ============================================================================

// ---- frame feed: polled single JPEGs (works in Safari, unlike MJPEG <img>) --
const CameraFeed = (function () {
  const img = document.getElementById('videoFeed');
  let currentUrl = null;
  let inFlight = false;
  let timer = null;
  let failures = 0;

  async function tick() {
    if (inFlight) return;          // never let two frame fetches overlap
    inFlight = true;
    try {
      const res = await fetch('/video_frame', { cache: 'no-store' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      img.src = url;
      // Revoke the PREVIOUS object URL only after the new one is assigned.
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      currentUrl = url;
      failures = 0;
    } catch (e) {
      failures++;
      if (failures === 5) console.log('[CAMERA] /video_frame failing:', e);
    } finally {
      inFlight = false;
    }
  }

  function start(intervalMs) {
    if (timer) return;
    tick();
    timer = setInterval(tick, intervalMs || 100);
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
  }

  return { start, stop, isRunning: () => timer !== null };
})();

let cameraWanted = true;   // the user's intent (Stop Camera sets it false)

async function startCamera() {
  cameraWanted = true;
  show('cameraStatusMessage', false);
  try {
    const data = await postJSON('/camera/start');
    noteNetworkOk();
    renderCameraStatus({ camera_status: data.camera_status, camera_running: true, detection_active: true });
  } catch (e) {
    noteNetworkError('/camera/start', e);
  }
  CameraFeed.start(100);
  startStatusPolling();
}

async function stopCamera() {
  cameraWanted = false;
  CameraFeed.stop();
  try { await postJSON('/camera/stop'); } catch (e) { noteNetworkError('/camera/stop', e); }
  pollStatus();
}

// ---- status ----------------------------------------------------------------
const CAMERA_STATUS_DISPLAY = {
  starting:  { label: 'STARTING',  cls: 'pill-warn' },
  connected: { label: 'CONNECTED', cls: 'pill-ok' },
  denied:    { label: 'UNAVAILABLE', cls: 'pill-bad' },
  error:     { label: 'ERROR',     cls: 'pill-bad' },
  stopped:   { label: 'STOPPED',   cls: 'pill-idle' },
};

function renderCameraStatus(data) {
  const cam = data.camera_status || {};
  let state = cam.state || 'starting';
  if (state === 'connected' && data.detection_active === false) state = 'paused';
  const display = CAMERA_STATUS_DISPLAY[state] || { label: 'PAUSED', cls: 'pill-warn' };

  const pill = document.getElementById('cameraStatusPill');
  pill.textContent = '● ' + display.label;
  pill.className = 'pill ' + display.cls;

  const failed = state === 'denied' || state === 'error';
  if (failed) {
    text('cameraStatusText', cam.detail ||
         'Camera access is unavailable. Please grant camera permission and try again.');
    // Nothing to fetch until the user retries.
    if (!data.camera_running) CameraFeed.stop();
  }
  show('cameraStatusMessage', failed);

  const running = !!data.camera_running;
  show('btnStartCamera', !running);
  show('btnStopCamera', running);
}

const FACE_BAR_COLORS = {
  happy: '#f59e0b', angry: '#dc2626', sad: '#4f46e5', neutral: '#0ea5e9',
  surprise: '#f97316', fear: '#7c3aed', disgust: '#16a34a', contempt: '#64748b',
};

// Built on first use from whatever emotions the face model reports.
function renderFaceBars(probs, tracking) {
  const box = document.getElementById('faceBars');
  const emotions = Object.keys(probs);
  if (emotions.length && box.childElementCount !== emotions.length) {
    box.textContent = '';
    emotions.forEach(e => {
      const row = document.createElement('div');
      row.className = 'voice-bar-row';
      row.innerHTML = `<span>${e.charAt(0).toUpperCase() + e.slice(1)}</span>` +
        `<div class="voice-bar-track"><div class="voice-bar-fill" id="fbar-${e}" ` +
        `style="background:${FACE_BAR_COLORS[e] || '#2563eb'}"></div></div>` +
        `<span class="voice-bar-pct" id="fpct-${e}">0%</span>`;
      box.appendChild(row);
    });
  }
  box.querySelectorAll('.voice-bar-fill').forEach(bar => {
    const e = bar.id.slice(5);
    const pct = Math.round((probs[e] || 0) * 100);
    bar.style.width = pct + '%';
    text('fpct-' + e, pct + '%');
  });
  box.classList.toggle('idle', !tracking);
}

function renderStatus(data) {
  const face = data.face || {};
  applyEmotionState({
    state_seq: data.state_seq,
    emotion: (data.final || {}).emotion,
    source: (data.final || {}).source,
    confidence: (data.final || {}).confidence,
    pose_key: data.robot_pose_key,
    background: data.background,
    song: data.song,
    last_action: data.last_action,
  });
  renderCameraStatus(data);
  renderRobotPills(data.robot_connected, data.robot_busy);

  const emotion = face.emotion ? face.emotion.toLowerCase() : null;
  text('faceEmotion', face.emotion ? face.emotion.toUpperCase()
                     : (data.camera_running ? 'Looking for a face...' : '-'));
  text('faceEmoji', EMOTION_EMOJI[emotion] || '🙂');
  text('faceConfidence', face.emotion ? Math.round((face.confidence || 0) * 100) + '%' : '-');
  text('faceTracking', face.tracking ? 'yes' : 'no');
  renderFaceBars(face.probabilities || {}, !!face.tracking);
  text('faceModelWarning', face.model_warning || '');
  show('faceModelWarning', !!face.model_warning);
  text('detectionStatus', !data.camera_running ? 'camera off'
                          : data.detection_active ? 'running' : 'paused');
}

let statusBusy = false;
let statusTimer = null;

async function pollStatus() {
  if (statusBusy) return;                // no overlapping /status requests
  statusBusy = true;
  try {
    const data = await getJSON('/status');
    noteNetworkOk();
    renderStatus(data);
  } catch (e) {
    noteNetworkError('/status', e);
  } finally {
    statusBusy = false;
  }
}
refreshPageStatus = pollStatus;

function startStatusPolling() {
  if (!statusTimer) statusTimer = setInterval(pollStatus, 500);
  pollStatus();
}

function stopStatusPolling() {
  if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
}

// ---- detection control -----------------------------------------------------
async function resetDetection() {
  try {
    showRobotOutput(await getJSON('/reset'));
    Music.stop();
    pollStatus();
  } catch (e) { noteNetworkError('/reset', e); }
}
async function stopDetection() {
  try { showRobotOutput(await getJSON('/stop_detect')); pollStatus(); }
  catch (e) { noteNetworkError('/stop_detect', e); }
}
async function resumeDetection() {
  try { showRobotOutput(await getJSON('/resume_detect')); pollStatus(); }
  catch (e) { noteNetworkError('/resume_detect', e); }
}

// ---- page lifecycle --------------------------------------------------------
function releaseCamera() {
  CameraFeed.stop();
  stopStatusPolling();
  beacon('/camera/stop');
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    releaseCamera();                      // no one is looking: free the webcam
  } else if (cameraWanted) {
    startCamera();
  } else {
    startStatusPolling();
  }
});

window.addEventListener('pagehide', releaseCamera);
window.addEventListener('pageshow', (event) => {
  // Restored from the back/forward cache: the camera was released on pagehide.
  if (event.persisted && cameraWanted) startCamera();
});

startCamera();
