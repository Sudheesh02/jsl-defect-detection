/**
 * Jindal Stainless AI Surface Defect Inspection Platform - Frontend Controller
 */

let currentInspectionData = null;
let currentRawImageUrl = '/raw_images/scratches_1.jpg';
let activeSelectedDefectId = null;

document.addEventListener('DOMContentLoaded', () => {
  initHealthTelemetry();
  initGradeSelector();
  initPresetButtons();
  initDropZone();
  initSliders();
  initActionButtons();
  initViewToggles();
  initBenchmarkModal();

  // Run initial inspection on default scratches sample
  loadAndInspectSample('scratches_1.jpg');
});

/**
 * Fetch system health and telemetry
 */
async function initHealthTelemetry() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.device === 'cuda') {
      document.getElementById('telemetryDevice').textContent = `${data.cuda_device_name} (CUDA)`;
    } else {
      document.getElementById('telemetryDevice').textContent = 'CPU Mode';
    }
  } catch (err) {
    console.error('Failed to fetch health telemetry:', err);
  }
}

/**
 * Stainless Steel Grade Selector
 */
function initGradeSelector() {
  const gradeSelect = document.getElementById('gradeSelect');
  const gradeHint = document.getElementById('gradeHint');

  const gradeHints = {
    'SS_304': 'Aesthetic sensitivity is high. Scratches and roll marks are penalized for mirror/2B finish.',
    'SS_316L': 'Extreme pitting sensitivity (PREN >= 24). Zero tolerance for surface pits and smelting inclusions.',
    'SS_430': 'Ferritic matrix. Strict control on crazing microcracks to prevent forming roping/tearing.',
    'SS_201': 'Commercial economy austenitic. Tolerates minor superficial blemishes for cost-optimized stamping.',
    'DUPLEX_2205': 'High-strength structural duplex. Crazing and surface notches strictly rejected due to SCC risk.'
  };

  gradeSelect.addEventListener('change', (e) => {
    gradeHint.textContent = gradeHints[e.target.value] || '';
    // Re-inspect with updated grade tolerances if image loaded
    if (currentInspectionData) {
      runInspection();
    }
  });
}

/**
 * Preset Benchmark Buttons
 */
function initPresetButtons() {
  const presetBtns = document.querySelectorAll('.preset-btn');
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filename = btn.getAttribute('data-file');
      loadAndInspectSample(filename);
    });
  });
}

/**
 * Drop Zone and File Upload
 */
function initDropZone() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleCustomFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleCustomFileUpload(e.target.files[0]);
    }
  });
}

/**
 * Sliders and Inputs
 */
function initSliders() {
  const confSlider = document.getElementById('confSlider');
  const confVal = document.getElementById('confVal');
  confSlider.addEventListener('input', (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  const speedSlider = document.getElementById('speedSlider');
  const speedVal = document.getElementById('speedVal');
  const speedSub = document.getElementById('speedSub');
  speedSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    speedVal.textContent = val;
    const mps = (val / 60).toFixed(1);
    speedSub.textContent = `${mps} m/s (${mps > 15 ? 'Hot Rolling Mill' : 'Cold Rolling / Finishing'})`;
  });
}

/**
 * Inspection Actions
 */
function initActionButtons() {
  document.getElementById('inspectBtn').addEventListener('click', () => {
    runInspection();
  });
}

/**
 * Toggle Annotated vs Raw View
 */
function initViewToggles() {
  const viewAnnotatedBtn = document.getElementById('viewAnnotatedBtn');
  const viewOriginalBtn = document.getElementById('viewOriginalBtn');
  const imgEl = document.getElementById('inspectionImage');

  viewAnnotatedBtn.addEventListener('click', () => {
    viewAnnotatedBtn.classList.add('active');
    viewOriginalBtn.classList.remove('active');
    if (currentInspectionData && currentInspectionData.annotated_image_b64) {
      imgEl.src = `data:image/jpeg;base64,${currentInspectionData.annotated_image_b64}`;
    }
  });

  viewOriginalBtn.addEventListener('click', () => {
    viewOriginalBtn.classList.add('active');
    viewAnnotatedBtn.classList.remove('active');
    imgEl.src = currentRawImageUrl;
  });

  document.getElementById('exportReportBtn').addEventListener('click', exportInspectionReport);
}

/**
 * Inspect a sample from server
 */
async function loadAndInspectSample(filename) {
  showLoading(true);
  currentRawImageUrl = `/raw_images/${filename}`;

  const gradeCode = document.getElementById('gradeSelect').value;
  const conf = document.getElementById('confSlider').value;
  const speed = document.getElementById('speedSlider').value;
  const verify = document.getElementById('verifyToggle').checked;

  try {
    const url = `/api/sample-detect/${filename}?grade_code=${gradeCode}&conf_threshold=${conf}&line_speed_m_per_min=${speed}&verify_false_alarms=${verify}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderInspectionResults(data);
  } catch (err) {
    console.error('Error inspecting sample:', err);
    alert('Failed to inspect sample: ' + err.message);
  } finally {
    showLoading(false);
  }
}

let lastUploadedFile = null;

/**
 * Inspect custom uploaded image
 */
async function handleCustomFileUpload(file) {
  showLoading(true);
  lastUploadedFile = file;

  // Deactivate preset buttons since user uploaded custom image
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));

  // Ensure reader completes so raw view is immediately available
  await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      currentRawImageUrl = e.target.result;
      resolve();
    };
    reader.onerror = () => resolve();
    reader.readAsDataURL(file);
  });

  const gradeCode = document.getElementById('gradeSelect').value;
  const conf = document.getElementById('confSlider').value;
  const speed = document.getElementById('speedSlider').value;
  const verify = document.getElementById('verifyToggle').checked;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const url = `/api/detect?grade_code=${gradeCode}&conf_threshold=${conf}&line_speed_m_per_min=${speed}&verify_false_alarms=${verify}`;
    const res = await fetch(url, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderInspectionResults(data);
  } catch (err) {
    console.error('Error uploading and inspecting:', err);
    alert('Error inspecting uploaded file: ' + err.message);
  } finally {
    showLoading(false);
  }
}

/**
 * Re-run current inspection with updated sliders/grade
 */
function runInspection() {
  const activeBtn = document.querySelector('.preset-btn.active');
  if (activeBtn) {
    loadAndInspectSample(activeBtn.getAttribute('data-file'));
  } else if (lastUploadedFile) {
    handleCustomFileUpload(lastUploadedFile);
  } else {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
      handleCustomFileUpload(fileInput.files[0]);
    }
  }
}

/**
 * Render Complete Inspection Results to UI
 */
function renderInspectionResults(data) {
  currentInspectionData = data;

  // 1. Update Viewport Image
  const imgEl = document.getElementById('inspectionImage');
  if (data.annotated_image_b64) {
    imgEl.src = `data:image/jpeg;base64,${data.annotated_image_b64}`;
    document.getElementById('viewAnnotatedBtn').classList.add('active');
    document.getElementById('viewOriginalBtn').classList.remove('active');
  } else {
    imgEl.src = currentRawImageUrl;
  }

  document.getElementById('viewportResolution').textContent = 
    `${data.image_dimensions.width} x ${data.image_dimensions.height} px`;

  // 2. Top Ribbon Metrics
  const disp = data.disposition;
  const dispValEl = document.getElementById('dispositionValue');
  dispValEl.textContent = disp.disposition;
  dispValEl.style.color = disp.status_color;
  document.getElementById('dispositionAction').textContent = disp.action_summary;

  const gradeBadge = document.getElementById('gradeBadge');
  gradeBadge.textContent = disp.quality_grade;
  gradeBadge.style.color = disp.status_color;
  gradeBadge.style.borderColor = disp.status_color;
  document.getElementById('gradeScore').textContent = `${disp.quality_score_pct}% Score`;
  document.getElementById('compositeSev').textContent = `Severity: ${disp.overall_severity} / 10`;

  document.getElementById('defectCountVal').textContent = data.defect_count;
  
  // Defect pills
  const pillsEl = document.getElementById('defectPillsSummary');
  pillsEl.innerHTML = '';
  if (data.defect_count === 0) {
    pillsEl.innerHTML = '<span class="pill-badge text-green">Zero Defects (Prime Strip)</span>';
  } else {
    const counts = {};
    data.defects.forEach(d => counts[d.defect_class] = (counts[d.defect_class] || 0) + 1);
    for (const [cls, cnt] of Object.entries(counts)) {
      const span = document.createElement('span');
      span.className = `pill-badge tag-${cls}`;
      span.textContent = `${cls}: ${cnt}`;
      pillsEl.appendChild(span);
    }
  }

  // Throughput & Speed
  const lineM = data.production_line_metrics;
  document.getElementById('throughputFps').textContent = `${data.latency.fps} FPS`;
  document.getElementById('throughputHeadroom').textContent = 
    `Headroom: ${lineM.throughput_headroom_ratio}x (${lineM.frame_drop_risk.split(' ')[0]})`;

  // Latency footer
  document.getElementById('latencyPre').textContent = `${data.latency.preprocess_ms} ms`;
  document.getElementById('latencyInfer').textContent = `${data.latency.inference_ms} ms`;
  document.getElementById('latencyPost').textContent = `${data.latency.postprocess_ms} ms`;
  document.getElementById('latencyTotal').textContent = `${data.latency.total_pipeline_ms} ms`;
  document.getElementById('maxLineSpeed').textContent = `${lineM.max_sustainable_speed_m_per_min} m/min`;
  document.getElementById('telemetrySpeed').textContent = `${data.latency.total_pipeline_ms} ms`;

  // 3. Defect Table
  const tableBody = document.getElementById('defectTableBody');
  tableBody.innerHTML = '';
  document.getElementById('tableDefectCount').textContent = `${data.defect_count} detected`;

  if (data.defect_count === 0) {
    tableBody.innerHTML = '<tr><td colspan="7" class="empty-state">Zero defects found. Surface meets prime stainless steel standard.</td></tr>';
    renderEmptyRootCause();
    return;
  }

  data.defects.forEach((d, idx) => {
    const row = document.createElement('tr');
    row.setAttribute('data-id', d.defect_id);
    if (idx === 0) row.classList.add('selected');

    const thumbHtml = d.thumbnail_b64 
      ? `<img src="data:image/jpeg;base64,${d.thumbnail_b64}" class="defect-thumb" alt="${d.defect_class}">`
      : `<div class="defect-thumb" style="background:#1e2838"></div>`;

    const isFalseAlarm = d.is_potential_false_alarm || d.severity_tier === 'Suppressed / False Alarm';
    const sevColor = isFalseAlarm ? '#78909c' : (d.severity_score >= 7.5 ? 'var(--color-red)' : d.severity_score >= 4.5 ? 'var(--color-orange)' : 'var(--color-green)');
    const zoneBadge = isFalseAlarm ? '<span style="color:#90a4ae">Non-Hazard</span>' : (d.is_edge_defect ? '<span class="text-amber">Edge Risk (1.35x)</span>' : 'Center Body');
    const reworkBadge = isFalseAlarm ? '<span class="text-green">Benign (Cleared)</span>' : (d.reworkable ? '<span class="text-green">Yes (Pickle/Grind)</span>' : '<span class="text-red">Non-reworkable</span>');
    const confDisplay = d.consensus_confidence 
      ? `${(d.consensus_confidence * 100).toFixed(1)}% <small style="color:#00e5ff">(Consensus)</small>` 
      : `${(d.confidence * 100).toFixed(1)}%`;
    const tagClass = isFalseAlarm ? 'tag-clean' : `tag-${d.defect_class}`;
    const nameDisplay = isFalseAlarm ? `[Suppressed] ${d.full_name}` : d.full_name;

    row.innerHTML = `
      <td><strong>${d.defect_id}</strong></td>
      <td>${thumbHtml}</td>
      <td><span class="preset-tag ${tagClass}">${nameDisplay}</span></td>
      <td>${confDisplay}</td>
      <td><span style="color:${sevColor};font-weight:700">${d.severity_score}</span> (${d.severity_tier})</td>
      <td>${zoneBadge}</td>
      <td>${reworkBadge}</td>
    `;

    row.addEventListener('click', () => {
      document.querySelectorAll('.defect-table tr').forEach(r => r.classList.remove('selected'));
      row.classList.add('selected');
      renderRootCause(d);
    });

    tableBody.appendChild(row);
  });

  // Render first defect in root cause panel by default
  renderRootCause(data.defects[0]);
}

/**
 * Render Metallurgical Root Cause
 */
function renderRootCause(defect) {
  const container = document.getElementById('rootCauseContent');
  if (!defect) {
    renderEmptyRootCause();
    return;
  }

  const actionsHtml = defect.corrective_actions.map(act => `<li>${act}</li>`).join('');

  container.innerHTML = `
    <div class="root-cause-detail">
      <div class="rc-header">
        <span class="rc-title">${defect.full_name} (${defect.defect_id})</span>
        <span class="pill-badge tag-${defect.defect_class}">Category: ${defect.category}</span>
      </div>

      <div class="rc-block">
        <h4>Metallurgical Root Cause</h4>
        <p>${defect.root_cause}</p>
      </div>

      <div class="rc-block">
        <h4>Impact &amp; Downstream Hazard</h4>
        <p>${defect.metallurgical_hazard}</p>
      </div>

      <div class="rc-block">
        <h4>Shop-Floor Corrective Directives</h4>
        <ul class="rc-actions-list">
          ${actionsHtml}
        </ul>
      </div>
    </div>
  `;
}

function renderEmptyRootCause() {
  document.getElementById('rootCauseContent').innerHTML = `
    <div class="empty-root-cause">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="info-icon">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </svg>
      <p>Zero defects detected on this strip frame. Clean mirror steel surface.</p>
    </div>
  `;
}

function showLoading(show) {
  const overlay = document.getElementById('loadingOverlay');
  overlay.style.display = show ? 'flex' : 'none';
}

/**
 * Production Benchmark Modal
 */
function initBenchmarkModal() {
  const modal = document.getElementById('benchmarkModal');
  const openBtn = document.getElementById('benchmarkBtn');
  const closeBtn = document.getElementById('closeModalBtn');
  const runBtn = document.getElementById('runBenchmarkNowBtn');

  openBtn.addEventListener('click', () => {
    modal.style.display = 'flex';
  });

  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });

  runBtn.addEventListener('click', async () => {
    runBtn.disabled = true;
    runBtn.textContent = 'Running 30 iterations...';
    try {
      const speed = document.getElementById('speedSlider').value;
      const res = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          iterations: 30,
          line_speed_m_per_min: parseFloat(speed),
          conf_threshold: 0.25
        })
      });
      const bm = await res.json();
      
      document.getElementById('bmP50').textContent = `${bm.latency_percentiles_ms.median_ms} ms`;
      document.getElementById('bmP95').textContent = `${bm.latency_percentiles_ms.p95_ms} ms`;
      document.getElementById('bmP99').textContent = `${bm.latency_percentiles_ms.p99_ms} ms`;
      document.getElementById('bmFps').textContent = `${bm.fps_percentiles.p95_fps} FPS`;

      const lm = bm.line_speed_metrics;
      document.getElementById('bmLineVerdict').innerHTML = `
        Target Speed: <strong>${lm.target_line_speed_m_per_min} m/min</strong> (${lm.target_line_speed_m_per_s} m/s)<br>
        Maximum Real-Time Line Capacity: <strong>${lm.max_sustainable_speed_m_per_min} m/min</strong><br>
        Throughput Headroom Buffer: <span class="text-green"><strong>${lm.throughput_headroom_ratio}x</strong></span> (${lm.frame_drop_risk})
      `;
    } catch (err) {
      alert('Benchmark failed: ' + err.message);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = 'Execute 30-Frame Stress Test';
    }
  });
}

/**
 * Export JSON Inspection Log
 */
function exportInspectionReport() {
  if (!currentInspectionData) {
    alert('No inspection data to export.');
    return;
  }
  const exportPayload = {
    timestamp: new Date().toISOString(),
    plant: 'Jindal Stainless Manufacturing Line #1',
    system: 'AI-Powered Strip Surface Inspection QMS',
    grade: currentInspectionData.grade_applied,
    disposition: currentInspectionData.disposition,
    defect_count: currentInspectionData.defect_count,
    defects: currentInspectionData.defects.map(d => ({
      defect_id: d.defect_id,
      defect_class: d.defect_class,
      confidence: d.confidence,
      severity_score: d.severity_score,
      severity_tier: d.severity_tier,
      bounding_box: d.bounding_box,
      strip_zone: d.strip_zone,
      root_cause: d.root_cause,
      corrective_actions: d.corrective_actions
    })),
    production_line_metrics: currentInspectionData.production_line_metrics,
    latency: currentInspectionData.latency
  };

  const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `JSL_Inspection_Report_${new Date().getTime()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
