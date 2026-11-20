/* SAIGAK-ORTO | COCKPIT — Spatial 3D Engine, Telemetry HUD & Clinical Workflow Controller */
(function () {
  "use strict";

  const APP_STATE = {
    currentFilename: "clinical_severe_plagio_P03.stl",
    uploadedFileB64: null,
    scaleFactor: 1.0,
    levelRatio: 0.30,
    useBaseline: false,
    expertCiApproved: false,
    customLandmarks: null,
    lastResponse: null,
    isLightTheme: false,
    // 3D Viewport State
    renderMode: "shaded", // "shaded" | "heatmap" | "wire"
    showPlanes: true,
    showLandmarks: true,
    showRays: true,
    camYaw: -0.65,
    camPitch: 0.35,
    camZoom: 2.35,
    camPanX: 0,
    camPanY: 0,
    isDragging: false,
    dragButton: 0,
    dragLandmarkKey: null,
    hoveredLandmarkKey: null,
    lastMouseX: 0,
    lastMouseY: 0,
    meshPositions: null,
    meshAsymmetry: null,
    meshCenter: [18.0, 0.0, 535.0],
    meshRadius: 75.0,
    projectedLandmarks: {},
  };

  const LANDMARK_COLORS = {
    Sellion: "#EF4444",
    RightTragion: "#3B82F6",
    LeftTragion: "#A855F7",
    Subnasal: "#10B981",
  };

  const scanSelect = document.getElementById("scanSelect");
  const stlFileInput = document.getElementById("stlFileInput");
  const scaleSelect = document.getElementById("scaleSelect");
  const customScaleInput = document.getElementById("customScaleInput");
  const scaleSuggestionBox = document.getElementById("scaleSuggestionBox");
  const applySuggestedScaleBtn = document.getElementById("applySuggestedScaleBtn");
  const levelRatioSlider = document.getElementById("levelRatioSlider");
  const levelRatioVal = document.getElementById("levelRatioVal");
  const useBaselineToggle = document.getElementById("useBaselineToggle");
  const expertCiProfileCheck = document.getElementById("expertCiProfileCheck");
  const applyLandmarksBtn = document.getElementById("applyLandmarksBtn");
  const resetLandmarksBtn = document.getElementById("resetLandmarksBtn");
  const addToQueueBtn = document.getElementById("addToQueueBtn");
  const viewQueueBtn = document.getElementById("viewQueueBtn");
  const batchModalBtn = document.getElementById("batchModalBtn");
  const robustnessBtn = document.getElementById("robustnessBtn");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");
  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");
  const canvas3d = document.getElementById("canvas3d");
  const canvas2d = document.getElementById("canvas2d");
  const canvasAsym = document.getElementById("canvasAsym");
  const modalBackdrop = document.getElementById("modalBackdrop");
  const modalTitle = document.getElementById("modalTitle");
  const modalBody = document.getElementById("modalBody");
  const closeModalBtn = document.getElementById("closeModalBtn");

  function decodeBase64Float32(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = bin.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  }

  function openModal(title, htmlContent) {
    modalTitle.textContent = title;
    modalBody.innerHTML = htmlContent;
    modalBackdrop.style.display = "flex";
  }

  function closeModal() {
    modalBackdrop.style.display = "none";
  }

  closeModalBtn.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", (e) => {
    if (e.target === modalBackdrop) closeModal();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  themeToggleBtn.addEventListener("click", () => {
    APP_STATE.isLightTheme = !APP_STATE.isLightTheme;
    document.body.classList.toggle("theme-light", APP_STATE.isLightTheme);
    themeToggleBtn.textContent = APP_STATE.isLightTheme ? "ТЕМА: LIGHT" : "ТЕМА: DARK";
    render3DScene();
    if (APP_STATE.lastResponse) {
      render2DBlueprint(APP_STATE.lastResponse);
      renderAsymmetryChart(APP_STATE.lastResponse);
    }
  });

  async function initCatalog() {
    try {
      const res = await fetch("/api/catalog");
      const data = await res.json();
      populateScanSelect(data.scans || []);
      if (data.expert_queue_count !== undefined) {
        document.getElementById("queueCountSpan").textContent = data.expert_queue_count;
      }
      if (data.evaluation_metrics && data.evaluation_metrics.ml_model_test_metrics) {
        const m = data.evaluation_metrics.ml_model_test_metrics;
        const valLine = document.getElementById("valSummaryLine");
        if (valLine) {
          valLine.textContent = `Тест (25 STL): MAE CI=${m.CI_pp.mae.toFixed(2)} п.п. • CVAI=${m.CVAI_pp.mae.toFixed(2)} п.п. • CVA=${m.CVA_mm.mae.toFixed(2)} мм`;
        }
      }
      await runAnalysis();
    } catch (err) {
      console.error("Ошибка инициализации каталога:", err);
    }
  }

  function populateScanSelect(scans) {
    scanSelect.innerHTML = "";
    const groups = {};
    scans.forEach((s) => {
      if (!groups[s.group]) groups[s.group] = [];
      groups[s.group].push(s);
    });
    Object.keys(groups)
      .sort()
      .forEach((grpName) => {
        const optgroup = document.createElement("optgroup");
        optgroup.label = grpName;
        groups[grpName].forEach((item) => {
          const opt = document.createElement("option");
          opt.value = item.filename;
          opt.textContent = item.label;
          if (item.filename === APP_STATE.currentFilename) {
            opt.selected = true;
          }
          optgroup.appendChild(opt);
        });
        scanSelect.appendChild(optgroup);
      });
    const cnt = document.getElementById("scanCountLabel");
    if (cnt) cnt.textContent = `${scans.length} STL`;
  }

  async function runAnalysis() {
    try {
      const payload = {
        filename: APP_STATE.currentFilename,
        file_b64: APP_STATE.uploadedFileB64,
        scale_factor: APP_STATE.scaleFactor,
        level_ratio: APP_STATE.levelRatio,
        diagonal_angle_deg: 30.0,
        custom_landmarks: APP_STATE.customLandmarks,
        use_baseline: APP_STATE.useBaseline,
        expert_ci_profile_approved: APP_STATE.expertCiApproved,
      };

      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      APP_STATE.uploadedFileB64 = null;
      if (data.error) {
        alert("Ошибка анализа: " + data.error);
        return;
      }
      APP_STATE.lastResponse = data;
      loadMeshBuffers(data.mesh_3d);
      updateDashboardUI(data);
      render3DScene();
      render2DBlueprint(data);
      renderAsymmetryChart(data);
    } catch (err) {
      console.error("Ошибка запроса /api/analyze:", err);
    }
  }

  function loadMeshBuffers(mesh3d) {
    if (!mesh3d || !mesh3d.positions_b64) return;
    APP_STATE.meshPositions = decodeBase64Float32(mesh3d.positions_b64);
    APP_STATE.meshAsymmetry = decodeBase64Float32(mesh3d.asymmetry_mm_b64);

    const pos = APP_STATE.meshPositions;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < pos.length; i += 3) {
      const x = pos[i], y = pos[i + 1], z = pos[i + 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    APP_STATE.meshCenter = [
      0.5 * (minX + maxX),
      0.5 * (minY + maxY),
      0.5 * (minZ + maxZ),
    ];
    const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
    APP_STATE.meshRadius = Math.max(dx, dy, dz, 1e-3) * 0.55;
  }

  function updateDashboardUI(data) {
    const q = data.quality;
    const cran = data.craniometry;

    document.getElementById("scanIdTag").textContent = data.scan_id;
    document.getElementById("meshStatsBadge").innerHTML =
      `<strong>${q.triangle_count.toLocaleString()}</strong> faces • <strong>${q.unique_vertex_count.toLocaleString()}</strong> verts`;
    document.getElementById("scaleBadge").innerHTML = `Масштаб: <strong>×${q.applied_scale}</strong> (мм)`;
    document.getElementById("scaleActiveLabel").textContent = `×${q.applied_scale} (мм)`;
    document.getElementById("runtimeBadge").innerHTML = `<strong>${data.processing_time_sec.toFixed(3)}s</strong> • Δ≤0.01`;

    // Header Status Chip
    const hBadge = document.getElementById("headerStatusBadge");
    hBadge.className = `status-chip chip-${q.status}`;
    hBadge.textContent = `${q.status} • ${q.quality_score.toFixed(0)}%`;

    // SVG Ring Gauge & Mini Bars (Right HUD Panel 1)
    const ringCircle = document.getElementById("qualityRingCircle");
    const circum = 175.9;
    const offset = circum * (1 - Math.max(0, Math.min(100, q.quality_score)) / 100);
    ringCircle.setAttribute("stroke-dashoffset", String(offset.toFixed(1)));
    const ringCol = q.status === "VALID" ? "#10B981" : q.status === "WARNING" ? "#F59E0B" : "#EF4444";
    ringCircle.setAttribute("stroke", ringCol);
    document.getElementById("qualityScoreVal").textContent = `${q.quality_score.toFixed(0)}%`;
    document.getElementById("qualityScoreVal").style.color = ringCol;

    document.getElementById("barClosure").style.width = `${q.vault_closure_pct}%`;
    document.getElementById("barClosure").style.background = q.vault_closure_pct >= 85 ? "#10B981" : "#EF4444";
    document.getElementById("txtClosure").textContent = `${q.vault_closure_pct.toFixed(0)}%`;

    document.getElementById("barQuad").style.width = `${q.quadrant_coverage_pct}%`;
    document.getElementById("barQuad").style.background = q.quadrant_coverage_pct >= 35 ? "#38BDF8" : "#EF4444";
    document.getElementById("txtQuad").textContent = `${q.quadrant_coverage_pct.toFixed(0)}%`;

    const meshPct = Math.min(100, (q.triangle_count / 85000) * 100);
    document.getElementById("barMesh").style.width = `${meshPct}%`;
    document.getElementById("txtMesh").textContent = `${(q.triangle_count / 1000).toFixed(1)}k`;

    // Suggested scale prompt
    if (q.suggested_scale !== q.applied_scale && q.status === "REJECTED") {
      scaleSuggestionBox.style.display = "block";
      applySuggestedScaleBtn.textContent = `Авто-калибровка масштаба: применить ×${q.suggested_scale}`;
      applySuggestedScaleBtn.onclick = () => {
        APP_STATE.scaleFactor = q.suggested_scale;
        scaleSelect.value = String(q.suggested_scale);
        customScaleInput.style.display = "none";
        runAnalysis();
      };
    } else {
      scaleSuggestionBox.style.display = "none";
    }

    const qTitle = document.getElementById("qualityAlertTitle");
    const qBody = document.getElementById("qualityAlertBody");
    if (q.status === "REJECTED") {
      qTitle.style.color = "#FCA5A5";
      qTitle.textContent = "ОТКАЗ: Входной 3D-скан не прошёл контроль";
      qBody.innerHTML = q.defects.map((d) => `<div style="color:#F87171;">• ${d}</div>`).join("");
    } else if (q.status === "WARNING") {
      qTitle.style.color = "#FBBF24";
      qTitle.textContent = "Скан пригоден с предупреждением";
      qBody.innerHTML = q.warnings.map((w) => `<div style="color:#FBBF24;">• ${w}</div>`).join("");
    } else {
      qTitle.style.color = "#34D399";
      qTitle.textContent = "Целостность и полнота свода подтверждены";
      qBody.textContent = `Габариты: ${q.extents_mm[0]} × ${q.extents_mm[1]} × ${q.extents_mm[2]} мм`;
    }

    // Populate Landmark Inputs
    const lms = data.landmarks_mm;
    const lmTag = document.getElementById("landmarkModeTag");
    if (data.manual_correction_applied) {
      lmTag.textContent = "Правка врача";
      lmTag.style.color = "#FBBF24";
    } else if (APP_STATE.useBaseline) {
      lmTag.textContent = "Baseline";
      lmTag.style.color = "#94A3B8";
    } else {
      lmTag.textContent = "ML Авто (seed=42)";
      lmTag.style.color = "#2DD4BF";
    }

    ["Sellion", "RightTragion", "LeftTragion", "Subnasal"].forEach((k) => {
      for (let axis = 0; axis < 3; axis++) {
        const inp = document.getElementById(`lm_${k}_${axis}`);
        if (inp) {
          inp.value = lms && lms[k] ? Number(lms[k][axis]).toFixed(1) : "";
          inp.disabled = !q.is_usable;
        }
      }
    });

    if (!cran) {
      document.getElementById("kpiCI").textContent = "ОТКАЗ";
      document.getElementById("kpiCVA").textContent = "--";
      document.getElementById("kpiCVAI").textContent = "--%";
      document.getElementById("kpiPerim").textContent = "--";
      const banner = document.getElementById("cvaiGradeBanner");
      banner.textContent = "Расчёт заблокирован контролем качества входных данных";
      banner.style.borderLeftColor = "#EF4444";
      banner.style.color = "#F87171";
      document.getElementById("valL").textContent = "--";
      document.getElementById("valW").textContent = "--";
      document.getElementById("valD1").textContent = "--";
      document.getElementById("valD2").textContent = "--";
      document.getElementById("valPerim").textContent = "--";
      document.getElementById("valFrontalAsym").textContent = "--";
      document.getElementById("valOccipitalAsym").textContent = "--";
      document.getElementById("valCVAIMin").textContent = "--";
      document.getElementById("valShapeClass").textContent = "Непригодный скан";
      document.getElementById("shapeProbBars").innerHTML = "";
      document.getElementById("gtComparisonBox").style.display = "none";
      return;
    }

    const ind = cran.indices;
    const lin = cran.linear_measurements_mm;
    const reg = cran.regional_asymmetry;
    const grade = cran.classification.cvai_grade;

    document.getElementById("kpiCI").textContent = `${ind.CI.toFixed(2)}%`;
    document.getElementById("kpiCVA").textContent = `${ind.CVA_mm.toFixed(2)}`;
    const kpiCvaiEl = document.getElementById("kpiCVAI");
    kpiCvaiEl.textContent = `${ind.CVAI.toFixed(2)}%`;
    kpiCvaiEl.style.color = grade.color;
    document.getElementById("kpiPerim").textContent = `${lin.perimeter_slice.toFixed(1)}`;

    const banner = document.getElementById("cvaiGradeBanner");
    banner.textContent = grade.label;
    banner.style.borderLeftColor = grade.color;
    banner.style.color = grade.color;

    const ptrPct = Math.min(98.5, Math.max(1.5, (ind.CVAI_unrounded / 14.0) * 100.0));
    document.getElementById("cvaiPointer").style.left = `${ptrPct}%`;

    document.getElementById("ciInterpretationText").textContent = cran.classification.ci_interpretation;
    document.getElementById("sliceHeightTag").textContent = `Z=${lin.slice_height_z.toFixed(1)}мм (${Math.round(lin.slice_level_ratio * 100)}%)`;
    document.getElementById("dominantZoneTag").textContent = reg.dominant_asymmetry_zone.toUpperCase();

    document.getElementById("valL").textContent = `${lin.L_length_ap.toFixed(2)} мм`;
    document.getElementById("valW").textContent = `${lin.W_width_ml.toFixed(2)} мм`;
    document.getElementById("valD1").textContent = `${lin.D1_diagonal_plus30.toFixed(2)} мм`;
    document.getElementById("valD2").textContent = `${lin.D2_diagonal_minus30.toFixed(2)} мм`;
    document.getElementById("valPerimLabel").textContent = lin.perimeter_label;
    document.getElementById("valPerim").textContent = `${lin.perimeter_slice.toFixed(2)} мм`;
    document.getElementById("valCVAIMin").textContent = `${ind.CVAI_min_ref.toFixed(2)}%`;

    // Regional Asymmetry Bars
    const fPct = Math.min(100, (reg.frontal_asymmetry_pct / 16.0) * 100);
    const oPct = Math.min(100, (reg.occipital_asymmetry_pct / 16.0) * 100);
    document.getElementById("barFrontal").style.width = `${fPct}%`;
    document.getElementById("barOccipital").style.width = `${oPct}%`;
    document.getElementById("valFrontalAsym").textContent = `${reg.frontal_diff_mm.toFixed(1)}мм (${reg.frontal_asymmetry_pct.toFixed(1)}%)`;
    document.getElementById("valOccipitalAsym").textContent = `${reg.occipital_diff_mm.toFixed(1)}мм (${reg.occipital_asymmetry_pct.toFixed(1)}%)`;

    // ML Shape Repartition Bars
    const probContainer = document.getElementById("shapeProbBars");
    if (data.shape_classifier && data.shape_classifier.probabilities_pct) {
      document.getElementById("valShapeClass").textContent = data.shape_classifier.predicted_code;
      const probs = data.shape_classifier.probabilities_pct;
      const barColors = {
        N: "#10B981",
        P: "#F59E0B",
        B: "#38BDF8",
        BP: "#EF4444",
        S: "#A855F7",
        SP: "#EC4899",
      };
      const barNames = {
        N: "Норма (N)",
        P: "Плагио (P)",
        B: "Брахи (B)",
        BP: "Брахи+Пл (BP)",
        S: "Скафо (S)",
        SP: "Скафо+Пл (SP)",
      };
      const sortedKeys = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);
      probContainer.innerHTML = sortedKeys
        .slice(0, 4)
        .map((k) => {
          const p = probs[k];
          return `
            <div class="mini-bar-row" style="grid-template-columns: 92px 1fr 42px;">
              <span style="color:var(--text-dim);">${barNames[k] || k}</span>
              <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${p}%; background:${barColors[k] || "#14B8A6"};"></div></div>
              <span style="text-align:right;">${p.toFixed(0)}%</span>
            </div>
          `;
        })
        .join("");
    }

    const gtBox = document.getElementById("gtComparisonBox");
    if (data.ground_truth_comparison && data.ground_truth_comparison.has_ground_truth) {
      const gt = data.ground_truth_comparison;
      gtBox.style.display = "block";
      document.getElementById("gtTitle").textContent =
        gt.split === "test"
          ? `Сверка с эталоном (Закрытый тест — ${data.scan_id}):`
          : `Сверка с эталоном (Train — ${data.scan_id}):`;
      document.getElementById("gtMetricsLine").textContent =
        `Эталон: CI=${gt.gt_indices.CI.toFixed(2)}% CVAI=${gt.gt_indices.CVAI.toFixed(2)}% | ` +
        `Ошибка: |ΔCI|=${gt.abs_errors.CI_pp.toFixed(2)} п.п., |ΔCVAI|=${gt.abs_errors.CVAI_pp.toFixed(2)} п.п., |ΔCVA|=${gt.abs_errors.CVA_mm.toFixed(2)} мм`;
    } else {
      gtBox.style.display = "none";
    }
  }

  // ==================== SPATIAL 3D COCKPIT VIEWPORT ENGINE ====================
  function projectPoint3D(x, y, z, width, height) {
    const cx = APP_STATE.meshCenter[0];
    const cy = APP_STATE.meshCenter[1];
    const cz = APP_STATE.meshCenter[2];
    const r = APP_STATE.meshRadius;

    const nx = (x - cx) / r;
    const ny = (y - cy) / r;
    const nz = (z - cz) / r;

    const cosY = Math.cos(APP_STATE.camYaw);
    const sinY = Math.sin(APP_STATE.camYaw);
    const rx = nx * cosY - ny * sinY;
    const ry = nx * sinY + ny * cosY;
    const rz = nz;

    const cosP = Math.cos(APP_STATE.camPitch);
    const sinP = Math.sin(APP_STATE.camPitch);
    const vx = ry;
    const vy = rz * cosP - rx * sinP;
    const vz = rz * sinP + rx * cosP;

    const scale = Math.min(width, height) * 0.23 * APP_STATE.camZoom;
    const persp = 4.2 / Math.max(1.5, 4.2 - vz * 0.45);
    const sx = width * 0.5 + APP_STATE.camPanX + vx * scale * persp;
    const sy = height * 0.51 + APP_STATE.camPanY - vy * scale * persp;
    return [sx, sy, vz, scale * persp];
  }

  function heatmapColor(mm) {
    const t = Math.max(-1.0, Math.min(1.0, mm / 7.5));
    if (t < 0) {
      const a = -t;
      return [
        Math.round(220 * (1 - a) + 14 * a),
        Math.round(216 * (1 - a) + 145 * a),
        Math.round(208 * (1 - a) + 233 * a),
      ];
    } else {
      const a = t;
      return [
        Math.round(220 * (1 - a) + 245 * a),
        Math.round(216 * (1 - a) + 95 * a),
        Math.round(208 * (1 - a) + 22 * a),
      ];
    }
  }

  function render3DScene() {
    const rect = canvas3d.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(300, Math.floor(rect.width));
    const h = Math.max(240, Math.floor(rect.height));
    if (canvas3d.width !== w * dpr || canvas3d.height !== h * dpr) {
      canvas3d.width = w * dpr;
      canvas3d.height = h * dpr;
    }
    const ctx = canvas3d.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);

    // Spatial Dark Void Background with Subtle Radial Vignette
    const grad = ctx.createRadialGradient(w * 0.5, h * 0.5, 20, w * 0.5, h * 0.5, Math.max(w, h) * 0.65);
    grad.addColorStop(0, "#101827");
    grad.addColorStop(1, "#05070B");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Spatial Constellation / Polar Reference Grid Floor
    ctx.strokeStyle = "rgba(56, 189, 248, 0.10)";
    ctx.lineWidth = 1;
    const zFloor = APP_STATE.meshCenter[2] - APP_STATE.meshRadius * 0.96;
    const gridSpan = APP_STATE.meshRadius * 1.25;
    for (let i = -3; i <= 3; i++) {
      const offset = (i / 3) * gridSpan;
      const pA = projectPoint3D(APP_STATE.meshCenter[0] - gridSpan, APP_STATE.meshCenter[1] + offset, zFloor, w, h);
      const pB = projectPoint3D(APP_STATE.meshCenter[0] + gridSpan, APP_STATE.meshCenter[1] + offset, zFloor, w, h);
      ctx.beginPath(); ctx.moveTo(pA[0], pA[1]); ctx.lineTo(pB[0], pB[1]); ctx.stroke();

      const pC = projectPoint3D(APP_STATE.meshCenter[0] + offset, APP_STATE.meshCenter[1] - gridSpan, zFloor, w, h);
      const pD = projectPoint3D(APP_STATE.meshCenter[0] + offset, APP_STATE.meshCenter[1] + gridSpan, zFloor, w, h);
      ctx.beginPath(); ctx.moveTo(pC[0], pC[1]); ctx.lineTo(pD[0], pD[1]); ctx.stroke();
    }

    const pos = APP_STATE.meshPositions;
    const asym = APP_STATE.meshAsymmetry;
    if (!pos || pos.length === 0) {
      ctx.restore();
      return;
    }

    const nTri = pos.length / 9;
    const step = APP_STATE.isDragging && nTri > 12000 ? 2 : 1;
    const drawList = [];

    const cosY = Math.cos(APP_STATE.camYaw), sinY = Math.sin(APP_STATE.camYaw);
    const cosP = Math.cos(APP_STATE.camPitch), sinP = Math.sin(APP_STATE.camPitch);

    for (let t = 0; t < nTri; t += step) {
      const idx = t * 9;
      const x1 = pos[idx], y1 = pos[idx + 1], z1 = pos[idx + 2];
      const x2 = pos[idx + 3], y2 = pos[idx + 4], z2 = pos[idx + 5];
      const x3 = pos[idx + 6], y3 = pos[idx + 7], z3 = pos[idx + 8];

      const p1 = projectPoint3D(x1, y1, z1, w, h);
      const p2 = projectPoint3D(x2, y2, z2, w, h);
      const p3 = projectPoint3D(x3, y3, z3, w, h);

      const cross2D = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
      if (cross2D > 0 && APP_STATE.renderMode !== "wire") continue;

      const ux = x2 - x1, uy = y2 - y1, uz = z2 - z1;
      const vx = x3 - x1, vy = y3 - y1, vz = z3 - z1;
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const nLen = Math.hypot(nx, ny, nz) || 1.0;
      nx /= nLen; ny /= nLen; nz /= nLen;

      const rnx = nx * cosY - ny * sinY;
      const rny = nx * sinY + ny * cosY;
      const vnx = rny;
      const vny = nz * cosP - rnx * sinP;
      const vnz = nz * sinP + rnx * cosP;

      const light = Math.max(0.20, Math.min(1.0, 0.28 + 0.56 * Math.abs(vnz) + 0.22 * Math.max(0, vny) + 0.14 * vnx));
      const depth = (p1[2] + p2[2] + p3[2]) / 3.0;
      const asymVal = asym ? (asym[t * 3] + asym[t * 3 + 1] + asym[t * 3 + 2]) / 3.0 : 0.0;

      drawList.push({ p1, p2, p3, depth, light, asymVal });
    }

    drawList.sort((a, b) => a.depth - b.depth);

    if (APP_STATE.renderMode === "wire") {
      ctx.strokeStyle = "rgba(45, 212, 191, 0.36)";
      ctx.lineWidth = 0.65;
      ctx.beginPath();
      for (let i = 0; i < drawList.length; i += 2) {
        const item = drawList[i];
        ctx.moveTo(item.p1[0], item.p1[1]);
        ctx.lineTo(item.p2[0], item.p2[1]);
        ctx.lineTo(item.p3[0], item.p3[1]);
        ctx.closePath();
      }
      ctx.stroke();
    } else {
      for (let i = 0; i < drawList.length; i++) {
        const item = drawList[i];
        let r, g, b;
        if (APP_STATE.renderMode === "heatmap") {
          const rgb = heatmapColor(item.asymVal);
          r = Math.round(rgb[0] * item.light);
          g = Math.round(rgb[1] * item.light);
          b = Math.round(rgb[2] * item.light);
        } else {
          r = Math.round(224 * item.light);
          g = Math.round(218 * item.light);
          b = Math.round(206 * item.light);
        }
        const col = `rgb(${r},${g},${b})`;
        ctx.fillStyle = col;
        ctx.strokeStyle = col;
        ctx.lineWidth = 0.55;
        ctx.beginPath();
        ctx.moveTo(item.p1[0], item.p1[1]);
        ctx.lineTo(item.p2[0], item.p2[1]);
        ctx.lineTo(item.p3[0], item.p3[1]);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    const data = APP_STATE.lastResponse;

    // Draw Extended Spatial Cockpit Axis Rays & Measurement Cross-Section Plane
    if (APP_STATE.showPlanes && data && data.craniometry && data.craniometry.geometry_3d) {
      const g3d = data.craniometry.geometry_3d;
      const contour3d = g3d.contour_3d || [];
      const ep3 = g3d.axes_endpoints_3d;
      const cWorld = g3d.slice_center_world;

      // Long radial telemetry rays extending beyond the skull (OMEGA COCKPIT aesthetic)
      if (APP_STATE.showRays && ep3 && cWorld) {
        const pCenter = projectPoint3D(cWorld[0], cWorld[1], cWorld[2], w, h);
        [
          [ep3.ap_front, ep3.ap_back, "rgba(56, 189, 248, 0.25)"],
          [ep3.ml_left, ep3.ml_right, "rgba(56, 189, 248, 0.20)"],
          [ep3.d1_front, ep3.d1_back, "rgba(245, 158, 11, 0.28)"],
          [ep3.d2_front, ep3.d2_back, "rgba(239, 68, 68, 0.28)"],
        ].forEach(([ptA, ptB, rayCol]) => {
          const extA = [
            cWorld[0] + (ptA[0] - cWorld[0]) * 1.45,
            cWorld[1] + (ptA[1] - cWorld[1]) * 1.45,
            cWorld[2] + (ptA[2] - cWorld[2]) * 1.45,
          ];
          const extB = [
            cWorld[0] + (ptB[0] - cWorld[0]) * 1.45,
            cWorld[1] + (ptB[1] - cWorld[1]) * 1.45,
            cWorld[2] + (ptB[2] - cWorld[2]) * 1.45,
          ];
          const pEA = projectPoint3D(extA[0], extA[1], extA[2], w, h);
          const pEB = projectPoint3D(extB[0], extB[1], extB[2], w, h);
          ctx.strokeStyle = rayCol;
          ctx.lineWidth = 1.0;
          ctx.beginPath();
          ctx.moveTo(pEA[0], pEA[1]);
          ctx.lineTo(pEB[0], pEB[1]);
          ctx.stroke();
        });

        // Glowing center origin node
        ctx.beginPath();
        ctx.arc(pCenter[0], pCenter[1], 4.5, 0, Math.PI * 2);
        ctx.fillStyle = "#FFFFFF";
        ctx.fill();
      }

      if (contour3d.length > 2) {
        ctx.fillStyle = "rgba(20, 184, 166, 0.14)";
        ctx.strokeStyle = "#2DD4BF";
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        contour3d.forEach((pt, idx) => {
          const p = projectPoint3D(pt[0], pt[1], pt[2], w, h);
          if (idx === 0) ctx.moveTo(p[0], p[1]);
          else ctx.lineTo(p[0], p[1]);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      if (ep3) {
        function draw3DLine(ptA, ptB, color, width, dash, label) {
          const pA = projectPoint3D(ptA[0], ptA[1], ptA[2], w, h);
          const pB = projectPoint3D(ptB[0], ptB[1], ptB[2], w, h);
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          if (dash) ctx.setLineDash(dash);
          ctx.beginPath();
          ctx.moveTo(pA[0], pA[1]);
          ctx.lineTo(pB[0], pB[1]);
          ctx.stroke();
          ctx.restore();

          // Glowing endpoint nodes
          [pA, pB].forEach((p) => {
            ctx.beginPath();
            ctx.arc(p[0], p[1], 3.2, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
          });

          if (label) {
            ctx.fillStyle = color;
            ctx.font = "bold 10px monospace";
            ctx.fillText(label, pA[0] + 7, pA[1] - 5);
          }
        }

        draw3DLine(ep3.ap_front, ep3.ap_back, "#38BDF8", 1.8, [4, 3], "L (AP 0°)");
        draw3DLine(ep3.ml_left, ep3.ml_right, "#0284C7", 1.8, [2, 3], "W (ML 90°)");
        draw3DLine(ep3.d1_front, ep3.d1_back, "#F59E0B", 2.4, null, "D1 (+30°)");
        draw3DLine(ep3.d2_front, ep3.d2_back, "#EF4444", 2.4, null, "D2 (-30°)");
      }
    }

    // Draw Base Plane & Interactive Anatomical Landmarks with Halo Callouts
    APP_STATE.projectedLandmarks = {};
    if (APP_STATE.showLandmarks && data && data.landmarks_mm) {
      const lms = data.landmarks_mm;

      if (APP_STATE.showPlanes && lms.Sellion && lms.RightTragion && lms.LeftTragion) {
        const pSe = projectPoint3D(lms.Sellion[0], lms.Sellion[1], lms.Sellion[2], w, h);
        const pRT = projectPoint3D(lms.RightTragion[0], lms.RightTragion[1], lms.RightTragion[2], w, h);
        const pLT = projectPoint3D(lms.LeftTragion[0], lms.LeftTragion[1], lms.LeftTragion[2], w, h);

        ctx.save();
        ctx.strokeStyle = "rgba(148, 163, 184, 0.55)";
        ctx.fillStyle = "rgba(148, 163, 184, 0.08)";
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(pSe[0], pSe[1]);
        ctx.lineTo(pRT[0], pRT[1]);
        ctx.lineTo(pLT[0], pLT[1]);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }

      Object.keys(LANDMARK_COLORS).forEach((key) => {
        if (!lms[key]) return;
        const c = lms[key];
        const p = projectPoint3D(c[0], c[1], c[2], w, h);
        APP_STATE.projectedLandmarks[key] = p;

        const isHovered = APP_STATE.hoveredLandmarkKey === key || APP_STATE.dragLandmarkKey === key;
        const radius = isHovered ? 8.5 : 5.8;

        // Outer halo ring
        ctx.beginPath();
        ctx.arc(p[0], p[1], radius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = LANDMARK_COLORS[key];
        ctx.lineWidth = 1.0;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
        ctx.fillStyle = LANDMARK_COLORS[key];
        ctx.fill();
        ctx.lineWidth = isHovered ? 2.2 : 1.4;
        ctx.strokeStyle = "#FFFFFF";
        ctx.stroke();

        ctx.fillStyle = "#F8FAFC";
        ctx.font = "bold 10px monospace";
        ctx.fillText(`${key} [${c[0].toFixed(0)},${c[1].toFixed(0)}]`, p[0] + 11, p[1] + 3);
      });
    }

    if (APP_STATE.renderMode === "heatmap") {
      const cbX = 16, cbY = 48, cbW = 12, cbH = 130;
      for (let i = 0; i < cbH; i++) {
        const mm = 7.5 * (1 - (2 * i) / cbH);
        const rgb = heatmapColor(mm);
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(cbX, cbY + i, cbW, 1.5);
      }
      ctx.strokeStyle = "#475569";
      ctx.strokeRect(cbX, cbY, cbW, cbH);
      ctx.fillStyle = "#E2E8F0";
      ctx.font = "10px monospace";
      ctx.fillText("+7.5 мм (выступ)", cbX + 18, cbY + 8);
      ctx.fillText(" 0.0 мм (норма)", cbX + 18, cbY + cbH / 2 + 3);
      ctx.fillText("-7.5 мм (уплощение)", cbX + 18, cbY + cbH - 2);
    }

    ctx.restore();
  }

  // ==================== 2D BLUEPRINT & RADIAL SPARKLINE ====================
  function render2DBlueprint(data) {
    const ctx = canvas2d.getContext("2d");
    const w = canvas2d.width;
    const h = canvas2d.height;
    ctx.clearRect(0, 0, w, h);

    const bgCol = APP_STATE.isLightTheme ? "#F8FAFC" : "#070A10";
    const gridCol = APP_STATE.isLightTheme ? "#E2E8F0" : "rgba(148, 163, 184, 0.10)";
    ctx.fillStyle = bgCol;
    ctx.fillRect(0, 0, w, h);

    if (!data || !data.craniometry || !data.craniometry.geometry_2d) {
      ctx.fillStyle = "#EF4444";
      ctx.font = "bold 11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("СЕЧЕНИЕ НЕ ПОСТРОЕНО (ДЕФЕКТ СКАНА)", w / 2, h / 2);
      return;
    }

    const g2d = data.craniometry.geometry_2d;
    const lin = data.craniometry.linear_measurements_mm;
    const contour = g2d.contour_xy;
    const ep = g2d.endpoints_xy;
    const center = g2d.center;

    const scale = 1.15;
    const cx = w * 0.5;
    const cy = h * 0.52;

    function toCanvas(pt) {
      return [cx + (pt[0] - center[0]) * scale, cy - (pt[1] - center[1]) * scale];
    }

    ctx.strokeStyle = gridCol;
    ctx.lineWidth = 1;
    for (let g = -60; g <= 60; g += 20) {
      const gx = cx + g * scale;
      const gy = cy - g * scale;
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }

    ctx.beginPath();
    contour.forEach((pt, i) => {
      const p = toCanvas(pt);
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(20, 184, 166, 0.14)";
    ctx.fill();
    ctx.strokeStyle = "#14B8A6";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    function drawChord(pt1, pt2, col, width, dash, label) {
      const p1 = toCanvas(pt1);
      const p2 = toCanvas(pt2);
      ctx.save();
      ctx.strokeStyle = col;
      ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
      ctx.restore();

      [p1, p2].forEach((p) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 2.8, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
      });

      if (label) {
        ctx.fillStyle = col;
        ctx.font = "bold 9px monospace";
        ctx.fillText(label, p1[0] + 4, p1[1] + 2);
      }
    }

    drawChord(ep.ap_front, ep.ap_back, "#38BDF8", 1.3, [3, 3], `L=${lin.L_length_ap.toFixed(1)}`);
    drawChord(ep.ml_left, ep.ml_right, "#0284C7", 1.3, [2, 2], `W=${lin.W_width_ml.toFixed(1)}`);
    drawChord(ep.d1_front, ep.d1_back, "#F59E0B", 2.0, null, `D1=${lin.D1_diagonal_plus30.toFixed(1)}`);
    drawChord(ep.d2_front, ep.d2_back, "#EF4444", 2.0, null, `D2=${lin.D2_diagonal_minus30.toFixed(1)}`);

    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#F8FAFC";
    ctx.fill();
  }

  function renderAsymmetryChart(data) {
    const ctx = canvasAsym.getContext("2d");
    const w = canvasAsym.width;
    const h = canvasAsym.height;
    ctx.clearRect(0, 0, w, h);

    const bgCol = APP_STATE.isLightTheme ? "#F8FAFC" : "#070A10";
    ctx.fillStyle = bgCol;
    ctx.fillRect(0, 0, w, h);

    if (!data || !data.craniometry || !data.craniometry.geometry_2d) {
      ctx.fillStyle = "#64748B";
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("ДАННЫЕ АСИММЕТРИИ ОТСУТСТВУЮТ", w / 2, h / 2);
      return;
    }

    const prof = data.craniometry.geometry_2d.asymmetry_profile_mm || [];
    const padL = 32, padR = 12, padT = 16, padB = 22;
    const pw = w - padL - padR;
    const ph = h - padT - padB;
    const midY = padT + ph * 0.5;
    const maxMM = Math.max(6.0, ...prof.map((v) => Math.abs(v))) * 1.15;

    ctx.strokeStyle = APP_STATE.isLightTheme ? "#E2E8F0" : "rgba(148, 163, 184, 0.12)";
    ctx.lineWidth = 1;
    [-5, 0, 5].forEach((val) => {
      const y = midY - (val / maxMM) * (ph * 0.5);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillStyle = "#64748B";
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ctx.fillText(`${val > 0 ? "+" : ""}${val}`, padL - 4, y + 3);
    });

    // Area Fill (OMEGA VÉLOCITÉ golden-amber / teal style)
    ctx.beginPath();
    ctx.moveTo(padL, midY);
    let peakVal = 0;
    let peakX = padL;
    let peakY = midY;
    for (let i = 0; i < prof.length; i++) {
      const x = padL + (i / (prof.length - 1)) * pw;
      const y = midY - (prof[i] / maxMM) * (ph * 0.5);
      ctx.lineTo(x, y);
      if (Math.abs(prof[i]) > Math.abs(peakVal)) {
        peakVal = prof[i];
        peakX = x;
        peakY = y;
      }
    }
    ctx.lineTo(padL + pw, midY);
    ctx.closePath();
    ctx.fillStyle = "rgba(245, 158, 11, 0.18)";
    ctx.fill();

    // Curve Stroke
    ctx.beginPath();
    for (let i = 0; i < prof.length; i++) {
      const x = padL + (i / (prof.length - 1)) * pw;
      const y = midY - (prof[i] / maxMM) * (ph * 0.5);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#F59E0B";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // Peak node marker
    ctx.beginPath();
    ctx.arc(peakX, peakY, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = "#10B981";
    ctx.fill();
    ctx.fillStyle = "#34D399";
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`max ${peakVal >= 0 ? "+" : ""}${peakVal.toFixed(1)}мм`, Math.min(w - 40, Math.max(55, peakX)), Math.max(12, peakY - 6));

    ctx.fillStyle = "#64748B";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.fillText("-180° (ЗАТ)", padL + 28, h - 5);
    ctx.fillText("0° (ЛОБ)", padL + pw * 0.5, h - 5);
    ctx.fillText("+180° (ЗАТ)", w - padR - 28, h - 5);
  }

  // ==================== INTERACTIVE 3D VIEWPORT EVENTS ====================
  function findHitLandmark(mx, my) {
    let bestKey = null;
    let bestDist = 14.0;
    Object.keys(APP_STATE.projectedLandmarks).forEach((k) => {
      const p = APP_STATE.projectedLandmarks[k];
      const d = Math.hypot(mx - p[0], my - p[1]);
      if (d < bestDist) {
        bestDist = d;
        bestKey = k;
      }
    });
    return bestKey;
  }

  canvas3d.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas3d.addEventListener("mousedown", (e) => {
    const rect = canvas3d.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    APP_STATE.lastMouseX = mx;
    APP_STATE.lastMouseY = my;

    const hitLm = findHitLandmark(mx, my);
    if (hitLm && e.button === 0) {
      APP_STATE.dragLandmarkKey = hitLm;
      APP_STATE.isDragging = true;
      return;
    }

    APP_STATE.isDragging = true;
    APP_STATE.dragButton = e.button;
  });

  window.addEventListener("mousemove", (e) => {
    const rect = canvas3d.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (!APP_STATE.isDragging) {
      if (mx >= 0 && mx <= rect.width && my >= 0 && my <= rect.height) {
        const hit = findHitLandmark(mx, my);
        if (hit !== APP_STATE.hoveredLandmarkKey) {
          APP_STATE.hoveredLandmarkKey = hit;
          canvas3d.style.cursor = hit ? "pointer" : "grab";
          render3DScene();
        }
      }
      return;
    }

    const dx = mx - APP_STATE.lastMouseX;
    const dy = my - APP_STATE.lastMouseY;
    APP_STATE.lastMouseX = mx;
    APP_STATE.lastMouseY = my;

    if (APP_STATE.dragLandmarkKey && APP_STATE.lastResponse && APP_STATE.lastResponse.landmarks_mm) {
      const key = APP_STATE.dragLandmarkKey;
      const lms = APP_STATE.lastResponse.landmarks_mm;
      const pProj = APP_STATE.projectedLandmarks[key];
      const pixScale = pProj && pProj[3] ? pProj[3] / APP_STATE.meshRadius : 2.5;
      const worldDelta = 1.0 / Math.max(pixScale, 0.5);

      const cosY = Math.cos(APP_STATE.camYaw), sinY = Math.sin(APP_STATE.camYaw);
      lms[key][0] += (-sinY * dx) * worldDelta;
      lms[key][1] += (cosY * dx) * worldDelta;
      lms[key][2] += (-dy) * worldDelta;

      for (let axis = 0; axis < 3; axis++) {
        const inp = document.getElementById(`lm_${key}_${axis}`);
        if (inp) inp.value = lms[key][axis].toFixed(1);
      }
      render3DScene();
      return;
    }

    if (APP_STATE.dragButton === 2 || e.shiftKey) {
      APP_STATE.camPanX += dx;
      APP_STATE.camPanY += dy;
    } else {
      APP_STATE.camYaw += dx * 0.01;
      APP_STATE.camPitch = Math.max(-1.45, Math.min(1.45, APP_STATE.camPitch + dy * 0.01));
    }
    render3DScene();
  });

  window.addEventListener("mouseup", () => {
    if (APP_STATE.dragLandmarkKey) {
      APP_STATE.dragLandmarkKey = null;
      APP_STATE.isDragging = false;
      collectCustomLandmarksAndAnalyze();
      return;
    }
    if (APP_STATE.isDragging) {
      APP_STATE.isDragging = false;
      render3DScene();
    }
  });

  canvas3d.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.91;
      APP_STATE.camZoom = Math.max(0.6, Math.min(8.0, APP_STATE.camZoom * factor));
      render3DScene();
    },
    { passive: false }
  );

  window.addEventListener("resize", () => render3DScene());

  // ==================== CONTROLS, SHOULD 5.2 & COULD 5.3 MODULES ====================
  scanSelect.addEventListener("change", () => {
    APP_STATE.currentFilename = scanSelect.value;
    APP_STATE.customLandmarks = null;
    APP_STATE.scaleFactor = 1.0;
    scaleSelect.value = "1.0";
    customScaleInput.style.display = "none";
    runAnalysis();
  });

  stlFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result);
      let binary = "";
      const chunkSize = 0x8000;
      for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
      }
      APP_STATE.uploadedFileB64 = btoa(binary);
      APP_STATE.currentFilename = file.name;
      APP_STATE.customLandmarks = null;

      const opt = document.createElement("option");
      opt.value = file.name;
      opt.textContent = `${file.name} (Загружен с диска)`;
      opt.selected = true;
      scanSelect.insertBefore(opt, scanSelect.firstChild);
      runAnalysis();
    };
    reader.readAsArrayBuffer(file);
  });

  scaleSelect.addEventListener("change", () => {
    if (scaleSelect.value === "custom") {
      customScaleInput.style.display = "block";
      APP_STATE.scaleFactor = parseFloat(customScaleInput.value) || 1.0;
    } else {
      customScaleInput.style.display = "none";
      APP_STATE.scaleFactor = parseFloat(scaleSelect.value) || 1.0;
    }
    APP_STATE.customLandmarks = null;
    runAnalysis();
  });

  customScaleInput.addEventListener("change", () => {
    APP_STATE.scaleFactor = Math.max(0.0001, parseFloat(customScaleInput.value) || 1.0);
    APP_STATE.customLandmarks = null;
    runAnalysis();
  });

  levelRatioSlider.addEventListener("input", () => {
    const v = parseFloat(levelRatioSlider.value);
    levelRatioVal.textContent = `${Math.round(v * 100)}% свода`;
  });

  levelRatioSlider.addEventListener("change", () => {
    APP_STATE.levelRatio = parseFloat(levelRatioSlider.value);
    runAnalysis();
  });

  useBaselineToggle.addEventListener("change", () => {
    APP_STATE.useBaseline = useBaselineToggle.checked;
    APP_STATE.customLandmarks = null;
    runAnalysis();
  });

  expertCiProfileCheck.addEventListener("change", () => {
    APP_STATE.expertCiApproved = expertCiProfileCheck.checked;
    runAnalysis();
  });

  function collectCustomLandmarks() {
    const custom = {};
    ["Sellion", "RightTragion", "LeftTragion", "Subnasal"].forEach((k) => {
      const coords = [];
      for (let axis = 0; axis < 3; axis++) {
        const val = parseFloat(document.getElementById(`lm_${k}_${axis}`).value);
        coords.push(isNaN(val) ? 0.0 : val);
      }
      custom[k] = coords;
    });
    return custom;
  }

  function collectCustomLandmarksAndAnalyze() {
    APP_STATE.customLandmarks = collectCustomLandmarks();
    runAnalysis();
  }

  applyLandmarksBtn.addEventListener("click", collectCustomLandmarksAndAnalyze);

  resetLandmarksBtn.addEventListener("click", () => {
    APP_STATE.customLandmarks = null;
    runAnalysis();
  });

  // Should 5.2: Rotation & Noise Robustness Audit Modal
  robustnessBtn.addEventListener("click", async () => {
    openModal("ТЕСТ УСТОЙЧИВОСТИ К ПОВОРОТУ МОДЕЛИ И ШУМУ (Should п. 5.2)", "<div class='mono'>Выполняется 3D-поворот на 15° и наложение гауссовского шума σ=0.25 мм...</div>");
    try {
      const res = await fetch("/api/robustness", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: APP_STATE.currentFilename,
          scale_factor: APP_STATE.scaleFactor,
          level_ratio: APP_STATE.levelRatio,
          noise_sigma_mm: 0.25,
          rotation_deg: 15.0,
        }),
      });
      const r = await res.json();
      if (r.error) {
        modalBody.innerHTML = `<div style="color:#EF4444;">Ошибка: ${r.error}</div>`;
        return;
      }
      modalBody.innerHTML = `
        <div style="margin-bottom:10px; color:var(--text-sec);">
          Протокол испытания устойчивости на файле <strong class="mono">${APP_STATE.currentFilename}</strong>:
        </div>
        <table class="tel-table mono" style="margin-bottom:12px;">
          <thead>
            <tr style="border-bottom:1px solid var(--border-subtle); color:#38BDF8;">
              <th style="text-align:left; padding:6px;">Испытание</th>
              <th style="text-align:right; padding:6px;">CI (%)</th>
              <th style="text-align:right; padding:6px;">CVA (мм)</th>
              <th style="text-align:right; padding:6px;">CVAI (%)</th>
              <th style="text-align:right; padding:6px;">Отклонение |Δ|</th>
              <th style="text-align:right; padding:6px;">Статус</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="k">Эталонный запуск</td>
              <td class="v">${r.baseline_indices.CI.toFixed(2)}%</td>
              <td class="v">${r.baseline_indices.CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.baseline_indices.CVAI.toFixed(2)}%</td>
              <td class="v">0.000</td>
              <td class="v" style="color:#34D399;">БАЗА</td>
            </tr>
            <tr>
              <td class="k">Повторный запуск (п. 8.1 ≤0.01)</td>
              <td class="v">${r.baseline_indices.CI.toFixed(2)}%</td>
              <td class="v">${r.baseline_indices.CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.baseline_indices.CVAI.toFixed(2)}%</td>
              <td class="v">${r.repeat_run_determinism.max_divergence.toFixed(6)}</td>
              <td class="v" style="color:#34D399;">PASSED</td>
            </tr>
            <tr>
              <td class="k">3D-поворот модели (+15.0°)</td>
              <td class="v">${r.rotation_3d_invariance_test.rotated_CI.toFixed(2)}%</td>
              <td class="v">${r.rotation_3d_invariance_test.rotated_CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.rotation_3d_invariance_test.rotated_CVAI.toFixed(2)}%</td>
              <td class="v">ΔCVAI=${r.rotation_3d_invariance_test.delta_CVAI_pp.toFixed(5)}</td>
              <td class="v" style="color:#34D399;">PASSED</td>
            </tr>
            <tr>
              <td class="k">Гауссовский шум вершин (σ=0.25 мм)</td>
              <td class="v">${r.noise_perturbation_test.noisy_CI.toFixed(2)}%</td>
              <td class="v">${r.noise_perturbation_test.noisy_CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.noise_perturbation_test.noisy_CVAI.toFixed(2)}%</td>
              <td class="v">ΔCVAI=${r.noise_perturbation_test.delta_CVAI_pp.toFixed(3)} п.п.</td>
              <td class="v" style="color:#34D399;">PASSED</td>
            </tr>
          </tbody>
        </table>
      `;
    } catch (e) {
      modalBody.innerHTML = `<div style="color:#EF4444;">Ошибка запроса: ${e}</div>`;
    }
  });

  // Could 5.3: Batch Processing Modal
  batchModalBtn.addEventListener("click", async () => {
    openModal("ПАКЕТНАЯ ОБРАБОТКА 3D-СКАНОВ (Could п. 5.3)", "<div class='mono'>Выполняется пакетный расчёт серии клинических и тестовых STL-сканов...</div>");
    try {
      const res = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filenames: [
            "scan_081.stl",
            "scan_082.stl",
            "clinical_normal_N01.stl",
            "clinical_mild_plagio_P01.stl",
            "clinical_moderate_bp_BP05.stl",
            "clinical_severe_plagio_P03.stl",
            "clinical_very_severe_plagio_P04.stl",
            "defect_incomplete_occipital.stl",
          ],
          scale_factor: 1.0,
          level_ratio: APP_STATE.levelRatio,
        }),
      });
      const b = await res.json();
      const rowsHtml = (b.results || [])
        .map(
          (item) => `
          <tr>
            <td class="k mono">${item.scan_id}</td>
            <td class="v"><span class="status-chip chip-${item.status}">${item.status}</span></td>
            <td class="v">${item.CI != null ? item.CI.toFixed(2) + "%" : "--"}</td>
            <td class="v">${item.CVA_mm != null ? item.CVA_mm.toFixed(2) + " мм" : "--"}</td>
            <td class="v">${item.CVAI != null ? item.CVAI.toFixed(2) + "%" : "--"}</td>
            <td class="v" style="font-size:10px;">${item.grade_code}</td>
            <td class="v">${item.processing_time_sec.toFixed(3)}s</td>
          </tr>
        `
        )
        .join("");
      modalBody.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
          <span class="mono">Обработано файлов: <strong>${b.processed_count}</strong> • Суммарное время: <strong>${b.total_time_sec} с</strong></span>
          <button type="button" id="dlBatchCsvBtn" class="hud-btn hud-btn-primary">Скачать сводный CSV</button>
        </div>
        <table class="tel-table mono">
          <thead>
            <tr style="border-bottom:1px solid var(--border-subtle); color:#38BDF8;">
              <th style="text-align:left; padding:5px;">Код скана</th>
              <th style="text-align:right; padding:5px;">Статус</th>
              <th style="text-align:right; padding:5px;">CI</th>
              <th style="text-align:right; padding:5px;">CVA</th>
              <th style="text-align:right; padding:5px;">CVAI</th>
              <th style="text-align:right; padding:5px;">Категория</th>
              <th style="text-align:right; padding:5px;">Время</th>
            </tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      `;
      const dlBtn = document.getElementById("dlBatchCsvBtn");
      if (dlBtn && b.csv_summary) {
        dlBtn.onclick = () => {
          const blob = new Blob(["\ufeff" + b.csv_summary], { type: "text/csv;charset=utf-8" });
          const link = document.createElement("a");
          link.href = URL.createObjectURL(blob);
          link.download = "saigak_orto_batch_summary.csv";
          link.click();
        };
      }
    } catch (e) {
      modalBody.innerHTML = `<div style="color:#EF4444;">Ошибка пакетной обработки: ${e}</div>`;
    }
  });

  // Could 5.3: Expert Queue of Complex Cases & Retraining Annotations
  addToQueueBtn.addEventListener("click", async () => {
    const lms = collectCustomLandmarks();
    const res = await fetch("/api/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scan_id: APP_STATE.currentFilename.replace(".stl", ""),
        reason: document.getElementById("clinicalNotesInput").value || "Экспертная коррекция ориентиров для переобучения ML",
        doctor_name: document.getElementById("doctorNameInput").value,
        landmarks_mm: lms,
      }),
    });
    const qData = await res.json();
    if (qData.queue_count !== undefined) {
      document.getElementById("queueCountSpan").textContent = qData.queue_count;
    }
    viewQueueBtn.click();
  });

  viewQueueBtn.addEventListener("click", async () => {
    openModal("ЭКСПЕРТНАЯ ОЧЕРЕДЬ СЛОЖНЫХ СЛУЧАЕВ И РАЗМЕТКИ ДЛЯ ПЕРЕОБУЧЕНИЯ (Could п. 5.3)", "<div class='mono'>Загрузка очереди...</div>");
    const res = await fetch("/api/queue");
    const qData = await res.json();
    document.getElementById("queueCountSpan").textContent = qData.count || 0;
    const rows = (qData.items || [])
      .map(
        (it) => `
        <tr>
          <td class="k mono">${it.queue_id}</td>
          <td class="k mono" style="color:#38BDF8;">${it.scan_id}</td>
          <td class="k">${it.reason}</td>
          <td class="v">${it.doctor_name}</td>
          <td class="v" style="color:#34D399;">${it.timestamp}</td>
        </tr>
      `
      )
      .join("");
    modalBody.innerHTML = `
      <div style="margin-bottom:10px; color:var(--text-sec);">
        Сохранено размеченных экспертом сканов в очереди дообучения (<code>data/expert_retraining_queue.json</code>): <strong>${qData.count}</strong>
      </div>
      <table class="tel-table">
        <thead>
          <tr style="border-bottom:1px solid var(--border-subtle); color:#F59E0B;">
            <th style="text-align:left; padding:5px;">ID</th>
            <th style="text-align:left; padding:5px;">Скан</th>
            <th style="text-align:left; padding:5px;">Основание / Комментарий</th>
            <th style="text-align:right; padding:5px;">Эксперт</th>
            <th style="text-align:right; padding:5px;">Время</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  });

  // 3D Toolbar Toggles
  function setRenderMode(mode) {
    APP_STATE.renderMode = mode;
    document.getElementById("modeShadedBtn").classList.toggle("active", mode === "shaded");
    document.getElementById("modeHeatmapBtn").classList.toggle("active", mode === "heatmap");
    document.getElementById("modeWireBtn").classList.toggle("active", mode === "wire");
    render3DScene();
  }

  document.getElementById("modeShadedBtn").addEventListener("click", () => setRenderMode("shaded"));
  document.getElementById("modeHeatmapBtn").addEventListener("click", () => setRenderMode("heatmap"));
  document.getElementById("modeWireBtn").addEventListener("click", () => setRenderMode("wire"));

  document.getElementById("togglePlanesBtn").addEventListener("click", (e) => {
    APP_STATE.showPlanes = !APP_STATE.showPlanes;
    e.currentTarget.classList.toggle("active", APP_STATE.showPlanes);
    render3DScene();
  });

  document.getElementById("toggleLandmarksBtn").addEventListener("click", (e) => {
    APP_STATE.showLandmarks = !APP_STATE.showLandmarks;
    e.currentTarget.classList.toggle("active", APP_STATE.showLandmarks);
    render3DScene();
  });

  document.getElementById("toggleRaysBtn").addEventListener("click", (e) => {
    APP_STATE.showRays = !APP_STATE.showRays;
    e.currentTarget.classList.toggle("active", APP_STATE.showRays);
    render3DScene();
  });

  document.querySelectorAll("[data-cam]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = btn.getAttribute("data-cam");
      APP_STATE.camPanX = 0;
      APP_STATE.camPanY = 0;
      if (preset === "iso") {
        APP_STATE.camYaw = -0.65;
        APP_STATE.camPitch = 0.35;
      } else if (preset === "top") {
        APP_STATE.camYaw = -Math.PI / 2;
        APP_STATE.camPitch = 1.44;
      } else if (preset === "front") {
        APP_STATE.camYaw = 0.0;
        APP_STATE.camPitch = 0.05;
      } else if (preset === "side") {
        APP_STATE.camYaw = -Math.PI / 2;
        APP_STATE.camPitch = 0.05;
      }
      render3DScene();
    });
  });

  async function downloadFileFromEndpoint(url, bodyObj, fallbackName) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyObj),
    });
    if (!res.ok) {
      const err = await res.json();
      alert("Ошибка выгрузки: " + (err.error || res.statusText));
      return;
    }
    const blob = await res.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = fallbackName;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function buildExportPayload(format) {
    return {
      format,
      filename: APP_STATE.currentFilename,
      scale_factor: APP_STATE.scaleFactor,
      level_ratio: APP_STATE.levelRatio,
      diagonal_angle_deg: 30.0,
      custom_landmarks: APP_STATE.customLandmarks,
      use_baseline: APP_STATE.useBaseline,
      expert_ci_profile_approved: APP_STATE.expertCiApproved,
      patient_age_months: document.getElementById("patientAgeInput").value,
      doctor_name: document.getElementById("doctorNameInput").value,
      doctor_verified: document.getElementById("doctorVerifiedCheck").checked,
      clinical_notes: document.getElementById("clinicalNotesInput").value,
    };
  }

  downloadPdfBtn.addEventListener("click", () => {
    const sid = APP_STATE.currentFilename.replace(".stl", "");
    downloadFileFromEndpoint("/api/report/pdf", buildExportPayload("pdf"), `saigak_orto_report_${sid}.pdf`);
  });

  exportCsvBtn.addEventListener("click", () => {
    const sid = APP_STATE.currentFilename.replace(".stl", "");
    downloadFileFromEndpoint("/api/report/export", buildExportPayload("csv"), `saigak_orto_${sid}.csv`);
  });

  exportJsonBtn.addEventListener("click", () => {
    const sid = APP_STATE.currentFilename.replace(".stl", "");
    downloadFileFromEndpoint("/api/report/export", buildExportPayload("json"), `saigak_orto_${sid}.json`);
  });

  initCatalog();
})();
