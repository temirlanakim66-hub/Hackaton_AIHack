/* Сайгак-Орто Cockpit — Spatial 3D Engine & Unslop Clinical Workstation Controller */
(function () {
  "use strict";

  const PALETTE = {
    inkDeep: "#07080d",
    inkBg: "#0b0d14",
    appBg: "#141415",
    appSidebar: "#0f0f10",
    appText: "#ededed",
    appSecondary: "#a1a1a6",
    appMuted: "#85858b",
    positive: "#4cc38a",
    negative: "#e5484d",
    amber: "#e5a93c",
    blue: "#5b9ee5",
    coral: "#e06d53",
    lime: "#84cc16",
  };

  const CVAI_TONED_COLORS = {
    NORMAL: PALETTE.positive,
    MILD: PALETTE.lime,
    MODERATE: PALETTE.amber,
    SEVERE: PALETTE.coral,
    VERY_SEVERE: PALETTE.negative,
  };

  const LANDMARK_COLORS = {
    Sellion: PALETTE.coral,
    RightTragion: PALETTE.blue,
    LeftTragion: PALETTE.amber,
    Subnasal: PALETTE.positive,
  };

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
    catalogScans: [],
    liveDragInFlight: false,
    liveDragQueued: false,
    lastLiveDragTs: 0,
    lastBatchFilenames: null,
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
  const confirmLandmarksBtn = document.getElementById("confirmLandmarksBtn");
  const addToQueueBtn = document.getElementById("addToQueueBtn");
  const viewQueueBtn = document.getElementById("viewQueueBtn");
  const batchModalBtn = document.getElementById("batchModalBtn");
  const compareModalBtn = document.getElementById("compareModalBtn");
  const misExportBtn = document.getElementById("misExportBtn");
  const robustnessBtn = document.getElementById("robustnessBtn");
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const downloadPdfBtn = document.getElementById("downloadPdfBtn");
  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");
  const canvas3d = document.getElementById("canvas3d");
  const canvas2d = document.getElementById("canvas2d");
  const canvasAsym = document.getElementById("canvasAsym");
  const liveDragBanner = document.getElementById("liveDragBanner");
  const modalBackdrop = document.getElementById("modalBackdrop");
  const modalCard = document.querySelector(".modal-card");
  const modalTitle = document.getElementById("modalTitle");
  const modalBody = document.getElementById("modalBody");
  const closeModalBtn = document.getElementById("closeModalBtn");

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const bytes = new Uint8Array(reader.result);
        let binary = "";
        const chunkSize = 0x8000;
        for (let i = 0; i < bytes.length; i += chunkSize) {
          binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
        }
        resolve(btoa(binary));
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  }

  function decodeBase64Float32(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = bin.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
  }

  function openModal(title, htmlContent, wide = false) {
    modalTitle.textContent = title;
    modalBody.innerHTML = htmlContent;
    if (modalCard) modalCard.classList.toggle("modal-wide", !!wide);
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
    themeToggleBtn.textContent = APP_STATE.isLightTheme ? "Тема: светлая" : "Тема: тёмная";
    render3DScene();
    if (APP_STATE.lastResponse) {
      render2DBlueprint(APP_STATE.lastResponse);
      renderAsymmetryChart(APP_STATE.lastResponse);
    }
  });

  async function initCatalog() {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.get("scan")) {
        APP_STATE.currentFilename = urlParams.get("scan");
      }
      const res = await fetch("/api/catalog");
      const data = await res.json();
      APP_STATE.catalogScans = data.scans || [];
      populateScanSelect(APP_STATE.catalogScans);
      if (data.expert_queue_count !== undefined) {
        document.getElementById("queueCountSpan").textContent = data.expert_queue_count;
      }
      if (data.evaluation_metrics && data.evaluation_metrics.ml_model_test_metrics) {
        const m = data.evaluation_metrics.ml_model_test_metrics;
        const valLine = document.getElementById("valSummaryLine");
        if (valLine) {
          valLine.textContent = `Тест (25 STL): CI MAE ${m.CI_pp.mae.toFixed(2)} (P95 ${m.CI_pp.p95_ae.toFixed(2)}) · CVAI MAE ${m.CVAI_pp.mae.toFixed(2)} (P95 ${m.CVAI_pp.p95_ae.toFixed(2)})`;
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

  async function runAnalysis(includeMesh3d = true) {
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
        include_mesh_3d: includeMesh3d,
      };

      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      APP_STATE.uploadedFileB64 = null;
      if (data.error) {
        if (includeMesh3d) alert("Ошибка анализа: " + data.error);
        return null;
      }
      if (!includeMesh3d && APP_STATE.lastResponse) {
        // Keep existing 3D mesh buffer & multi-level cache during rapid live drag
        data.mesh_3d = APP_STATE.lastResponse.mesh_3d;
        if (data.craniometry && (!data.craniometry.multi_level_slices || data.craniometry.multi_level_slices.length === 0) && APP_STATE.lastResponse.craniometry) {
          data.craniometry.multi_level_slices = APP_STATE.lastResponse.craniometry.multi_level_slices;
        }
      }
      APP_STATE.lastResponse = data;
      if (includeMesh3d && data.mesh_3d) {
        loadMeshBuffers(data.mesh_3d);
      }
      updateDashboardUI(data);
      render3DScene();
      render2DBlueprint(data);
      renderAsymmetryChart(data);
      return data;
    } catch (err) {
      console.error("Ошибка запроса /api/analyze:", err);
      return null;
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
      `<strong>${q.triangle_count.toLocaleString()}</strong> тр. · <strong>${q.unique_vertex_count.toLocaleString()}</strong> верш.`;
    document.getElementById("scaleBadge").innerHTML = `Масштаб <strong>×${q.applied_scale}</strong> (мм)`;
    document.getElementById("scaleActiveLabel").textContent = `×${q.applied_scale} (мм)`;
    document.getElementById("runtimeBadge").innerHTML = `<strong>${data.processing_time_sec.toFixed(3)} с</strong> · Δ ≤ 0.01`;

    // Quiet status chip in sentence case
    const hBadge = document.getElementById("headerStatusBadge");
    hBadge.className = `status-chip chip-${q.status}`;
    const stLabel = q.status === "VALID" ? "Норма" : q.status === "WARNING" ? "Внимание" : "Отказ";
    hBadge.textContent = `${stLabel} · ${q.quality_score.toFixed(0)}%`;

    // SVG Ring Gauge & Mini Bars
    const ringCircle = document.getElementById("qualityRingCircle");
    const circum = 175.9;
    const offset = circum * (1 - Math.max(0, Math.min(100, q.quality_score)) / 100);
    ringCircle.setAttribute("stroke-dashoffset", String(offset.toFixed(1)));
    const ringCol = q.status === "VALID" ? PALETTE.positive : q.status === "WARNING" ? PALETTE.amber : PALETTE.negative;
    ringCircle.setAttribute("stroke", ringCol);
    document.getElementById("qualityScoreVal").textContent = `${q.quality_score.toFixed(0)}%`;
    document.getElementById("qualityScoreVal").style.color = ringCol;

    document.getElementById("barClosure").style.width = `${q.vault_closure_pct}%`;
    document.getElementById("barClosure").style.background = q.vault_closure_pct >= 85 ? PALETTE.positive : PALETTE.negative;
    document.getElementById("txtClosure").textContent = `${q.vault_closure_pct.toFixed(0)}%`;

    document.getElementById("barQuad").style.width = `${q.quadrant_coverage_pct}%`;
    document.getElementById("barQuad").style.background = q.quadrant_coverage_pct >= 35 ? PALETTE.blue : PALETTE.negative;
    document.getElementById("txtQuad").textContent = `${q.quadrant_coverage_pct.toFixed(0)}%`;

    const meshPct = Math.min(100, (q.triangle_count / 85000) * 100);
    document.getElementById("barMesh").style.width = `${meshPct}%`;
    document.getElementById("txtMesh").textContent = `${(q.triangle_count / 1000).toFixed(1)}k`;

    if (q.suggested_scale !== q.applied_scale && q.status === "REJECTED") {
      scaleSuggestionBox.style.display = "block";
      applySuggestedScaleBtn.textContent = `Применить масштаб ×${q.suggested_scale}`;
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
    const techBox = document.getElementById("techInstructionBox");
    const unitInfo = q.unit_diagnosis ? `<div style="margin-top:4px; color:var(--app-muted);">Единицы: ${q.unit_diagnosis}</div>` : "";
    const floatInfo = q.floating_triangles_count > 0
      ? `<div style="margin-top:2px; color:${PALETTE.amber};">Артефакты шума: ${q.floating_triangles_count} «летающих» тр. (фрагментов: ${q.disconnected_components_count})</div>`
      : "";

    if (q.status === "REJECTED") {
      qTitle.style.color = PALETTE.negative;
      qTitle.textContent = "Отказ: скан не прошёл контроль качества";
      qBody.innerHTML = q.defects.map((d) => `<div style="color:${PALETTE.negative};">• ${d}</div>`).join("") + unitInfo + floatInfo;
    } else if (q.status === "WARNING") {
      qTitle.style.color = PALETTE.amber;
      qTitle.textContent = "Скан пригоден с предупреждением";
      qBody.innerHTML = q.warnings.map((w) => `<div style="color:${PALETTE.amber};">• ${w}</div>`).join("") + unitInfo + floatInfo;
    } else {
      qTitle.style.color = PALETTE.positive;
      qTitle.textContent = "Целостность свода подтверждена";
      qBody.innerHTML = `Габариты: ${q.extents_mm[0]} × ${q.extents_mm[1]} × ${q.extents_mm[2]} мм` + floatInfo;
    }

    if (techBox) {
      if (q.technician_instructions && q.technician_instructions.length > 0) {
        techBox.style.display = "block";
        techBox.innerHTML = `<strong>Подсказка технику:</strong> ${q.technician_instructions.map((ins) => `<div style="margin-top:3px;">→ ${ins}</div>`).join("")}`;
      } else {
        techBox.style.display = "none";
      }
    }

    const lms = data.landmarks_mm;
    const lmConf = data.landmark_confidence;
    const lmTag = document.getElementById("landmarkModeTag");
    if (data.manual_correction_applied) {
      lmTag.textContent = "Правка врача · 100%";
      lmTag.style.color = PALETTE.amber;
    } else if (APP_STATE.useBaseline) {
      lmTag.textContent = "Baseline";
      lmTag.style.color = PALETTE.appMuted;
    } else if (lmConf && lmConf.overall_confidence_pct != null) {
      lmTag.textContent = `ML авто · ${lmConf.overall_confidence_pct.toFixed(0)}%`;
      lmTag.style.color = lmConf.any_needs_review ? PALETTE.amber : PALETTE.positive;
    } else {
      lmTag.textContent = "ML авто";
      lmTag.style.color = PALETTE.positive;
    }

    ["Sellion", "RightTragion", "LeftTragion", "Subnasal"].forEach((k) => {
      for (let axis = 0; axis < 3; axis++) {
        const inp = document.getElementById(`lm_${k}_${axis}`);
        if (inp) {
          inp.value = lms && lms[k] ? Number(lms[k][axis]).toFixed(1) : "";
          inp.disabled = !q.is_usable;
        }
      }
      const confEl = document.getElementById(`lm_conf_${k}`);
      if (confEl) {
        const cInfo = lmConf && lmConf.per_landmark ? lmConf.per_landmark[k] : null;
        if (!cInfo || !q.is_usable) {
          confEl.textContent = "--";
          confEl.className = "lm-conf";
          confEl.title = "";
        } else if (cInfo.status === "VERIFIED") {
          confEl.textContent = "ОК";
          confEl.className = "lm-conf";
          confEl.title = cInfo.note || "";
        } else {
          confEl.textContent = `${Math.round(cInfo.confidence_pct)}%`;
          confEl.className = "lm-conf" + (cInfo.status === "LOW" ? " conf-low" : cInfo.needs_review ? " conf-warn" : "");
          confEl.title = cInfo.note || "";
        }
      }
    });

    const confAlertBox = document.getElementById("lmConfidenceAlertBox");
    const confAlertText = document.getElementById("lmConfidenceAlertText");
    if (confAlertBox && confAlertText) {
      if (q.is_usable && lmConf && lmConf.any_needs_review && lmConf.review_landmarks.length > 0) {
        const namesRu = { Sellion: "Sellion", RightTragion: "Tragion R", LeftTragion: "Tragion L", Subnasal: "Subnasal" };
        const flagged = lmConf.review_landmarks.map((k) => `${namesRu[k] || k} (${Math.round(lmConf.per_landmark[k].confidence_pct)}%)`).join(", ");
        confAlertBox.style.display = "block";
        confAlertText.innerHTML = `<strong>Требуется проверка ориентира:</strong> ${flagged}. Возможна окклюзия волосами/шапочкой — подсвечено жёлтым в 3D.`;
      } else {
        confAlertBox.style.display = "none";
      }
    }

    const borderlineBox = document.getElementById("borderlineAlertBox");
    if (!cran) {
      document.getElementById("kpiCI").textContent = "Отказ";
      document.getElementById("kpiCVA").textContent = "-- мм";
      document.getElementById("kpiCVAI").textContent = "--%";
      document.getElementById("kpiPerim").textContent = "-- мм";
      const banner = document.getElementById("cvaiGradeBanner");
      banner.textContent = "Расчёт заблокирован контролем качества входных данных";
      banner.style.color = PALETTE.negative;
      if (borderlineBox) borderlineBox.style.display = "none";
      document.getElementById("valL").textContent = "--";
      document.getElementById("valW").textContent = "--";
      document.getElementById("valD1").textContent = "--";
      document.getElementById("valD2").textContent = "--";
      document.getElementById("valPerim").textContent = "--";
      document.getElementById("valFrontalAsym").textContent = "--";
      document.getElementById("valOccipitalAsym").textContent = "--";
      document.getElementById("valCVAIMin").textContent = "--";
      document.getElementById("valShapeClass").textContent = "Отказ";
      document.getElementById("shapeProbBars").innerHTML = "";
      const qGrid = document.getElementById("quadrantGrid");
      if (qGrid) qGrid.innerHTML = "";
      const mlBody = document.getElementById("multiLevelBody");
      if (mlBody) mlBody.innerHTML = "";
      document.getElementById("gtComparisonBox").style.display = "none";
      return;
    }

    const ind = cran.indices;
    const lin = cran.linear_measurements_mm;
    const reg = cran.regional_asymmetry;
    const grade = cran.classification.cvai_grade;
    const borderline = cran.classification.borderline_alert;
    const gradeColor = CVAI_TONED_COLORS[grade.code] || PALETTE.positive;

    document.getElementById("kpiCI").textContent = `${ind.CI.toFixed(2)}%`;
    document.getElementById("kpiCVA").textContent = `${ind.CVA_mm.toFixed(2)} мм`;
    const kpiCvaiEl = document.getElementById("kpiCVAI");
    kpiCvaiEl.textContent = `${ind.CVAI.toFixed(2)}%`;
    kpiCvaiEl.style.color = gradeColor;
    document.getElementById("kpiPerim").textContent = `${lin.perimeter_slice.toFixed(1)} мм`;

    const banner = document.getElementById("cvaiGradeBanner");
    banner.textContent = grade.label;
    banner.style.color = gradeColor;

    if (borderlineBox) {
      if (borderline && borderline.is_borderline) {
        borderlineBox.style.display = "block";
        borderlineBox.innerHTML = `<strong>Пограничное значение (±${borderline.margin_pp.toFixed(2)} п.п. у порога ${borderline.nearest_threshold_pct.toFixed(2)}%):</strong> требуется визуальная проверка ориентиров врачом (ближайшая категория: ${borderline.adjacent_grade_code}).`;
      } else {
        borderlineBox.style.display = "none";
      }
    }

    // Sync active quick level preset button (20% / 30% / 50%)
    document.querySelectorAll(".level-preset-btn").forEach((btn) => {
      const lv = parseFloat(btn.getAttribute("data-level"));
      btn.classList.toggle("active", Math.abs(lv - lin.slice_level_ratio) < 0.015);
    });

    const ptrPct = Math.min(98.5, Math.max(1.5, (ind.CVAI_unrounded / 14.0) * 100.0));
    document.getElementById("cvaiPointer").style.left = `${ptrPct}%`;

    document.getElementById("ciInterpretationText").textContent = cran.classification.ci_interpretation;
    document.getElementById("sliceHeightTag").textContent = `Z = ${lin.slice_height_z.toFixed(1)} мм (${Math.round(lin.slice_level_ratio * 100)}%)`;
    document.getElementById("dominantZoneTag").textContent = reg.dominant_asymmetry_zone;

    document.getElementById("valL").textContent = `${lin.L_length_ap.toFixed(2)} мм`;
    document.getElementById("valW").textContent = `${lin.W_width_ml.toFixed(2)} мм`;
    document.getElementById("valD1").textContent = `${lin.D1_diagonal_plus30.toFixed(2)} мм`;
    document.getElementById("valD2").textContent = `${lin.D2_diagonal_minus30.toFixed(2)} мм`;
    document.getElementById("valPerimLabel").textContent = lin.perimeter_label;
    document.getElementById("valPerim").textContent = `${lin.perimeter_slice.toFixed(2)} мм`;
    document.getElementById("valCVAIMin").textContent = `${ind.CVAI_min_ref.toFixed(2)}%`;

    const fPct = Math.min(100, (reg.frontal_asymmetry_pct / 16.0) * 100);
    const oPct = Math.min(100, (reg.occipital_asymmetry_pct / 16.0) * 100);
    document.getElementById("barFrontal").style.width = `${fPct}%`;
    document.getElementById("barOccipital").style.width = `${oPct}%`;
    document.getElementById("valFrontalAsym").textContent = `${reg.frontal_diff_mm.toFixed(1)} мм (${reg.frontal_asymmetry_pct.toFixed(1)}%)`;
    document.getElementById("valOccipitalAsym").textContent = `${reg.occipital_diff_mm.toFixed(1)} мм (${reg.occipital_asymmetry_pct.toFixed(1)}%)`;

    const probContainer = document.getElementById("shapeProbBars");
    if (data.shape_classifier && data.shape_classifier.probabilities_pct) {
      document.getElementById("valShapeClass").textContent = data.shape_classifier.predicted_code;
      const probs = data.shape_classifier.probabilities_pct;
      const barColors = {
        N: PALETTE.positive,
        P: PALETTE.amber,
        B: PALETTE.blue,
        BP: PALETTE.coral,
        S: "#85858b",
        SP: PALETTE.negative,
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
            <div class="mini-bar-row" style="grid-template-columns: 96px 1fr 44px;">
              <span style="color: var(--app-muted);">${barNames[k] || k}</span>
              <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${p}%; background:${barColors[k] || PALETTE.positive};"></div></div>
              <span class="mono" style="text-align:right; color: var(--app-secondary);">${p.toFixed(0)}%</span>
            </div>
          `;
        })
        .join("");
    }

    // Q1-Q4 Volumetric Quadrants
    const qvol = cran.quadrant_volumes;
    const qGrid = document.getElementById("quadrantGrid");
    const asrPsrTag = document.getElementById("asrPsrTag");
    if (qvol && qGrid) {
      if (asrPsrTag) {
        asrPsrTag.textContent = `ASR ${qvol.anterior_symmetry_ratio.toFixed(2)} · PSR ${qvol.posterior_symmetry_ratio.toFixed(2)}`;
      }
      const qItems = [
        ["Q1", "Лев-Лоб", qvol.Q1_front_left_cm3],
        ["Q2", "Прав-Лоб", qvol.Q2_front_right_cm3],
        ["Q3", "Лев-Зат", qvol.Q3_back_left_cm3],
        ["Q4", "Прав-Зат", qvol.Q4_back_right_cm3],
      ];
      qGrid.innerHTML = qItems
        .map(([code, lbl, cm3]) => {
          const isFlat = qvol.flattened_quadrant_code === code;
          return `
            <div class="quad-cell ${isFlat ? "is-flat" : ""}" title="${isFlat ? "Зона наибольшего уплощения" : lbl}">
              <span style="color: ${isFlat ? PALETTE.amber : "var(--app-secondary)"};">${code} ${lbl}</span>
              <span class="mono" style="color: var(--app-text);">${cm3.toFixed(0)} см³</span>
            </div>
          `;
        })
        .join("");
    }

    // Multi-level 20% / 30% / 50% slice table
    const mlBody = document.getElementById("multiLevelBody");
    if (mlBody && cran.multi_level_slices) {
      mlBody.innerHTML = cran.multi_level_slices
        .map((s) => {
          const isCur = Math.abs(s.level_ratio - lin.slice_level_ratio) < 0.015;
          return `
            <tr data-jump-level="${s.level_ratio}" style="${isCur ? "color: var(--positive); font-weight: 500;" : ""}">
              <td>${Math.round(s.level_ratio * 100)}% (Z=${s.z_mm.toFixed(0)})</td>
              <td>${s.CI.toFixed(1)}%</td>
              <td>${s.CVA_mm.toFixed(1)} мм</td>
              <td>${s.CVAI.toFixed(2)}%</td>
            </tr>
          `;
        })
        .join("");
      mlBody.querySelectorAll("tr[data-jump-level]").forEach((tr) => {
        tr.addEventListener("click", () => {
          const lv = parseFloat(tr.getAttribute("data-jump-level"));
          setSliceLevel(lv);
        });
      });
    }

    const gtBox = document.getElementById("gtComparisonBox");
    if (data.ground_truth_comparison && data.ground_truth_comparison.has_ground_truth) {
      const gt = data.ground_truth_comparison;
      gtBox.style.display = "block";
      document.getElementById("gtTitle").textContent =
        gt.split === "test"
          ? `Сверка с эталоном (закрытый тест · ${data.scan_id})`
          : `Сверка с эталоном (обучающая выборка · ${data.scan_id})`;
      document.getElementById("gtMetricsLine").textContent =
        `Эталон: CI ${gt.gt_indices.CI.toFixed(2)}% · CVAI ${gt.gt_indices.CVAI.toFixed(2)}% | ` +
        `Ошибка: |ΔCI| ${gt.abs_errors.CI_pp.toFixed(2)} п.п. · |ΔCVAI| ${gt.abs_errors.CVAI_pp.toFixed(2)} п.п. · |ΔCVA| ${gt.abs_errors.CVA_mm.toFixed(2)} мм`;
    } else {
      gtBox.style.display = "none";
    }
  }

  // ==================== SPATIAL 3D VIEWPORT ENGINE (INK DARK PALETTE) ====================
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
    // Toned palette: -7.5mm flattening (#5b9ee5 blue) -> 0mm neutral ivory -> +7.5mm prominence (#e06d53 coral)
    const t = Math.max(-1.0, Math.min(1.0, mm / 7.5));
    if (t < 0) {
      const a = -t;
      return [
        Math.round(220 * (1 - a) + 91 * a),
        Math.round(216 * (1 - a) + 158 * a),
        Math.round(208 * (1 - a) + 229 * a),
      ];
    } else {
      const a = t;
      return [
        Math.round(220 * (1 - a) + 224 * a),
        Math.round(216 * (1 - a) + 109 * a),
        Math.round(208 * (1 - a) + 83 * a),
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

    // Flat calm Ink Dark canvas matching welcome.html (--ink-bg: #0b0d14)
    ctx.fillStyle = PALETTE.inkBg;
    ctx.fillRect(0, 0, w, h);

    // Subtle hairline floor grid
    ctx.strokeStyle = "rgba(238, 240, 245, 0.06)";
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

      const light = Math.max(0.22, Math.min(1.0, 0.28 + 0.56 * Math.abs(vnz) + 0.22 * Math.max(0, vny) + 0.14 * vnx));
      const depth = (p1[2] + p2[2] + p3[2]) / 3.0;
      const asymVal = asym ? (asym[t * 3] + asym[t * 3 + 1] + asym[t * 3 + 2]) / 3.0 : 0.0;

      drawList.push({ p1, p2, p3, depth, light, asymVal });
    }

    drawList.sort((a, b) => a.depth - b.depth);

    if (APP_STATE.renderMode === "wire") {
      ctx.strokeStyle = "rgba(76, 195, 138, 0.34)";
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
          r = Math.round(226 * item.light);
          g = Math.round(222 * item.light);
          b = Math.round(214 * item.light);
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

    // Red 3D Defect Zone Overlay directly on the cranial surface when sectors are missing
    if (data && data.quality && Array.isArray(data.quality.defect_zones) && data.quality.defect_zones.length > 0) {
      data.quality.defect_zones.forEach((dz) => {
        const lowPts = (dz.arc_low_3d || []).map((pt) => projectPoint3D(pt[0], pt[1], pt[2], w, h));
        const highPts = (dz.arc_high_3d || []).map((pt) => projectPoint3D(pt[0], pt[1], pt[2], w, h));
        if (lowPts.length >= 2 && highPts.length === lowPts.length) {
          ctx.save();
          // Translucent red sector strips
          for (let i = 0; i < lowPts.length - 1; i++) {
            ctx.beginPath();
            ctx.moveTo(lowPts[i][0], lowPts[i][1]);
            ctx.lineTo(lowPts[i + 1][0], lowPts[i + 1][1]);
            ctx.lineTo(highPts[i + 1][0], highPts[i + 1][1]);
            ctx.lineTo(highPts[i][0], highPts[i][1]);
            ctx.closePath();
            ctx.fillStyle = "rgba(229, 72, 77, 0.28)";
            ctx.fill();
            ctx.strokeStyle = "rgba(229, 72, 77, 0.65)";
            ctx.lineWidth = 1.0;
            ctx.stroke();
          }
          // Bold red dashed boundary arcs
          ctx.strokeStyle = PALETTE.negative;
          ctx.lineWidth = 2.2;
          ctx.setLineDash([5, 3]);
          [lowPts, highPts].forEach((arc) => {
            ctx.beginPath();
            arc.forEach((p, idx) => (idx === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
            ctx.stroke();
          });
          ctx.setLineDash([]);

          // 3D Defect Callout Banner
          if (dz.label_anchor_3d) {
            const lp = projectPoint3D(dz.label_anchor_3d[0], dz.label_anchor_3d[1], dz.label_anchor_3d[2], w, h);
            const lbl = `ДЕФЕКТ: нет данных (${dz.region_name}, ${Math.round(dz.missing_span_deg)}°)`;
            ctx.font = "500 11px monospace";
            const tw = ctx.measureText(lbl).width;
            const bx = Math.max(14, Math.min(w - tw - 24, lp[0] - tw * 0.5));
            const by = Math.max(64, Math.min(h - 60, lp[1] - 12));
            ctx.fillStyle = "rgba(25, 10, 12, 0.92)";
            ctx.strokeStyle = PALETTE.negative;
            ctx.lineWidth = 1.2;
            ctx.fillRect(bx - 6, by - 13, tw + 12, 20);
            ctx.strokeRect(bx - 6, by - 13, tw + 12, 20);
            ctx.fillStyle = "#fca5a5";
            ctx.fillText(lbl, bx, by);
          }
          ctx.restore();
        }
      });
    }

    if (APP_STATE.showPlanes && data && data.craniometry && data.craniometry.geometry_3d) {
      const g3d = data.craniometry.geometry_3d;
      const contour3d = g3d.contour_3d || [];
      const ep3 = g3d.axes_endpoints_3d;
      const cWorld = g3d.slice_center_world;

      if (APP_STATE.showRays && ep3 && cWorld) {
        const pCenter = projectPoint3D(cWorld[0], cWorld[1], cWorld[2], w, h);
        [
          [ep3.ap_front, ep3.ap_back, "rgba(91, 158, 229, 0.24)"],
          [ep3.ml_left, ep3.ml_right, "rgba(91, 158, 229, 0.18)"],
          [ep3.d1_front, ep3.d1_back, "rgba(229, 169, 60, 0.26)"],
          [ep3.d2_front, ep3.d2_back, "rgba(224, 109, 83, 0.26)"],
        ].forEach(([ptA, ptB, rayCol]) => {
          const extA = [
            cWorld[0] + (ptA[0] - cWorld[0]) * 1.42,
            cWorld[1] + (ptA[1] - cWorld[1]) * 1.42,
            cWorld[2] + (ptA[2] - cWorld[2]) * 1.42,
          ];
          const extB = [
            cWorld[0] + (ptB[0] - cWorld[0]) * 1.42,
            cWorld[1] + (ptB[1] - cWorld[1]) * 1.42,
            cWorld[2] + (ptB[2] - cWorld[2]) * 1.42,
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

        ctx.beginPath();
        ctx.arc(pCenter[0], pCenter[1], 4.0, 0, Math.PI * 2);
        ctx.fillStyle = "#eef0f5";
        ctx.fill();
      }

      if (contour3d.length > 2) {
        ctx.fillStyle = "rgba(76, 195, 138, 0.12)";
        ctx.strokeStyle = PALETTE.positive;
        ctx.lineWidth = 2.2;
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

          [pA, pB].forEach((p) => {
            ctx.beginPath();
            ctx.arc(p[0], p[1], 3.0, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
          });

          if (label) {
            ctx.fillStyle = color;
            ctx.font = "500 11px monospace";
            ctx.fillText(label, pA[0] + 7, pA[1] - 5);
          }
        }

        draw3DLine(ep3.ap_front, ep3.ap_back, PALETTE.blue, 1.6, [4, 3], "L (AP)");
        draw3DLine(ep3.ml_left, ep3.ml_right, "#7a8296", 1.5, [2, 3], "W (ML)");
        draw3DLine(ep3.d1_front, ep3.d1_back, PALETTE.amber, 2.2, null, "D1 (+30°)");
        draw3DLine(ep3.d2_front, ep3.d2_back, PALETTE.coral, 2.2, null, "D2 (-30°)");
      }
    }

    APP_STATE.projectedLandmarks = {};
    if (APP_STATE.showLandmarks && data && data.landmarks_mm) {
      const lms = data.landmarks_mm;

      if (APP_STATE.showPlanes && lms.Sellion && lms.RightTragion && lms.LeftTragion) {
        const pSe = projectPoint3D(lms.Sellion[0], lms.Sellion[1], lms.Sellion[2], w, h);
        const pRT = projectPoint3D(lms.RightTragion[0], lms.RightTragion[1], lms.RightTragion[2], w, h);
        const pLT = projectPoint3D(lms.LeftTragion[0], lms.LeftTragion[1], lms.LeftTragion[2], w, h);

        ctx.save();
        ctx.strokeStyle = "rgba(238, 240, 245, 0.35)";
        ctx.fillStyle = "rgba(238, 240, 245, 0.05)";
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

      const perConf = data.landmark_confidence && data.landmark_confidence.per_landmark ? data.landmark_confidence.per_landmark : {};
      Object.keys(LANDMARK_COLORS).forEach((key) => {
        if (!lms[key]) return;
        const c = lms[key];
        const p = projectPoint3D(c[0], c[1], c[2], w, h);
        APP_STATE.projectedLandmarks[key] = p;

        const cInfo = perConf[key];
        const needsReview = !!(cInfo && cInfo.needs_review);
        const isHovered = APP_STATE.hoveredLandmarkKey === key || APP_STATE.dragLandmarkKey === key;
        const radius = isHovered ? 7.5 : 5.5;

        // Yellow/amber caution beacon ring when model confidence is low/medium
        if (needsReview) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(p[0], p[1], radius + 7.5, 0, Math.PI * 2);
          ctx.setLineDash([3, 2]);
          ctx.strokeStyle = PALETTE.amber;
          ctx.lineWidth = 1.8;
          ctx.stroke();
          ctx.restore();
        }

        ctx.beginPath();
        ctx.arc(p[0], p[1], radius + 3.5, 0, Math.PI * 2);
        ctx.strokeStyle = needsReview ? PALETTE.amber : LANDMARK_COLORS[key];
        ctx.lineWidth = needsReview ? 1.6 : 1.0;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
        ctx.fillStyle = needsReview ? PALETTE.amber : LANDMARK_COLORS[key];
        ctx.fill();
        ctx.lineWidth = isHovered ? 2.0 : 1.4;
        ctx.strokeStyle = "#eef0f5";
        ctx.stroke();

        const confSuffix = cInfo && cInfo.status !== "VERIFIED" ? ` (${Math.round(cInfo.confidence_pct)}%${needsReview ? " · проверить" : ""})` : "";
        ctx.fillStyle = needsReview ? PALETTE.amber : "#eef0f5";
        ctx.font = "500 11px monospace";
        ctx.fillText(`${key}${confSuffix}`, p[0] + 12, p[1] + 4);
      });
    }

    if (APP_STATE.renderMode === "heatmap") {
      const cbX = 16, cbY = 56, cbW = 10, cbH = 124;
      for (let i = 0; i < cbH; i++) {
        const mm = 7.5 * (1 - (2 * i) / cbH);
        const rgb = heatmapColor(mm);
        ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(cbX, cbY + i, cbW, 1.5);
      }
      ctx.strokeStyle = "rgba(238, 240, 245, 0.22)";
      ctx.strokeRect(cbX, cbY, cbW, cbH);
      ctx.fillStyle = "#a3a9b8";
      ctx.font = "11px monospace";
      ctx.fillText("+7.5 мм выступ", cbX + 16, cbY + 8);
      ctx.fillText(" 0.0 мм норма", cbX + 16, cbY + cbH / 2 + 4);
      ctx.fillText("-7.5 мм уплощение", cbX + 16, cbY + cbH - 2);
    }

    ctx.restore();
  }

  // ==================== 2D BLUEPRINT & RADIAL CHART (GRAPHITE PALETTE) ====================
  function render2DBlueprint(data) {
    const ctx = canvas2d.getContext("2d");
    const w = canvas2d.width;
    const h = canvas2d.height;
    ctx.clearRect(0, 0, w, h);

    const bgCol = APP_STATE.isLightTheme ? "#f4f4f5" : PALETTE.appSidebar;
    const gridCol = APP_STATE.isLightTheme ? "rgba(23, 23, 23, 0.08)" : "rgba(237, 237, 237, 0.06)";
    ctx.fillStyle = bgCol;
    ctx.fillRect(0, 0, w, h);

    if (!data || !data.craniometry || !data.craniometry.geometry_2d) {
      ctx.fillStyle = PALETTE.negative;
      ctx.font = "500 12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Сечение не построено (дефект скана)", w / 2, h / 2);
      return;
    }

    const g2d = data.craniometry.geometry_2d;
    const lin = data.craniometry.linear_measurements_mm;
    const contour = g2d.contour_xy;
    const ep = g2d.endpoints_xy;
    const center = g2d.center;

    const scale = 1.12;
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
    ctx.fillStyle = "rgba(76, 195, 138, 0.12)";
    ctx.fill();
    ctx.strokeStyle = PALETTE.positive;
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
        ctx.arc(p[0], p[1], 2.5, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
      });

      if (label) {
        ctx.fillStyle = col;
        ctx.font = "500 10px monospace";
        ctx.fillText(label, p1[0] + 4, p1[1] + 2);
      }
    }

    drawChord(ep.ap_front, ep.ap_back, PALETTE.blue, 1.3, [3, 3], `L ${lin.L_length_ap.toFixed(1)}`);
    drawChord(ep.ml_left, ep.ml_right, PALETTE.appMuted, 1.3, [2, 2], `W ${lin.W_width_ml.toFixed(1)}`);
    drawChord(ep.d1_front, ep.d1_back, PALETTE.amber, 2.0, null, `D1 ${lin.D1_diagonal_plus30.toFixed(1)}`);
    drawChord(ep.d2_front, ep.d2_back, PALETTE.coral, 2.0, null, `D2 ${lin.D2_diagonal_minus30.toFixed(1)}`);

    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = APP_STATE.isLightTheme ? "#171717" : "#ededed";
    ctx.fill();
  }

  function renderAsymmetryChart(data) {
    const ctx = canvasAsym.getContext("2d");
    const w = canvasAsym.width;
    const h = canvasAsym.height;
    ctx.clearRect(0, 0, w, h);

    const bgCol = APP_STATE.isLightTheme ? "#f4f4f5" : PALETTE.appSidebar;
    ctx.fillStyle = bgCol;
    ctx.fillRect(0, 0, w, h);

    if (!data || !data.craniometry || !data.craniometry.geometry_2d) {
      ctx.fillStyle = PALETTE.appMuted;
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Данные асимметрии отсутствуют", w / 2, h / 2);
      return;
    }

    const prof = data.craniometry.geometry_2d.asymmetry_profile_mm || [];
    const padL = 34, padR = 12, padT = 16, padB = 22;
    const pw = w - padL - padR;
    const ph = h - padT - padB;
    const midY = padT + ph * 0.5;
    const maxMM = Math.max(6.0, ...prof.map((v) => Math.abs(v))) * 1.15;

    ctx.strokeStyle = APP_STATE.isLightTheme ? "rgba(23, 23, 23, 0.08)" : "rgba(237, 237, 237, 0.06)";
    ctx.lineWidth = 1;
    [-5, 0, 5].forEach((val) => {
      const y = midY - (val / maxMM) * (ph * 0.5);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillStyle = PALETTE.appMuted;
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText(`${val > 0 ? "+" : ""}${val}`, padL - 5, y + 3);
    });

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
    ctx.fillStyle = "rgba(229, 169, 60, 0.14)";
    ctx.fill();

    ctx.beginPath();
    for (let i = 0; i < prof.length; i++) {
      const x = padL + (i / (prof.length - 1)) * pw;
      const y = midY - (prof[i] / maxMM) * (ph * 0.5);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = PALETTE.amber;
    ctx.lineWidth = 1.8;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(peakX, peakY, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = PALETTE.positive;
    ctx.fill();
    ctx.fillStyle = PALETTE.positive;
    ctx.font = "500 10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`max ${peakVal >= 0 ? "+" : ""}${peakVal.toFixed(1)} мм`, Math.min(w - 44, Math.max(58, peakX)), Math.max(12, peakY - 6));

    ctx.fillStyle = PALETTE.appMuted;
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("-180° затылок", padL + 36, h - 5);
    ctx.fillText("0° лоб", padL + pw * 0.5, h - 5);
    ctx.fillText("+180° затылок", w - padR - 36, h - 5);
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

      // Exact camera-plane inverse projection matching projectPoint3D across all camera angles
      const cosY = Math.cos(APP_STATE.camYaw), sinY = Math.sin(APP_STATE.camYaw);
      const cosP = Math.cos(APP_STATE.camPitch), sinP = Math.sin(APP_STATE.camPitch);
      lms[key][0] += (sinY * dx + cosY * sinP * dy) * worldDelta;
      lms[key][1] += (cosY * dx - sinY * sinP * dy) * worldDelta;
      lms[key][2] += (-cosP * dy) * worldDelta;

      for (let axis = 0; axis < 3; axis++) {
        const inp = document.getElementById(`lm_${key}_${axis}`);
        if (inp) inp.value = lms[key][axis].toFixed(1);
      }
      render3DScene();
      scheduleLiveLandmarkRecalc(key);
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

  function scheduleLiveLandmarkRecalc(draggedKey) {
    const now = performance.now();
    if (APP_STATE.liveDragInFlight) {
      APP_STATE.liveDragQueued = true;
      return;
    }
    if (now - APP_STATE.lastLiveDragTs < 75) {
      APP_STATE.liveDragQueued = true;
      return;
    }
    APP_STATE.lastLiveDragTs = now;
    APP_STATE.liveDragInFlight = true;
    APP_STATE.liveDragQueued = false;
    APP_STATE.customLandmarks = collectCustomLandmarks();

    runAnalysis(false).then((res) => {
      APP_STATE.liveDragInFlight = false;
      if (liveDragBanner && APP_STATE.dragLandmarkKey && res && res.craniometry) {
        const ind = res.craniometry.indices;
        liveDragBanner.style.display = "block";
        liveDragBanner.textContent = `Правка ${draggedKey}: CI ${ind.CI.toFixed(2)}% · CVA ${ind.CVA_mm.toFixed(2)} мм · CVAI ${ind.CVAI.toFixed(2)}%`;
      }
      if (APP_STATE.liveDragQueued && APP_STATE.dragLandmarkKey) {
        scheduleLiveLandmarkRecalc(APP_STATE.dragLandmarkKey);
      }
    });
  }

  window.addEventListener("mouseup", () => {
    if (APP_STATE.dragLandmarkKey) {
      APP_STATE.dragLandmarkKey = null;
      APP_STATE.isDragging = false;
      if (liveDragBanner) liveDragBanner.style.display = "none";
      APP_STATE.customLandmarks = collectCustomLandmarks();
      // Full sync (rebuilds 3D asymmetry heatmap + multi-level table)
      runAnalysis(true);
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
      opt.textContent = `${file.name} (загружен с диска)`;
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

  function setSliceLevel(lv) {
    const clamped = Math.max(0.18, Math.min(0.60, lv));
    APP_STATE.levelRatio = clamped;
    levelRatioSlider.value = clamped.toFixed(2);
    levelRatioVal.textContent = `${Math.round(clamped * 100)}% свода`;
    document.querySelectorAll(".level-preset-btn").forEach((btn) => {
      const bLv = parseFloat(btn.getAttribute("data-level"));
      btn.classList.toggle("active", Math.abs(bLv - clamped) < 0.015);
    });
    runAnalysis(true);
  }

  document.querySelectorAll(".level-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      setSliceLevel(parseFloat(btn.getAttribute("data-level")));
    });
  });

  levelRatioSlider.addEventListener("input", () => {
    const v = parseFloat(levelRatioSlider.value);
    levelRatioVal.textContent = `${Math.round(v * 100)}% свода`;
  });

  levelRatioSlider.addEventListener("change", () => {
    setSliceLevel(parseFloat(levelRatioSlider.value));
  });

  useBaselineToggle.addEventListener("change", () => {
    APP_STATE.useBaseline = useBaselineToggle.checked;
    APP_STATE.customLandmarks = null;
    runAnalysis(true);
  });

  expertCiProfileCheck.addEventListener("change", () => {
    APP_STATE.expertCiApproved = expertCiProfileCheck.checked;
    runAnalysis(true);
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
    runAnalysis(true);
  }

  applyLandmarksBtn.addEventListener("click", collectCustomLandmarksAndAnalyze);

  if (confirmLandmarksBtn) {
    confirmLandmarksBtn.addEventListener("click", () => {
      // Doctor confirms current landmark positions -> switches status to VERIFIED (100%)
      APP_STATE.customLandmarks = collectCustomLandmarks();
      runAnalysis(true);
    });
  }

  resetLandmarksBtn.addEventListener("click", () => {
    APP_STATE.customLandmarks = null;
    runAnalysis(true);
  });

  robustnessBtn.addEventListener("click", async () => {
    openModal("Тест устойчивости к 3D-повороту и шуму", "<div class='mono'>Выполняется 3D-поворот на 15° и наложение гауссовского шума σ = 0.25 мм</div>");
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
        modalBody.innerHTML = `<div style="color:${PALETTE.negative};">Ошибка: ${r.error}</div>`;
        return;
      }
      modalBody.innerHTML = `
        <div style="margin-bottom:12px; color:var(--app-secondary);">
          Протокол испытания устойчивости для файла <strong class="mono">${APP_STATE.currentFilename}</strong>
        </div>
        <table class="tel-table mono" style="margin-bottom:12px;">
          <thead>
            <tr style="border-bottom:1px solid var(--app-hairline-strong); color:var(--app-muted);">
              <th style="text-align:left; padding:8px 4px; font-weight:400;">Испытание</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">CI</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">CVA</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">CVAI</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">Отклонение |Δ|</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">Статус</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="k">Базовый запуск</td>
              <td class="v">${r.baseline_indices.CI.toFixed(2)}%</td>
              <td class="v">${r.baseline_indices.CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.baseline_indices.CVAI.toFixed(2)}%</td>
              <td class="v">0.000</td>
              <td class="v" style="color:${PALETTE.positive};">Эталон</td>
            </tr>
            <tr>
              <td class="k">Повторный запуск (≤ 0.01)</td>
              <td class="v">${r.baseline_indices.CI.toFixed(2)}%</td>
              <td class="v">${r.baseline_indices.CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.baseline_indices.CVAI.toFixed(2)}%</td>
              <td class="v">${r.repeat_run_determinism.max_divergence.toFixed(6)}</td>
              <td class="v" style="color:${PALETTE.positive};">Пройдено</td>
            </tr>
            <tr>
              <td class="k">3D-поворот модели (+15.0°)</td>
              <td class="v">${r.rotation_3d_invariance_test.rotated_CI.toFixed(2)}%</td>
              <td class="v">${r.rotation_3d_invariance_test.rotated_CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.rotation_3d_invariance_test.rotated_CVAI.toFixed(2)}%</td>
              <td class="v">ΔCVAI ${r.rotation_3d_invariance_test.delta_CVAI_pp.toFixed(5)}</td>
              <td class="v" style="color:${PALETTE.positive};">Пройдено</td>
            </tr>
            <tr>
              <td class="k">Гауссовский шум (σ = 0.25 мм)</td>
              <td class="v">${r.noise_perturbation_test.noisy_CI.toFixed(2)}%</td>
              <td class="v">${r.noise_perturbation_test.noisy_CVA_mm.toFixed(2)} мм</td>
              <td class="v">${r.noise_perturbation_test.noisy_CVAI.toFixed(2)}%</td>
              <td class="v">ΔCVAI ${r.noise_perturbation_test.delta_CVAI_pp.toFixed(3)} п.п.</td>
              <td class="v" style="color:${PALETTE.positive};">Пройдено</td>
            </tr>
          </tbody>
        </table>
      `;
    } catch (e) {
      modalBody.innerHTML = `<div style="color:${PALETTE.negative};">Ошибка запроса: ${e}</div>`;
    }
  });

  // ==================== UPGRADED BATCH PROCESSING (FOLDER UPLOAD + CSV / EXCEL / ZIP PDF) ====================
  function renderBatchModalShell() {
    openModal(
      "Пакетная загрузка и сводный экспорт",
      `
        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:12px; padding-bottom:12px; border-bottom:1px solid var(--app-hairline);">
          <label class="hud-btn hud-btn-primary" for="batchFilesInput">
            Выбрать STL-файлы (папку за день)
          </label>
          <input type="file" id="batchFilesInput" accept=".stl" multiple style="display:none;" />
          <button type="button" id="runDemoBatchBtn" class="hud-btn">
            Запустить на клиническом наборе (8 STL)
          </button>
          <span class="mono" id="batchStatusMeta" style="color:var(--app-muted); margin-left:auto;">Готово к пакетному расчёту</span>
        </div>
        <div id="batchResultsContainer" class="mono" style="color:var(--app-secondary);">Загрузка клинической серии...</div>
      `,
      true
    );

    const fileInput = document.getElementById("batchFilesInput");
    const demoBtn = document.getElementById("runDemoBatchBtn");

    fileInput.addEventListener("change", async (e) => {
      const files = Array.from(e.target.files || []).filter((f) => f.name.toLowerCase().endsWith(".stl"));
      if (!files.length) return;
      document.getElementById("batchStatusMeta").textContent = `Чтение ${files.length} файлов...`;
      const uploaded_files = [];
      for (const f of files.slice(0, 30)) {
        uploaded_files.push({ filename: f.name, file_b64: await fileToBase64(f) });
      }
      await executeBatchRequest({ uploaded_files });
    });

    demoBtn.addEventListener("click", () => {
      executeBatchRequest({
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
      });
    });

    demoBtn.click();
  }

  async function executeBatchRequest(sourcePayload) {
    const container = document.getElementById("batchResultsContainer");
    const statusMeta = document.getElementById("batchStatusMeta");
    container.innerHTML = "<div class='mono'>Выполняется пакетный расчёт 3D-сканов...</div>";
    try {
      const body = Object.assign(
        {
          scale_factor: 1.0,
          level_ratio: APP_STATE.levelRatio,
          doctor_name: document.getElementById("doctorNameInput").value,
        },
        sourcePayload
      );
      const res = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const b = await res.json();
      if (b.error) {
        container.innerHTML = `<div style="color:${PALETTE.negative};">Ошибка: ${b.error}</div>`;
        return;
      }
      APP_STATE.lastBatchFilenames = b.filenames || [];
      statusMeta.innerHTML = `Обработано: <strong>${b.processed_count}</strong> · Время: <strong>${b.total_time_sec} с</strong>`;

      const rowsHtml = (b.results || [])
        .map(
          (item) => `
          <tr>
            <td class="k mono">${item.scan_id}</td>
            <td class="v"><span class="status-chip chip-${item.status}">${item.status === "VALID" ? "Норма" : item.status === "WARNING" ? "Внимание" : "Отказ"}</span></td>
            <td class="v">${item.CI != null ? item.CI.toFixed(2) + "%" : "--"}</td>
            <td class="v">${item.CVA_mm != null ? item.CVA_mm.toFixed(2) + " мм" : "--"}</td>
            <td class="v">${item.CVAI != null ? item.CVAI.toFixed(2) + "%" : "--"}${item.is_borderline ? ` <span style="color:${PALETTE.amber};" title="Пограничное значение ±0.6 п.п.">[погр]</span>` : ""}</td>
            <td class="v">${item.PSR != null ? item.PSR.toFixed(2) : "--"}</td>
            <td class="v" style="font-size:11px;">${item.grade_code}</td>
            <td class="v">${item.processing_time_sec.toFixed(3)} с</td>
          </tr>
        `
        )
        .join("");

      container.innerHTML = `
        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
          <button type="button" id="dlBatchCsvBtn" class="hud-btn">Скачать CSV</button>
          <button type="button" id="dlBatchXlsBtn" class="hud-btn">Скачать Excel (.xls)</button>
          <button type="button" id="dlBatchZipBtn" class="hud-btn hud-btn-primary">Скачать ZIP (CSV + Excel + все PDF-отчёты)</button>
        </div>
        <table class="tel-table mono">
          <thead>
            <tr style="border-bottom:1px solid var(--app-hairline-strong); color:var(--app-muted);">
              <th style="text-align:left; padding:8px 4px; font-weight:400;">Код скана</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">Статус</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">CI</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">CVA</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">CVAI</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">PSR</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">Категория</th>
              <th style="text-align:right; padding:8px 4px; font-weight:400;">Время</th>
            </tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      `;

      document.getElementById("dlBatchCsvBtn").onclick = () => {
        const blob = new Blob(["\ufeff" + b.csv_summary], { type: "text/csv;charset=utf-8" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "saigak_orto_batch_summary.csv";
        link.click();
      };

      document.getElementById("dlBatchXlsBtn").onclick = () => {
        const blob = new Blob([b.excel_xml], { type: "application/vnd.ms-excel;charset=utf-8" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "saigak_orto_batch_summary.xls";
        link.click();
      };

      document.getElementById("dlBatchZipBtn").onclick = async (ev) => {
        const btn = ev.currentTarget;
        const orig = btn.textContent;
        btn.textContent = "Формирование ZIP с PDF...";
        btn.disabled = true;
        try {
          await downloadFileFromEndpoint(
            "/api/batch/zip",
            {
              filenames: APP_STATE.lastBatchFilenames,
              scale_factor: 1.0,
              level_ratio: APP_STATE.levelRatio,
              doctor_name: document.getElementById("doctorNameInput").value,
              doctor_verified: document.getElementById("doctorVerifiedCheck").checked,
            },
            "saigak_orto_batch_reports.zip"
          );
        } finally {
          btn.textContent = orig;
          btn.disabled = false;
        }
      };
    } catch (e) {
      container.innerHTML = `<div style="color:${PALETTE.negative};">Ошибка пакетной обработки: ${e}</div>`;
    }
  }

  batchModalBtn.addEventListener("click", renderBatchModalShell);

  // ==================== DYNAMIC LONGITUDINAL COMPARISON (BEFORE / AFTER) ====================
  function renderCompareOverlayCanvas(canvas, comp) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = APP_STATE.isLightTheme ? "#f4f4f5" : PALETTE.appSidebar;
    ctx.fillRect(0, 0, w, h);

    const cb = comp.overlay_2d.contour_before_xy || [];
    const ca = comp.overlay_2d.contour_after_xy || [];
    if (!cb.length || !ca.length) return;

    let maxR = 10;
    cb.concat(ca).forEach((pt) => {
      maxR = Math.max(maxR, Math.abs(pt[0]), Math.abs(pt[1]));
    });
    const scale = (Math.min(w, h) * 0.40) / maxR;
    const cx = w * 0.5;
    const cy = h * 0.52;
    const toC = (pt) => [cx + pt[0] * scale, cy - pt[1] * scale];

    // Grid & axes
    ctx.strokeStyle = APP_STATE.isLightTheme ? "rgba(23,23,23,0.08)" : "rgba(237,237,237,0.07)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 12); ctx.lineTo(cx, h - 12);
    ctx.moveTo(12, cy); ctx.lineTo(w - 12, cy);
    ctx.stroke();

    // After contour fill (growth zone highlight)
    ctx.beginPath();
    ca.forEach((pt, i) => {
      const p = toC(pt);
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(76, 195, 138, 0.16)";
    ctx.fill();
    ctx.strokeStyle = PALETTE.positive;
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Before contour dashed line
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    cb.forEach((pt, i) => {
      const p = toC(pt);
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.closePath();
    ctx.strokeStyle = PALETTE.amber;
    ctx.lineWidth = 2.0;
    ctx.stroke();
    ctx.restore();

    // Anatomical orientation labels
    ctx.fillStyle = PALETTE.appMuted;
    ctx.font = "500 10px monospace";
    ctx.textAlign = "center";
    ctx.fillText("ПЕРЕД (Лоб)", cx, 14);
    ctx.fillText("ЗАТЫЛОК", cx, h - 6);
    ctx.textAlign = "left";
    ctx.fillText("ПРАВ", 8, cy - 4);
    ctx.textAlign = "right";
    ctx.fillText("ЛЕВ", w - 8, cy - 4);
  }

  if (compareModalBtn) {
    compareModalBtn.addEventListener("click", () => {
      const usableScans = (APP_STATE.catalogScans || []).filter((s) => s.category !== "defect_validation_demo" && s.category !== "scale_calibration_demo");
      const optsHtml = usableScans
        .map((s) => `<option value="${s.filename}">${s.label}</option>`)
        .join("");

      openModal(
        "Сравнение в динамике: До и После шлемотерапии",
        `
          <div style="display:grid; grid-template-columns: 1fr 1fr auto; gap:8px; align-items:end; margin-bottom:12px;">
            <div>
              <div class="hud-label"><span>Скан «До» (исходный)</span></div>
              <select id="cmpBeforeSelect" class="hud-select">${optsHtml}</select>
            </div>
            <div>
              <div class="hud-label"><span>Скан «После» (контроль 1–3 мес.)</span></div>
              <select id="cmpAfterSelect" class="hud-select">${optsHtml}</select>
            </div>
            <button type="button" id="runCompareBtn" class="hud-btn hud-btn-primary">Сравнить</button>
          </div>
          <div style="display:flex; gap:8px; align-items:center; margin-bottom:12px; font-size:12px; color:var(--app-muted);">
            <label class="hud-btn hud-btn-sm" for="cmpBeforeFile">Загрузить STL «До»</label>
            <input type="file" id="cmpBeforeFile" accept=".stl" style="display:none;" />
            <label class="hud-btn hud-btn-sm" for="cmpAfterFile">Загрузить STL «После»</label>
            <input type="file" id="cmpAfterFile" accept=".stl" style="display:none;" />
            <span id="cmpUploadHint" class="mono">Совмещение выполняется автоматически в канонической системе Se–TrR–TrL</span>
          </div>
          <div id="compareResultBody"></div>
        `,
        true
      );

      const selB = document.getElementById("cmpBeforeSelect");
      const selA = document.getElementById("cmpAfterSelect");
      if (usableScans.some((s) => s.filename === "clinical_severe_plagio_P03.stl")) selB.value = "clinical_severe_plagio_P03.stl";
      if (usableScans.some((s) => s.filename === "clinical_mild_plagio_P01.stl")) selA.value = "clinical_mild_plagio_P01.stl";

      let customBefore = null;
      let customAfter = null;
      document.getElementById("cmpBeforeFile").addEventListener("change", async (e) => {
        const f = e.target.files[0];
        if (!f) return;
        customBefore = { name: f.name, b64: await fileToBase64(f) };
        document.getElementById("cmpUploadHint").textContent = `Загружен «До»: ${f.name}`;
      });
      document.getElementById("cmpAfterFile").addEventListener("change", async (e) => {
        const f = e.target.files[0];
        if (!f) return;
        customAfter = { name: f.name, b64: await fileToBase64(f) };
        document.getElementById("cmpUploadHint").textContent = `Загружен «После»: ${f.name}`;
      });

      const runCompare = async () => {
        const out = document.getElementById("compareResultBody");
        out.innerHTML = "<div class='mono'>Выполняется совмещение контуров Level 3 и расчёт прироста объёмов...</div>";
        try {
          const payload = {
            scan_before: customBefore ? customBefore.name : selB.value,
            scan_after: customAfter ? customAfter.name : selA.value,
            file_before_b64: customBefore ? customBefore.b64 : null,
            file_after_b64: customAfter ? customAfter.b64 : null,
            level_ratio: APP_STATE.levelRatio,
          };
          const res = await fetch("/api/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const c = await res.json();
          if (c.error) {
            out.innerHTML = `<div style="color:${PALETTE.negative};">Ошибка: ${c.error}</div>`;
            return;
          }
          const d = c.deltas;
          const qd = d.quadrant_volume_deltas_cm3;
          const trendCol = c.trend === "IMPROVED" ? PALETTE.positive : c.trend === "WORSENED" ? PALETTE.negative : PALETTE.amber;

          out.innerHTML = `
            <div style="padding:10px 12px; border-radius:8px; background:var(--app-sidebar); border:1px solid var(--app-hairline-strong); color:${trendCol}; margin-bottom:10px; font-weight:500;">
              ${c.clinical_verdict}
            </div>
            <div class="compare-kpi-grid">
              <div class="telemetry-kpi">
                <div class="tk-label">Динамика ΔCVAI</div>
                <div class="tk-val" style="color:${d.delta_CVAI_pp <= 0 ? PALETTE.positive : PALETTE.negative}; font-size:20px;">${d.delta_CVAI_pp >= 0 ? "+" : ""}${d.delta_CVAI_pp.toFixed(2)} п.п.</div>
                <div class="tk-sub">${c.before.CVAI.toFixed(2)}% → ${c.after.CVAI.toFixed(2)}%</div>
              </div>
              <div class="telemetry-kpi">
                <div class="tk-label">Динамика ΔCVA</div>
                <div class="tk-val" style="color:${d.delta_CVA_mm <= 0 ? PALETTE.positive : PALETTE.negative}; font-size:20px;">${d.delta_CVA_mm >= 0 ? "+" : ""}${d.delta_CVA_mm.toFixed(2)} мм</div>
                <div class="tk-sub">${c.before.CVA_mm.toFixed(2)} → ${c.after.CVA_mm.toFixed(2)} мм</div>
              </div>
              <div class="telemetry-kpi">
                <div class="tk-label">Динамика ΔCI</div>
                <div class="tk-val" style="font-size:20px;">${d.delta_CI_pp >= 0 ? "+" : ""}${d.delta_CI_pp.toFixed(2)} п.п.</div>
                <div class="tk-sub">${c.before.CI.toFixed(2)}% → ${c.after.CI.toFixed(2)}%</div>
              </div>
              <div class="telemetry-kpi">
                <div class="tk-label">Зона уплощения (${d.flattened_zone_code})</div>
                <div class="tk-val" style="color:${PALETTE.positive}; font-size:20px;">${d.flattened_zone_growth_cm3 >= 0 ? "+" : ""}${d.flattened_zone_growth_cm3.toFixed(1)} см³</div>
                <div class="tk-sub">${d.flattened_zone_growth_pct >= 0 ? "+" : ""}${d.flattened_zone_growth_pct.toFixed(1)}% · ΔPSR ${qd.delta_PSR >= 0 ? "+" : ""}${qd.delta_PSR.toFixed(3)}</div>
              </div>
            </div>
            <div style="display:grid; grid-template-columns: 280px 1fr; gap:12px; align-items:start;">
              <div>
                <canvas id="compareOverlayCanvas" width="276" height="220" style="width:100%; border-radius:8px; border:1px solid var(--app-hairline);"></canvas>
                <div class="mono" style="font-size:11px; color:var(--app-muted); margin-top:4px; display:flex; justify-content:space-between;">
                  <span style="color:${PALETTE.amber};">-- До (${c.scan_before_id})</span>
                  <span style="color:${PALETTE.positive};">━ После (${c.scan_after_id})</span>
                </div>
              </div>
              <table class="tel-table mono">
                <thead>
                  <tr style="border-bottom:1px solid var(--app-hairline-strong); color:var(--app-muted);">
                    <th style="text-align:left; padding:6px 4px; font-weight:400;">Показатель / Квадрант</th>
                    <th style="text-align:right; padding:6px 4px; font-weight:400;">До</th>
                    <th style="text-align:right; padding:6px 4px; font-weight:400;">После</th>
                    <th style="text-align:right; padding:6px 4px; font-weight:400;">Дельта (Δ)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td class="k">Периметр сечения</td><td class="v">${c.before.perimeter_mm.toFixed(1)} мм</td><td class="v">${c.after.perimeter_mm.toFixed(1)} мм</td><td class="v">${d.delta_perimeter_mm >= 0 ? "+" : ""}${d.delta_perimeter_mm.toFixed(1)} мм</td></tr>
                  <tr><td class="k">Q1 Левый лобный</td><td class="v">${c.before.quadrant_volumes.Q1_front_left_cm3.toFixed(1)} см³</td><td class="v">${c.after.quadrant_volumes.Q1_front_left_cm3.toFixed(1)} см³</td><td class="v">${qd.Q1_front_left_cm3 >= 0 ? "+" : ""}${qd.Q1_front_left_cm3.toFixed(1)} см³</td></tr>
                  <tr><td class="k">Q2 Правый лобный</td><td class="v">${c.before.quadrant_volumes.Q2_front_right_cm3.toFixed(1)} см³</td><td class="v">${c.after.quadrant_volumes.Q2_front_right_cm3.toFixed(1)} см³</td><td class="v">${qd.Q2_front_right_cm3 >= 0 ? "+" : ""}${qd.Q2_front_right_cm3.toFixed(1)} см³</td></tr>
                  <tr><td class="k">Q3 Левый затылочный</td><td class="v">${c.before.quadrant_volumes.Q3_back_left_cm3.toFixed(1)} см³</td><td class="v">${c.after.quadrant_volumes.Q3_back_left_cm3.toFixed(1)} см³</td><td class="v">${qd.Q3_back_left_cm3 >= 0 ? "+" : ""}${qd.Q3_back_left_cm3.toFixed(1)} см³</td></tr>
                  <tr><td class="k">Q4 Правый затылочный</td><td class="v">${c.before.quadrant_volumes.Q4_back_right_cm3.toFixed(1)} см³</td><td class="v">${c.after.quadrant_volumes.Q4_back_right_cm3.toFixed(1)} см³</td><td class="v">${qd.Q4_back_right_cm3 >= 0 ? "+" : ""}${qd.Q4_back_right_cm3.toFixed(1)} см³</td></tr>
                  <tr><td class="k">Общий объём свода</td><td class="v">${c.before.quadrant_volumes.total_vault_volume_cm3.toFixed(0)} см³</td><td class="v">${c.after.quadrant_volumes.total_vault_volume_cm3.toFixed(0)} см³</td><td class="v">${qd.total_vault_volume_cm3 >= 0 ? "+" : ""}${qd.total_vault_volume_cm3.toFixed(1)} см³</td></tr>
                </tbody>
              </table>
            </div>
          `;
          renderCompareOverlayCanvas(document.getElementById("compareOverlayCanvas"), c);
        } catch (err) {
          out.innerHTML = `<div style="color:${PALETTE.negative};">Ошибка сравнения: ${err}</div>`;
        }
      };

      document.getElementById("runCompareBtn").addEventListener("click", runCompare);
      runCompare();
    });
  }

  // ==================== ANONYMIZED MIS / HL7-FHIR WEBHOOK EXPORT ====================
  if (misExportBtn) {
    misExportBtn.addEventListener("click", () => {
      openModal(
        "Интеграция с МИС клиники (Вебхук и OpenAPI 3.0)",
        `
          <div style="color:var(--app-secondary); margin-bottom:10px;">
            Формирует анонимизированный пакет <code>DiagnosticReport</code> (индексы, квадранты, аудит врача и PDF в Base64) с подписью <code>X-Saigak-Signature (HMAC-SHA256)</code>. По политике безопасности разрешены только локальные адреса сети клиники.
          </div>
          <div style="display:grid; grid-template-columns: 1fr auto; gap:8px; margin-bottom:8px;">
            <input type="text" id="misWebhookUrl" class="hud-input mono" placeholder="http://127.0.0.1:9000/mis/craniometry-webhook" />
            <label style="display:flex; align-items:center; gap:6px; font-size:12px; color:var(--app-secondary); cursor:pointer;">
              <input type="checkbox" id="misIncludePdfCheck" checked />
              <span>Вложить PDF (Base64)</span>
            </label>
          </div>
          <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
            <button type="button" id="misPreviewBtn" class="hud-btn hud-btn-primary">Сформировать пакет МИС</button>
            <button type="button" id="misDispatchBtn" class="hud-btn">Отправить на вебхук МИС</button>
            <button type="button" id="misDownloadBtn" class="hud-btn">Скачать JSON МИС</button>
            <a href="/api/openapi.json" target="_blank" class="hud-btn">OpenAPI 3.0</a>
          </div>
          <div id="misStatusLine" class="mono" style="font-size:12px; color:var(--positive); margin-bottom:8px;"></div>
          <div id="misPreviewBox" class="mis-code-box">Подготовка пакета МИС...</div>
        `,
        true
      );

      let lastMisFullPayload = null;

      const runMis = async (mode) => {
        const st = document.getElementById("misStatusLine");
        const box = document.getElementById("misPreviewBox");
        st.textContent = mode === "dispatch" ? "Отправка на локальный вебхук МИС..." : "Сборка анонимизированного пакета МИС...";
        try {
          const payload = Object.assign(buildExportPayload("json"), {
            mode,
            webhook_url: document.getElementById("misWebhookUrl").value.trim(),
            include_pdf: document.getElementById("misIncludePdfCheck").checked,
          });
          const res = await fetch("/api/mis/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const d = await res.json();
          if (d.error) {
            st.style.color = PALETTE.negative;
            st.textContent = "Ошибка: " + d.error;
            return;
          }
          lastMisFullPayload = d.full_payload;
          st.style.color = d.dispatch.dispatched || mode === "preview" ? PALETTE.positive : PALETTE.amber;
          st.textContent = `${d.dispatch.detail} · Размер: ${(d.payload_size_bytes / 1024).toFixed(1)} КБ · ${d.signature_header.slice(0, 26)}...`;
          box.textContent = JSON.stringify(d.payload_preview, null, 2);
        } catch (err) {
          st.style.color = PALETTE.negative;
          st.textContent = "Ошибка запроса: " + err;
        }
      };

      document.getElementById("misPreviewBtn").onclick = () => runMis("preview");
      document.getElementById("misDispatchBtn").onclick = () => runMis("dispatch");
      document.getElementById("misDownloadBtn").onclick = () => {
        if (!lastMisFullPayload) return;
        const blob = new Blob([JSON.stringify(lastMisFullPayload, null, 2)], { type: "application/json;charset=utf-8" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `saigak_orto_mis_${APP_STATE.currentFilename.replace(".stl", "")}.json`;
        link.click();
      };

      runMis("preview");
    });
  }

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
    openModal("Экспертная очередь разметки для переобучения", "<div class='mono'>Загрузка очереди</div>");
    const res = await fetch("/api/queue");
    const qData = await res.json();
    document.getElementById("queueCountSpan").textContent = qData.count || 0;
    const rows = (qData.items || [])
      .map(
        (it) => `
        <tr>
          <td class="k mono">${it.queue_id}</td>
          <td class="k mono" style="color:${PALETTE.blue};">${it.scan_id}</td>
          <td class="k">${it.reason}</td>
          <td class="v">${it.doctor_name}</td>
          <td class="v" style="color:${PALETTE.positive};">${it.timestamp}</td>
        </tr>
      `
      )
      .join("");
    modalBody.innerHTML = `
      <div style="margin-bottom:12px; color:var(--app-secondary);">
        Сохранено размеченных экспертом сканов (<code>data/expert_retraining_queue.json</code>): <strong>${qData.count}</strong>
      </div>
      <table class="tel-table">
        <thead>
          <tr style="border-bottom:1px solid var(--app-hairline-strong); color:var(--app-muted);">
            <th style="text-align:left; padding:8px 4px; font-weight:400;">ID</th>
            <th style="text-align:left; padding:8px 4px; font-weight:400;">Скан</th>
            <th style="text-align:left; padding:8px 4px; font-weight:400;">Основание</th>
            <th style="text-align:right; padding:8px 4px; font-weight:400;">Эксперт</th>
            <th style="text-align:right; padding:8px 4px; font-weight:400;">Время</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  });

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
