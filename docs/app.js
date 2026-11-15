/**
 * UI controller, high-DPI multi-trace telemetry charts, and bundle hydrator.
 * Connects Bioreactor3DView + HybridTwinEngine + exported GAM-86 (B=15) bundle.
 */

(function (global) {
  "use strict";

  function fmt(v, digits) {
    if (v == null || !Number.isFinite(v)) return "--";
    return Number(v).toFixed(digits == null ? 2 : digits);
  }

  function clampVal(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function getThemeColors() {
    const cs = getComputedStyle(document.documentElement);
    const get = (k, fb) => cs.getPropertyValue(k).trim() || fb;
    return {
      bg: get("--bg-canvas", "#f7f6f2"),
      panel: get("--bg-panel", "#efece6"),
      recessed: get("--bg-recessed", "#e5e0d6"),
      border: get("--border-subtle", "#d5d0c5"),
      borderStrong: get("--border-strong", "#756e61"),
      ink: get("--ink-primary", "#181a1b"),
      muted: get("--ink-secondary", "#4a5056"),
      gt: get("--trace-gt", "#0f766e"),
      gtInk: get("--trace-gt-ink", "#0d6861"),
      gtWash: get("--trace-gt-wash", "#dce8e4"),
      hyb: get("--trace-hyb", "#b45309"),
      hybInk: get("--trace-hyb-ink", "#92400e"),
      hybWash: get("--trace-hyb-wash", "#e8dfd0"),
      hybBand: get("--trace-hyb-band", "rgba(180, 83, 9, 0.16)"),
      nom: get("--trace-nom", "#475569"),
      alert: get("--trace-alert", "#b91c1c"),
      alertInk: get("--trace-alert-ink", "#991b1b"),
      alertWash: get("--trace-alert-wash", "#f2e2e0"),
    };
  }

  function setTextById(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function renderKaTeXMath(rootEl) {
    if (!global.katex || typeof global.katex.render !== "function") return;
    const nodes = (rootEl || document).querySelectorAll("[data-katex]");
    nodes.forEach((node) => {
      const expr = node.getAttribute("data-katex");
      if (!expr) return;
      try {
        global.katex.render(expr, node, {
          throwOnError: false,
          displayMode: node.getAttribute("data-display") === "true",
        });
      } catch (_err) {
        // Leave fallback text intact if expression fails
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    const bundle = global.HYBRID_TWIN_BUNDLE;
    const engine = global.HybridTwinEngine;
    const statusBanner = document.getElementById("sim-status-banner");
    const statusText = document.getElementById("sim-status-text");

    if (!bundle || !engine) {
      if (statusBanner) statusBanner.setAttribute("data-state", "error");
      if (statusText) {
        statusText.textContent =
          "[ERROR] Failed to load twin_bundle.js or twin_engine.js. Verify docs/assets path and reload.";
      }
      return;
    }

    const appState = {
      theme: localStorage.getItem("hft_theme_v2") || "light",
      scaleL: 5,
      controlMode: "closed_loop",
      muTarget: 0.26,
      manualFeedRate: 0.08,
      rpm: 450,
      vvm: 1.2,
      disturbanceMode: "none",
      showTruth: false,
      timeIdx: 160,
      playing: false,
      hoverIdx: null,
      sim: null,
      reactor3D: null,
      sec2Reactor3D: null,
      monographRedraw: null,
    };

    const elThemeBtn = document.getElementById("theme-toggle-btn");
    const elMobileNavBtn = document.getElementById("mobile-nav-toggle");
    const elPrimaryNav = document.getElementById("primary-nav");
    const elScale5Btn = document.getElementById("scale-5l-btn");
    const elScale1000Btn = document.getElementById("scale-1000l-btn");
    const elVesselLabel = document.getElementById("vessel-mode-label");
    const elCtrlSelect = document.getElementById("ctrl-mode-select");
    const elCtrlTag = document.getElementById("ctrl-mode-tag");
    const elRowMu = document.getElementById("row-mu-target");
    const elRowFeed = document.getElementById("row-manual-feed");
    const elSliderMu = document.getElementById("slider-mu-target");
    const elValMu = document.getElementById("val-mu-target");
    const elSliderFeed = document.getElementById("slider-manual-feed");
    const elValFeed = document.getElementById("val-manual-feed");
    const elSliderRpm = document.getElementById("slider-rpm");
    const elValRpm = document.getElementById("val-rpm");
    const elSliderVvm = document.getElementById("slider-vvm");
    const elValVvm = document.getElementById("val-vvm");
    const elSelectDist = document.getElementById("select-disturbance");
    const elSliderTime = document.getElementById("slider-time");
    const elValTime = document.getElementById("val-time");
    const elPlayBtn = document.getElementById("play-batch-btn");
    const elJumpEndBtn = document.getElementById("jump-end-btn");
    const elToggleTruth = document.getElementById("toggle-show-truth");
    const elParityNote = document.getElementById("operator-parity-note");
    const elLegendGt = document.getElementById("legend-gt-item");
    const elTimingBadge = document.getElementById("sim-timing-badge");
    const elMotionBtn = document.getElementById("toggle-motion-btn");
    const elResetCamBtn = document.getElementById("reset-camera-btn");
    const elResetDefBtn = document.getElementById("reset-defaults-btn");

    // Mobile navigation drawer with Escape support
    if (elMobileNavBtn && elPrimaryNav) {
      function setNavOpen(open) {
        elPrimaryNav.classList.toggle("open", open);
        elMobileNavBtn.setAttribute("aria-expanded", open ? "true" : "false");
        elMobileNavBtn.textContent = open ? "Close" : "Menu";
      }

      elMobileNavBtn.addEventListener("click", function () {
        setNavOpen(!elPrimaryNav.classList.contains("open"));
      });

      elPrimaryNav.querySelectorAll("a").forEach((link) => {
        link.addEventListener("click", () => setNavOpen(false));
      });

      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && elPrimaryNav.classList.contains("open")) {
          setNavOpen(false);
          elMobileNavBtn.focus();
        }
      });
    }

    function applyTheme(nextTheme) {
      appState.theme = nextTheme;
      document.documentElement.setAttribute("data-theme", nextTheme);
      localStorage.setItem("hft_theme_v2", nextTheme);
      elThemeBtn.textContent = nextTheme === "light" ? "Theme: Light" : "Theme: Dark";
      if (appState.reactor3D) appState.reactor3D.setTheme(nextTheme);
      if (appState.sec2Reactor3D) appState.sec2Reactor3D.setTheme(nextTheme);
      drawAllCharts();
      if (typeof appState.monographRedraw === "function") appState.monographRedraw();
    }

    const host3d = document.getElementById("reactor-3d-host");
    if (global.Bioreactor3DView && host3d) {
      appState.reactor3D = new global.Bioreactor3DView(host3d, { theme: appState.theme });
      if (appState.reactor3D.motionPaused) {
        elMotionBtn.textContent = "Resume 3D Motion";
      }
    }
    const hostSec2 = document.getElementById("sec2-reactor-3d-host");
    if (global.Bioreactor3DView && hostSec2) {
      appState.sec2Reactor3D = new global.Bioreactor3DView(hostSec2, { theme: appState.theme });
      appState.sec2Reactor3D.setHighlightZone("growth");
    }
    applyTheme(appState.theme);

    function syncControlUI() {
      const sf = appState.scaleL / 5.0;
      if (appState.scaleL === 1000) {
        elSliderRpm.min = "60";
        elSliderRpm.max = "300";
        elSliderRpm.step = "5";
        elSliderVvm.min = "0.05";
        elSliderVvm.max = "0.60";
        elSliderVvm.step = "0.01";
        elSliderFeed.min = String((0.01 * sf).toFixed(1));
        elSliderFeed.max = String((0.28 * sf).toFixed(1));
        elSliderFeed.step = "1.0";
        elVesselLabel.textContent = "1000 L Stainless Cutaway";
      } else {
        elSliderRpm.min = "200";
        elSliderRpm.max = "900";
        elSliderRpm.step = "10";
        elSliderVvm.min = "0.30";
        elSliderVvm.max = "2.50";
        elSliderVvm.step = "0.05";
        elSliderFeed.min = "0.01";
        elSliderFeed.max = "0.28";
        elSliderFeed.step = "0.005";
        elVesselLabel.textContent = "5 L Borosilicate Benchtop";
      }

      elSliderMu.value = String(appState.muTarget);
      elValMu.textContent = fmt(appState.muTarget, 2) + " 1/h";
      elSliderFeed.value = String(appState.manualFeedRate);
      elValFeed.textContent = fmt(appState.manualFeedRate, appState.scaleL >= 500 ? 1 : 3) + " L/h";
      elSliderRpm.value = String(appState.rpm);
      elValRpm.textContent = fmt(appState.rpm, 0) + " RPM";
      elSliderVvm.value = String(appState.vvm);
      elValVvm.textContent = fmt(appState.vvm, 2) + " vvm";

      const isClosed = appState.controlMode === "closed_loop";
      elRowMu.style.display = isClosed ? "" : "none";
      elRowFeed.style.display = isClosed ? "none" : "";
      elCtrlTag.textContent = isClosed ? "Closed-Loop mu-Stat" : "Manual Feed";

      elToggleTruth.checked = appState.showTruth;
      elLegendGt.style.display = appState.showTruth ? "inline-flex" : "none";
      elParityNote.textContent = appState.showTruth
        ? "Validation Oracle ON: comparing Ground-Truth Plant against Hybrid EKF (+/- 2 sigma) and Nominal Baseline."
        : "5 L Operator Mode hides plant truth by default (Portela Level 1 offline simulator / Level 2-ready EKF observer): displaying EKF state +/- 2 sigma, online PAT, and sparse at-line LIMS samples.";
    }

    function runAndRenderSimulation() {
      if (statusBanner) statusBanner.setAttribute("data-state", "loading");
      const t0 = performance.now();
      try {
        appState.sim = engine.runTwinSimulation(
          {
            scaleL: appState.scaleL,
            controlMode: appState.controlMode,
            muTarget: appState.muTarget,
            manualFeedRate: appState.manualFeedRate,
            rpm: appState.rpm,
            vvm: appState.vvm,
            disturbanceMode: appState.disturbanceMode,
          },
          bundle
        );
      } catch (err) {
        if (statusBanner) statusBanner.setAttribute("data-state", "error");
        if (statusText) {
          statusText.textContent = "[ERROR] Simulation failed: " + (err && err.message ? err.message : String(err));
        }
        return;
      }
      const dtMs = performance.now() - t0;
      elTimingBadge.textContent = fmt(dtMs, 1) + " ms (160 steps, B=15 EKF)";
      renderFrameAt(appState.timeIdx);
    }

    function renderFrameAt(idx) {
      const sim = appState.sim;
      if (!sim) return;
      const i = Math.max(0, Math.min(sim.t.length - 1, Math.round(idx)));
      appState.timeIdx = i;
      elSliderTime.value = String(i);
      elValTime.textContent = fmt(sim.t[i], 1) + " h";

      const xEst = sim.estHyb[i];
      const xStd = sim.stdHyb[i];
      const xTrue = sim.trueStates[i];
      const xNom = sim.estNom[i];
      const u = sim.inputs[i];
      const tH = sim.t[i];
      const doEstPct = (100.0 * xEst[3]) / sim.C_star;
      const inDom = Boolean(sim.inDomain[i]);
      const dm = sim.d_m[i];

      if (appState.reactor3D) {
        appState.reactor3D.updateState({
          t_h: tH,
          X_v: xEst[0],
          X_d: xEst[1],
          S: xEst[2],
          C_L: xEst[3],
          DO_pct: doEstPct,
          A: xEst[4],
          P: xEst[5],
          V: xEst[6],
          V_max: sim.V_max,
          F: u[0],
          rpm: u[1],
          vvm: u[2],
          v_tip: sim.v_tip[i],
          Pg_V: sim.Pg_V[i],
          k_La: sim.klaHyb[i],
          in_domain: inDom,
          d_m: dm,
        });
      }

      document.getElementById("hud-time").textContent = "t = " + fmt(tH, 1) + " h";
      const hudDov = document.getElementById("hud-dov");
      if (inDom) {
        hudDov.className = "figure-dov-status";
        hudDov.textContent = "DoV IN-DOMAIN (D_M = " + fmt(dm, 2) + " <= " + fmt(sim.p99Global, 2) + ")";
      } else {
        hudDov.className = "figure-dov-status ood";
        hudDov.textContent = "OOD PRIOR RELAXED (D_M = " + fmt(dm, 2) + ")";
      }
      document.getElementById("hud-pgv").textContent = "Pg/V: " + fmt(sim.Pg_V[i], 0) + " W/m3";
      document.getElementById("hud-vtip").textContent = "v_tip: " + fmt(sim.v_tip[i], 2) + " m/s";
      document.getElementById("hud-kla").textContent = "k_La: " + fmt(sim.klaHyb[i], 0) + " 1/h";

      const summ = sim.summary;
      const redPct = 100.0 * (1.0 - summ.rmse_ratio);
      if (!inDom || summ.ood_fraction > 0.15) {
        statusBanner.setAttribute("data-state", "warning");
        statusText.textContent =
          "[OOD ALERT] t=" +
          fmt(tH, 1) +
          " h (D_M=" +
          fmt(dm, 2) +
          " > p99=" +
          fmt(sim.p99Global, 2) +
          "): metabolic heads relax toward mechanistic prior while inflating sigma.";
      } else {
        statusBanner.setAttribute("data-state", "ready");
        statusText.textContent =
          "[NOMINAL] " +
          appState.scaleL +
          " L Twin Active: full-batch Hybrid EKF cut error by " +
          fmt(redPct, 1) +
          "% vs Nominal EKF (Ratio " +
          fmt(summ.rmse_ratio, 3) +
          "x).";
      }

      document.getElementById("kpi-xv").textContent = fmt(xEst[0], 2) + " +- " + fmt(2 * xStd[0], 2);
      document.getElementById("kpi-xv-sub").textContent = appState.showTruth
        ? "True: " + fmt(xTrue[0], 2) + " | Nom: " + fmt(xNom[0], 2)
        : "Online PAT capacitance + EKF";

      document.getElementById("kpi-xd").textContent = fmt(xEst[1], 2) + " +- " + fmt(2 * xStd[1], 2);
      document.getElementById("kpi-xd-sub").textContent = appState.showTruth
        ? "True X_d: " + fmt(xTrue[1], 2) + " | Nom: " + fmt(xNom[1], 2)
        : "Inferred (zero X_d sensor)";

      document.getElementById("kpi-do").textContent = fmt(doEstPct, 1) + "%";
      document.getElementById("kpi-do-sub").textContent = appState.showTruth
        ? "C_L=" + fmt(xEst[3], 2) + " vs True " + fmt(xTrue[3], 2) + " mg/L"
        : "C_L=" + fmt(xEst[3], 2) + " mg/L (min " + fmt(summ.min_DO_est, 1) + "%)";

      document.getElementById("kpi-s").textContent = fmt(xEst[2], 3) + " g/L";
      document.getElementById("kpi-s-sub").textContent = appState.showTruth
        ? "True S: " + fmt(xTrue[2], 3) + " g/L (S_crit 0.35)"
        : xEst[2] > 0.35
          ? "Overflow zone (S > 0.35)"
          : "Below overflow ceiling";

      document.getElementById("kpi-ap").textContent = fmt(xEst[4], 2) + " / " + fmt(xEst[5], 2);
      document.getElementById("kpi-ap-sub").textContent = appState.showTruth
        ? "True A/P: " + fmt(xTrue[4], 2) + " / " + fmt(xTrue[5], 2) + " g/L"
        : "At-line A & Luedeking-Piret P";

      const fDec = appState.scaleL >= 500 ? 1 : 3;
      document.getElementById("kpi-vf").textContent =
        fmt(xEst[6], appState.scaleL >= 500 ? 0 : 2) + " / " + fmt(u[0], fDec);
      document.getElementById("kpi-vf-sub").textContent =
        "k_La=" + fmt(sim.klaHyb[i], 0) + " vs nom " + fmt(sim.klaNom[i], 0) + " 1/h";

      document.getElementById("kpi-mu").textContent = fmt(sim.muHyb[i], 3) + " 1/h";
      document.getElementById("kpi-mu-sub").textContent = appState.showTruth
        ? "True mu=" + fmt(sim.muTrue[i], 3) + " | Nom=" + fmt(sim.muNom[i], 3)
        : "Target mu_set=" + fmt(appState.muTarget, 2) + " 1/h";

      document.getElementById("kpi-ekf").textContent = fmt(summ.rmse_ratio, 3) + "x";
      document.getElementById("kpi-ekf-sub").textContent =
        "Hyb " + fmt(summ.hyb_norm_rmse, 4) + " vs Nom " + fmt(summ.nom_norm_rmse, 4);

      drawAllCharts();
    }

    function setupHiDpiCanvas(canvas) {
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = Math.min(global.devicePixelRatio || 1, 2);
      const w = Math.max(220, rect.width);
      const h = Math.max(160, rect.height);
      const tw = Math.round(w * dpr);
      const th = Math.round(h * dpr);
      if (canvas.width !== tw || canvas.height !== th) {
        canvas.width = tw;
        canvas.height = th;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx: ctx, w: w, h: h };
    }

    function drawTelemetryChart(canvas, spec) {
      const sim = appState.sim;
      if (!canvas || !sim) return;
      const C = getThemeColors();
      const { ctx, w, h } = setupHiDpiCanvas(canvas);
      ctx.clearRect(0, 0, w, h);

      const pad = { l: 46, r: 46, t: 18, b: 26 };
      const pw = Math.max(40, w - pad.l - pad.r);
      const ph = Math.max(40, h - pad.t - pad.b);
      const N = sim.t.length;
      const tMax = sim.t[N - 1] || 16.0;

      const xPx = (k) => pad.l + (sim.t[k] / tMax) * pw;
      const yPx = (v, minV, maxV) => pad.t + ph * (1.0 - (v - minV) / Math.max(1e-9, maxV - minV));

      let maxL = spec.minMaxLeft || 1.0;
      const minL = spec.fixedMinLeft != null ? spec.fixedMinLeft : 0.0;
      let maxR = spec.minMaxRight || 1.0;
      const minR = 0.0;

      for (const s of spec.series) {
        for (let k = 0; k < N; k++) {
          const val = s.get(k);
          if (!Number.isFinite(val)) continue;
          const up = s.band ? val + s.band(k) : val;
          if (s.axis === "right") {
            if (up > maxR) maxR = up;
          } else {
            if (up > maxL) maxL = up;
          }
        }
      }
      maxL *= 1.12;
      maxR *= 1.15;

      // Axis unit headers
      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.muted;
      if (spec.leftUnit) {
        ctx.textAlign = "left";
        ctx.fillText(spec.leftUnit, pad.l, 11);
      }
      if (spec.hasRightAxis && spec.rightUnit) {
        ctx.textAlign = "right";
        ctx.fillText(spec.rightUnit, w - pad.r, 11);
      }

      // Horizontal gridlines and numeric ticks
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const y = Math.round(pad.t + (ph * g) / 4) + 0.5;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();

        const lVal = maxL - ((maxL - minL) * g) / 4;
        ctx.textAlign = "right";
        ctx.fillStyle = C.muted;
        ctx.fillText(fmt(lVal, maxL < 2 ? 2 : maxL < 10 ? 1 : 0), pad.l - 5, y + 3);

        if (spec.hasRightAxis) {
          const rVal = maxR - ((maxR - minR) * g) / 4;
          ctx.textAlign = "left";
          ctx.fillText(fmt(rVal, maxR < 2 ? 2 : maxR < 10 ? 1 : 0), w - pad.r + 5, y + 3);
        }
      }

      // X-axis time ticks (0, 4, 8, 12, 16 h)
      ctx.textAlign = "center";
      ctx.fillStyle = C.muted;
      for (let hr = 0; hr <= 16; hr += 4) {
        const xp = pad.l + (hr / tMax) * pw;
        ctx.fillText(hr + "h", xp, h - 8);
      }

      // Threshold reference lines
      if (spec.thresholds) {
        for (const th of spec.thresholds) {
          const maxUse = th.axis === "right" ? maxR : maxL;
          const minUse = th.axis === "right" ? minR : minL;
          if (th.val > maxUse || th.val < minUse) continue;
          const yy = yPx(th.val, minUse, maxUse);
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = th.color || C.alert;
          ctx.lineWidth = 1.25;
          ctx.beginPath();
          ctx.moveTo(pad.l, yy);
          ctx.lineTo(w - pad.r, yy);
          ctx.stroke();
          ctx.restore();
          ctx.fillStyle = th.color || C.alert;
          ctx.textAlign = th.axis === "right" ? "right" : "left";
          ctx.fillText(th.label, th.axis === "right" ? w - pad.r - 4 : pad.l + 4, yy - 4);
        }
      }

      // Confidence bands (+/- 2 sigma)
      for (const s of spec.series) {
        if (!s.band || s.hidden) continue;
        const maxUse = s.axis === "right" ? maxR : maxL;
        const minUse = s.axis === "right" ? minR : minL;
        ctx.save();
        ctx.fillStyle = s.bandColor || C.hybBand;
        ctx.beginPath();
        for (let k = 0; k < N; k++) {
          const xp = xPx(k);
          const yp = yPx(Math.min(maxUse, s.get(k) + s.band(k)), minUse, maxUse);
          if (k === 0) ctx.moveTo(xp, yp);
          else ctx.lineTo(xp, yp);
        }
        for (let k = N - 1; k >= 0; k--) {
          const xp = xPx(k);
          const yp = yPx(Math.max(minUse, s.get(k) - s.band(k)), minUse, maxUse);
          ctx.lineTo(xp, yp);
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      // Line traces & sparse LIMS diamond markers
      for (const s of spec.series) {
        if (s.hidden) continue;
        const maxUse = s.axis === "right" ? maxR : maxL;
        const minUse = s.axis === "right" ? minR : minL;
        ctx.save();
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width || 2;
        if (s.dash) ctx.setLineDash(s.dash);
        ctx.beginPath();
        let started = false;
        for (let k = 0; k < N; k++) {
          const v = s.get(k);
          if (!Number.isFinite(v)) continue;
          const xp = xPx(k);
          const yp = yPx(clampVal(v, minUse, maxUse), minUse, maxUse);
          if (!started) {
            ctx.moveTo(xp, yp);
            started = true;
          } else {
            ctx.lineTo(xp, yp);
          }
        }
        ctx.stroke();
        ctx.restore();

        if (s.markers) {
          ctx.save();
          ctx.fillStyle = s.markerColor || C.ink;
          for (let k = 0; k < N; k++) {
            const mv = s.markers(k);
            if (!Number.isFinite(mv)) continue;
            const xp = xPx(k);
            const yp = yPx(clampVal(mv, minUse, maxUse), minUse, maxUse);
            ctx.beginPath();
            ctx.moveTo(xp, yp - 4);
            ctx.lineTo(xp + 4, yp);
            ctx.lineTo(xp, yp + 4);
            ctx.lineTo(xp - 4, yp);
            ctx.closePath();
            ctx.fill();
          }
          ctx.restore();
        }
      }

      // Synchronized playhead / hover crosshair + point markers
      const activeIdx = appState.hoverIdx != null ? appState.hoverIdx : appState.timeIdx;
      const cxp = xPx(activeIdx);
      ctx.save();
      ctx.strokeStyle = C.ink;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(cxp, pad.t);
      ctx.lineTo(cxp, pad.t + ph);
      ctx.stroke();
      ctx.restore();

      for (const s of spec.series) {
        if (s.hidden || s.dash) continue;
        const val = s.get(activeIdx);
        if (!Number.isFinite(val)) continue;
        const maxUse = s.axis === "right" ? maxR : maxL;
        const minUse = s.axis === "right" ? minR : minL;
        const yp = yPx(clampVal(val, minUse, maxUse), minUse, maxUse);
        ctx.save();
        ctx.fillStyle = s.color;
        ctx.strokeStyle = C.bg;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cxp, yp, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    }

    function drawAllCharts() {
      const sim = appState.sim;
      if (!sim) return;
      const C = getThemeColors();
      const idx = appState.hoverIdx != null ? appState.hoverIdx : appState.timeIdx;

      drawTelemetryChart(document.getElementById("chart-biomass"), {
        minMaxLeft: 20,
        minMaxRight: 3,
        hasRightAxis: true,
        leftUnit: "X_v (g/L)",
        rightUnit: "X_d (g/L)",
        series: [
          { get: (k) => sim.estNom[k][0], color: C.nom, width: 1.6, dash: [5, 4] },
          { get: (k) => sim.trueStates[k][0], color: C.gt, width: 2.0, hidden: !appState.showTruth },
          {
            get: (k) => sim.estHyb[k][0],
            band: (k) => 2.0 * sim.stdHyb[k][0],
            bandColor: C.hybBand,
            color: C.hyb,
            width: 2.2,
          },
          { axis: "right", get: (k) => sim.estHyb[k][1], color: C.alert, width: 1.7 },
          {
            axis: "right",
            get: (k) => sim.trueStates[k][1],
            color: C.gt,
            width: 1.3,
            dash: [2, 2],
            hidden: !appState.showTruth,
          },
        ],
      });
      document.getElementById("readout-chart-1").textContent =
        "t=" + fmt(sim.t[idx], 1) + "h | Xv=" + fmt(sim.estHyb[idx][0], 2) + " g/L | Xd=" + fmt(sim.estHyb[idx][1], 2) + " g/L";

      drawTelemetryChart(document.getElementById("chart-do"), {
        minMaxLeft: 9.0,
        minMaxRight: 700,
        hasRightAxis: true,
        leftUnit: "C_L (mg/L)",
        rightUnit: "k_La (1/h)",
        thresholds: [{ val: 1.64, label: "DO 20%", color: C.alert }],
        series: [
          { get: (k) => sim.estNom[k][3], color: C.nom, width: 1.5, dash: [5, 4] },
          { get: (k) => sim.trueStates[k][3], color: C.gt, width: 2.0, hidden: !appState.showTruth },
          {
            get: (k) => sim.estHyb[k][3],
            band: (k) => 2.0 * sim.stdHyb[k][3],
            bandColor: C.hybBand,
            color: C.hyb,
            width: 2.1,
          },
          { axis: "right", get: (k) => sim.klaHyb[k], color: C.gt, width: 1.5 },
        ],
      });
      document.getElementById("readout-chart-2").textContent =
        "t=" + fmt(sim.t[idx], 1) + "h | CL=" + fmt(sim.estHyb[idx][3], 2) + " mg/L | kLa=" + fmt(sim.klaHyb[idx], 0) + " 1/h";

      drawTelemetryChart(document.getElementById("chart-metabolites"), {
        minMaxLeft: 12.0,
        minMaxRight: 1.5,
        hasRightAxis: true,
        leftUnit: "A, P (g/L)",
        rightUnit: "S (g/L)",
        thresholds: [{ val: 0.35, axis: "right", label: "S_crit 0.35", color: C.alert }],
        series: [
          { get: (k) => sim.estNom[k][4], color: C.nom, width: 1.5, dash: [5, 4] },
          {
            get: (k) => sim.estHyb[k][4],
            color: C.alert,
            width: 2.0,
            markers: (k) => (sim.obsAtline[k] ? sim.measStates[k][4] : NaN),
            markerColor: C.alert,
          },
          { get: (k) => sim.estHyb[k][5], color: C.hyb, width: 2.0 },
          { get: (k) => sim.trueStates[k][5], color: C.gt, width: 1.6, hidden: !appState.showTruth },
          {
            axis: "right",
            get: (k) => sim.estHyb[k][2],
            color: C.gtInk,
            width: 1.7,
            markers: (k) => (sim.obsAtline[k] ? sim.measStates[k][2] : NaN),
            markerColor: C.ink,
          },
        ],
      });
      document.getElementById("readout-chart-3").textContent =
        "t=" + fmt(sim.t[idx], 1) + "h | S=" + fmt(sim.estHyb[idx][2], 3) + " | A=" + fmt(sim.estHyb[idx][4], 2) + " | P=" + fmt(sim.estHyb[idx][5], 2) + " g/L";

      drawTelemetryChart(document.getElementById("chart-control-dov"), {
        minMaxLeft: 0.48,
        minMaxRight: 4.2,
        hasRightAxis: true,
        leftUnit: "mu (1/h)",
        rightUnit: "D_M",
        thresholds: [
          { val: appState.muTarget, label: "mu_set", color: C.hybInk },
          { val: sim.p99Global, axis: "right", label: "DoV p99", color: C.alert },
        ],
        series: [
          { get: (k) => sim.muNom[k], color: C.nom, width: 1.4, dash: [5, 4] },
          { get: (k) => sim.muTrue[k], color: C.gt, width: 1.8, hidden: !appState.showTruth },
          { get: (k) => sim.muHyb[k], color: C.hyb, width: 2.1 },
          { axis: "right", get: (k) => sim.d_m[k], color: C.alert, width: 1.7 },
        ],
      });
      document.getElementById("readout-chart-4").textContent =
        "t=" + fmt(sim.t[idx], 1) + "h | mu=" + fmt(sim.muHyb[idx], 3) + " 1/h | D_M=" + fmt(sim.d_m[idx], 2);
    }

    for (const cid of ["chart-biomass", "chart-do", "chart-metabolites", "chart-control-dov"]) {
      const cv = document.getElementById(cid);
      if (!cv) continue;
      cv.addEventListener("pointermove", function (e) {
        if (!appState.sim) return;
        const rect = cv.getBoundingClientRect();
        const frac = clampVal((e.clientX - rect.left - 46) / Math.max(40, rect.width - 92), 0, 1);
        appState.hoverIdx = Math.round(frac * (appState.sim.t.length - 1));
        drawAllCharts();
      });
      cv.addEventListener("pointerleave", function () {
        appState.hoverIdx = null;
        drawAllCharts();
      });
    }

    let playTimer = null;
    function setPlaying(shouldPlay) {
      appState.playing = shouldPlay;
      elPlayBtn.textContent = shouldPlay ? "Pause Playback" : "Play 16 h Batch";
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
      }
      if (shouldPlay) {
        if (appState.timeIdx >= 160) appState.timeIdx = 0;
        playTimer = setInterval(function () {
          if (appState.timeIdx >= 160) {
            setPlaying(false);
            return;
          }
          renderFrameAt(appState.timeIdx + 1);
        }, 55);
      }
    }

    elThemeBtn.addEventListener("click", function () {
      applyTheme(appState.theme === "dark" ? "light" : "dark");
    });

    function switchScale(targetScale) {
      if (appState.scaleL === targetScale) return;
      const matched = engine.matchOperatingPointForScale(
        appState.rpm,
        appState.vvm,
        appState.scaleL,
        targetScale,
        bundle.p_rec
      );
      const scaleFactorRatio = targetScale / appState.scaleL;
      appState.scaleL = targetScale;
      appState.rpm = matched.rpm;
      appState.vvm = matched.vvm;
      appState.manualFeedRate = Number((appState.manualFeedRate * scaleFactorRatio).toFixed(3));
      appState.showTruth = targetScale === 1000;

      elScale5Btn.classList.toggle("active", targetScale === 5);
      elScale1000Btn.classList.toggle("active", targetScale === 1000);
      if (appState.reactor3D) appState.reactor3D.setScale(targetScale);
      syncControlUI();
      runAndRenderSimulation();
    }

    elScale5Btn.addEventListener("click", () => switchScale(5));
    elScale1000Btn.addEventListener("click", () => switchScale(1000));

    elCtrlSelect.addEventListener("change", function () {
      appState.controlMode = this.value;
      syncControlUI();
      runAndRenderSimulation();
    });

    elSliderMu.addEventListener("input", function () {
      appState.muTarget = Number(this.value);
      syncControlUI();
      runAndRenderSimulation();
    });

    elSliderFeed.addEventListener("input", function () {
      appState.manualFeedRate = Number(this.value);
      syncControlUI();
      runAndRenderSimulation();
    });

    elSliderRpm.addEventListener("input", function () {
      appState.rpm = Number(this.value);
      syncControlUI();
      runAndRenderSimulation();
    });

    elSliderVvm.addEventListener("input", function () {
      appState.vvm = Number(this.value);
      syncControlUI();
      runAndRenderSimulation();
    });

    elSelectDist.addEventListener("change", function () {
      appState.disturbanceMode = this.value;
      runAndRenderSimulation();
    });

    elSliderTime.addEventListener("input", function () {
      setPlaying(false);
      renderFrameAt(Number(this.value));
    });

    elPlayBtn.addEventListener("click", function () {
      setPlaying(!appState.playing);
    });

    elJumpEndBtn.addEventListener("click", function () {
      setPlaying(false);
      renderFrameAt(160);
    });

    elToggleTruth.addEventListener("change", function () {
      appState.showTruth = this.checked;
      syncControlUI();
      renderFrameAt(appState.timeIdx);
    });

    elMotionBtn.addEventListener("click", function () {
      if (!appState.reactor3D) return;
      const nextPaused = !appState.reactor3D.motionPaused;
      appState.reactor3D.setMotionPaused(nextPaused);
      elMotionBtn.textContent = nextPaused ? "Resume 3D Motion" : "Pause 3D Motion";
    });

    elResetCamBtn.addEventListener("click", function () {
      if (appState.reactor3D) appState.reactor3D.resetCamera();
    });

    elResetDefBtn.addEventListener("click", function () {
      setPlaying(false);
      appState.controlMode = "closed_loop";
      elCtrlSelect.value = "closed_loop";
      appState.muTarget = 0.26;
      appState.disturbanceMode = "none";
      elSelectDist.value = "none";
      if (appState.scaleL === 1000) {
        const matched = engine.matchOperatingPointForScale(450, 1.2, 5, 1000, bundle.p_rec);
        appState.rpm = matched.rpm;
        appState.vvm = matched.vvm;
        appState.manualFeedRate = 16.0;
      } else {
        appState.rpm = 450;
        appState.vvm = 1.2;
        appState.manualFeedRate = 0.08;
      }
      syncControlUI();
      runAndRenderSimulation();
    });

    function hydrateCaseStudy() {
      const s1 = bundle.stage1_recovery;
      const g86 = bundle.gam86;
      const hm = g86.heldout_metrics;
      const ctrlBench = bundle.benchmarks.control_strategies_5L.hybrid_closed_loop;
      const scaleBench = bundle.benchmarks.scale_1000L;

      document.getElementById("hero-bundle-tag").textContent =
        "GAM-86 | B=" + g86.n_bootstrap + " | SHA " + bundle.bundle_sha256.slice(0, 12);
      document.getElementById("hero-stage1-err").textContent =
        "max " + fmt(s1.max_relative_error_pct, 2) + "%";
      document.getElementById("hero-stage1-kappa").textContent =
        "14-seed mean " + fmt(s1.seed_14_summary.mean_err_pct, 2) + "% | kappa " + fmt(s1.condition_number_rel, 2);
      document.getElementById("hero-gam-r2").textContent =
        "mu " + fmt(hm.mu_r2, 4) + " | kLa " + fmt(hm.kla_r2, 4);
      const hybRatio = ctrlBench.hybrid_norm_rmse / Math.max(1e-9, ctrlBench.nominal_norm_rmse);
      document.getElementById("hero-ekf-ratio").textContent = fmt(hybRatio, 3) + "x";

      // Governance progression table rows
      const auditBody = document.getElementById("audit-progression-tbody");
      if (auditBody) {
        auditBody.innerHTML = bundle.audit_progression
          .map((item) => {
            const isGo = item.verdict.indexOf("PASS") === 0 || item.verdict.indexOf("RELEASE") === 0 || item.verdict.indexOf("GO") === 0;
            const markCls = isGo ? "mark-pass" : "mark-fail";
            return (
              "<tr>" +
              "<td><strong>" + escapeHtml(item.iteration) + "</strong></td>" +
              '<td><span class="' + markCls + '">' + escapeHtml(item.verdict) + "</span></td>" +
              '<td class="num">' + item.p0_count + "</td>" +
              '<td class="num">' + item.p1_count + "</td>" +
              "<td>" + escapeHtml(item.summary) + "</td>" +
              "</tr>"
            );
          })
          .join("");
      }

      document.getElementById("stage1-cond-summary").textContent =
        "kappa_rel = " + fmt(s1.condition_number_rel, 2) + " (< 45.0), max_err = " + fmt(s1.max_relative_error_pct, 2) + "% (< 8.0%)";
      const s1Body = document.querySelector("#stage1-params-table tbody");
      s1Body.innerHTML = s1.param_names
        .map((pname) => {
          const err = s1.relative_error_pct[pname];
          const dec = pname === "k_La0" ? 2 : 4;
          return (
            "<tr>" +
            '<td class="mono">' + escapeHtml(pname) + "</td>" +
            '<td class="num">' + fmt(s1.initial_guess[pname], dec) + "</td>" +
            '<td class="num">' + fmt(s1.true_values[pname], dec) + "</td>" +
            '<td class="num">' + fmt(s1.recovered_values[pname], dec) + "</td>" +
            '<td class="num">' + fmt(s1.standard_errors[pname], dec) + "</td>" +
            '<td class="num"><span class="mark-pass">' + fmt(err, 2) + "%</span></td>" +
            "</tr>"
          );
        })
        .join("");

      const s1Sum = s1.seed_14_summary;
      document.getElementById("stage1-14seed-summary").textContent =
        "14-Seed Distribution Summary: mean error = " +
        fmt(s1Sum.mean_err_pct, 2) +
        "%, worst max error = " +
        fmt(s1Sum.worst_max_err_pct, 2) +
        "% (limit < 8.0%), max kappa_rel = " +
        fmt(s1Sum.max_kappa_rel, 2) +
        " (limit < 45.0). Calibration SHA-256: " +
        s1.calibration_sha256.slice(0, 16) +
        "...";

      const scBody = document.querySelector("#scale-comparison-table tbody");
      const s1000 = scaleBench.summary;
      const scRows = [
        ["Matched Impeller Agitation (RPM)", "450.0", fmt(scaleBench.rpm_matched, 2)],
        ["Matched Aeration Rate (vvm)", "1.200", fmt(scaleBench.vvm_matched, 4)],
        ["Volumetric Gassed Power Pg/V (W/m3)", fmt(engine.computeHydrodynamics(450, 1.2, 2.0, bundle.p_rec).Pg_V, 2), fmt(scaleBench.Pg_V_matched, 2)],
        ["Superficial Gas Velocity v_s (m/s)", fmt(engine.computeHydrodynamics(450, 1.2, 2.0, bundle.p_rec).v_s, 5), fmt(scaleBench.v_s_matched, 5)],
        ["Impeller Tip Speed v_tip (m/s)", fmt(scaleBench.v_tip_5L, 3), fmt(scaleBench.v_tip_1000L, 3)],
        ["Closed-Loop Hybrid EKF nRMSE", fmt(ctrlBench.hybrid_norm_rmse, 4), fmt(s1000.hybrid_norm_rmse, 4)],
        ["Closed-Loop Nominal EKF nRMSE", fmt(ctrlBench.nominal_norm_rmse, 4), fmt(s1000.nominal_norm_rmse, 4)],
        ["Final Viable Biomass X_v (g/L)", fmt(ctrlBench.final_Xv_g_L, 2), fmt(s1000.final_Xv_g_L, 2)],
      ];
      scBody.innerHTML = scRows
        .map(
          (r) =>
            "<tr><td>" + escapeHtml(r[0]) + '</td><td class="num">' + escapeHtml(r[1]) + '</td><td class="num">' + escapeHtml(r[2]) + "</td></tr>"
        )
        .join("");

      const hoBody = document.querySelector("#heldout-regimes-table tbody");
      hoBody.innerHTML = g86.heldout_per_batch
        .map(
          (b) =>
            "<tr>" +
            '<td class="mono">' + escapeHtml(b.regime) + "</td>" +
            '<td class="num">' + fmt(b.kd_r2, 4) + "</td>" +
            '<td class="num">' + fmt(b.kla_r2, 4) + "</td>" +
            '<td class="num">' + fmt(b.qAnet_r2, 4) + "</td>" +
            '<td class="num">' + fmt(b.qAnet_rmse, 4) + "</td>" +
            '<td class="num">' + fmt(b.mu_r2, 4) + "</td>" +
            '<td class="num">' + fmt(b.hybrid_norm_rmse, 4) + "</td>" +
            '<td class="num">' + fmt(b.nominal_norm_rmse, 4) + "</td>" +
            "</tr>"
        )
        .join("");

      const distBody = document.querySelector("#disturbances-table tbody");
      distBody.innerHTML = bundle.benchmarks.disturbances_5L
        .map(
          (d) =>
            "<tr>" +
            '<td class="mono">' + escapeHtml(d.mode) + "</td>" +
            '<td class="num">' + fmt(d.hybrid_norm_rmse, 4) + "</td>" +
            '<td class="num">' + fmt(d.nominal_norm_rmse, 4) + "</td>" +
            '<td class="num"><span class="mark-pass">' + fmt(d.rmse_ratio, 3) + "x (-" + fmt(d.error_reduction_pct, 1) + "%)</span></td>" +
            '<td class="num">' + fmt(d.peak_acetate_g_L, 3) + "</td>" +
            '<td class="num">' + fmt(d.min_DO_pct, 1) + "%</td>" +
            '<td class="num">' + fmt(d.max_d_m, 2) + "</td>" +
            "</tr>"
        )
        .join("");

      const aicBody = document.querySelector("#arch-aic-table tbody");
      aicBody.innerHTML = g86.arch_aic_bic
        .map((r) => {
          const badge = r.selected
            ? '<span class="mark-pass">Selected (GAM-86, Conv.)</span>'
            : '<span class="mono">' + (r.converged ? "Candidate (Conv.)" : "Cap (500 it.)") + "</span>";
          return (
            "<tr>" +
            '<td class="mono">' + escapeHtml(r.cadence) + "</td>" +
            '<td class="num">(' + r.h_ace + ", " + r.h_mu + ")</td>" +
            '<td class="num">' + r.n_params + "</td>" +
            '<td class="num">' + fmt(r.rss, 1) + "</td>" +
            '<td class="num">' + fmt(r.aic, 1) + "</td>" +
            '<td class="num">' + fmt(r.bic, 1) + "</td>" +
            '<td class="num">' + badge + "</td>" +
            "</tr>"
          );
        })
        .join("");

      const hydRef = engine.computeHydrodynamics(450, 1.2, 2.0, bundle.p_rec);
      const muM = bundle.p_rec.mu_max * (0.3 / (bundle.p_rec.K_S + 0.3)) * (4.5 / (bundle.p_rec.K_O2 + 4.5));
      const p84 = engine.predictGamWithUQ(80.0, 4.0, 0.3, 4.5, 0.2, hydRef.v_tip, muM, bundle);
      const trueEta84 = g86.reference_84gL_true_eta_kla;
      const z84 = Math.abs(p84.eta_kla - trueEta84) / Math.max(1e-9, p84.std_kla);
      const pOod = engine.predictGamWithUQ(10.0, 0.3, 15.0, 4.0, 8.0, hydRef.v_tip, 0.3, bundle);
      document.getElementById("ood-calibration-summary").textContent =
        "Live JS Parity Verification: At X_tot=84 g/L, eta_kla=" +
        fmt(p84.eta_kla, 4) +
        " +- " +
        fmt(p84.std_kla, 4) +
        " (true " +
        fmt(trueEta84, 4) +
        ", z=" +
        fmt(z84, 2) +
        " <= 2.5, in_domain=" +
        p84.in_domain +
        "). At extreme OOD (S=15, A=8), eta_mu relaxes to " +
        fmt(pOod.eta_mu, 4) +
        " (mechanistic prior 1.0000).";

      const pyBody = document.querySelector("#pytest-matrix-table tbody");
      pyBody.innerHTML = bundle.pytest_suite
        .map(
          (t) =>
            "<tr>" +
            '<td class="mono" style="font-size:12px;">' + escapeHtml(t.name).replace(/_/g, "_<wbr>") + "</td>" +
            '<td><span class="mark-pass">' + escapeHtml(t.status) + "</span></td>" +
            "<td>" + escapeHtml(t.focus) + "</td>" +
            "</tr>"
        )
        .join("");

      const al = bundle.alcoa_sample;
      document.getElementById("alcoa-batch-id").textContent = al.batch_id;
      document.getElementById("alcoa-count-badge").textContent = al.audit_log_count + " SHA-256 events";
      document.getElementById("alcoa-digests-box").innerHTML =
        "standard:      " + escapeHtml(al.alcoa_standard) + "<br />" +
        "params_sha256: " + escapeHtml(al.params_sha256) + "<br />" +
        "model_sha256:  " + escapeHtml(al.model_sha256) + "<br />" +
        "record_sha256: " + escapeHtml(al.record_sha256);

      const alBody = document.querySelector("#alcoa-events-table tbody");
      alBody.innerHTML = (al.audit_log_preview || [])
        .map(
          (ev) =>
            "<tr>" +
            '<td class="num">' + ev.seq + "</td>" +
            '<td class="num">' + fmt(ev.process_t_h, 2) + "</td>" +
            '<td class="mono">' + escapeHtml(ev.event_type) + "</td>" +
            '<td class="mono">' + escapeHtml(ev.entry_sha256) + "</td>" +
            "</tr>"
        )
        .join("");
    }

    // ARIA tabs with keyboard arrow support (A11Y-PATTERNS.md)
    const tabBtns = Array.from(document.querySelectorAll(".tab-btn"));
    function activateTab(btn) {
      const target = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => {
        const active = b.getAttribute("data-tab") === target;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
        b.setAttribute("tabindex", active ? "0" : "-1");
      });
      document.querySelectorAll(".tab-panel").forEach((p) => {
        const show = p.id === "tab-" + target;
        p.classList.toggle("active", show);
        p.hidden = !show;
      });
    }

    tabBtns.forEach((btn, idx) => {
      btn.addEventListener("click", () => activateTab(btn));
      btn.addEventListener("keydown", function (e) {
        let nextIdx = null;
        if (e.key === "ArrowRight") nextIdx = (idx + 1) % tabBtns.length;
        else if (e.key === "ArrowLeft") nextIdx = (idx - 1 + tabBtns.length) % tabBtns.length;
        else if (e.key === "Home") nextIdx = 0;
        else if (e.key === "End") nextIdx = tabBtns.length - 1;
        if (nextIdx != null) {
          e.preventDefault();
          activateTab(tabBtns[nextIdx]);
          tabBtns[nextIdx].focus();
        }
      });
    });

    function triggerDownload(filename, content, mimeType) {
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    document.getElementById("download-sim-csv-btn").addEventListener("click", function () {
      const sim = appState.sim;
      if (!sim) return;
      const header = [
        "t_h", "Xv_est", "Xd_est", "S_est", "CL_est", "A_est", "P_est", "V_est",
        "Xv_std", "F_cmd", "rpm", "vvm", "mu_est", "kLa_est", "d_m", "in_domain",
      ];
      if (appState.showTruth) {
        header.push("Xv_true", "Xd_true", "S_true", "CL_true", "A_true", "P_true");
      }
      const lines = [header.join(",")];
      for (let i = 0; i < sim.t.length; i++) {
        const row = [
          sim.t[i],
          sim.estHyb[i][0], sim.estHyb[i][1], sim.estHyb[i][2], sim.estHyb[i][3],
          sim.estHyb[i][4], sim.estHyb[i][5], sim.estHyb[i][6],
          sim.stdHyb[i][0],
          sim.inputs[i][0], sim.inputs[i][1], sim.inputs[i][2],
          sim.muHyb[i], sim.klaHyb[i], sim.d_m[i], sim.inDomain[i],
        ];
        if (appState.showTruth) {
          row.push(
            sim.trueStates[i][0], sim.trueStates[i][1], sim.trueStates[i][2],
            sim.trueStates[i][3], sim.trueStates[i][4], sim.trueStates[i][5]
          );
        }
        lines.push(row.map((v) => (typeof v === "number" ? v.toFixed(6) : v)).join(","));
      }
      triggerDownload(
        "hybrid_twin_" + appState.scaleL + "L_" + appState.disturbanceMode + ".csv",
        lines.join("\n"),
        "text/csv"
      );
    });

    document.getElementById("download-alcoa-json-btn").addEventListener("click", function () {
      triggerDownload(
        "alcoa_audit_record_gam86.json",
        JSON.stringify(bundle.alcoa_sample, null, 2),
        "application/json"
      );
    });

    // =========================================================================
    // Distill.pub Monograph Interactive Chapters (Sections 1 to 4 + TOC + KaTeX)
    // =========================================================================
    const monographState = {
      dilemmaMode: "optimal",
      dilemmaFeed: 0.07,
      dilemmaPhase: 0.35,
      dilemmaVisible: false,
      dilemmaSimOpt: null,
      dilemmaSimOvf: null,
      dilemmaLastTs: 0,
      sec2Zone: "growth",
      patankarDt: 0.24,
      auditStep: 2,
      auditCompare: "both",
      auditSim: null,
      scaleMode: "matched",
      scaleXtot: 25,
      scaleVtip: 2.125,
      scaleEnsCache: null,
    };

    const reducedMotion =
      global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function initTocScrollSpy() {
      const links = Array.from(document.querySelectorAll(".toc-link[data-toc]"));
      if (!links.length) return;
      const sectionIds = links.map((a) => a.getAttribute("data-toc"));
      const sections = sectionIds.map((id) => document.getElementById(id)).filter(Boolean);
      if (!sections.length) return;

      function setActiveToc(activeId) {
        links.forEach((a) => {
          const hit = a.getAttribute("data-toc") === activeId;
          a.classList.toggle("active", hit);
          if (hit) a.setAttribute("aria-current", "location");
          else a.removeAttribute("aria-current");
        });
      }

      if (global.IntersectionObserver) {
        const obs = new IntersectionObserver(
          (entries) => {
            const visible = entries
              .filter((e) => e.isIntersecting)
              .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
            if (visible.length) {
              setActiveToc(visible[0].target.id);
            }
          },
          { rootMargin: "-18% 0px -64% 0px", threshold: [0.02, 0.2] }
        );
        sections.forEach((sec) => obs.observe(sec));
      }

      let scrollTicking = false;
      global.addEventListener(
        "scroll",
        function () {
          if (scrollTicking) return;
          scrollTicking = true;
          global.requestAnimationFrame(function () {
            scrollTicking = false;
            const probeY = 160;
            let bestId = sections[0].id;
            for (const sec of sections) {
              const rect = sec.getBoundingClientRect();
              if (rect.top <= probeY) bestId = sec.id;
            }
            setActiveToc(bestId);
          });
        },
        { passive: true }
      );
    }

    // ---- Section 1: Metabolic Overflow & Oxygen Dilemma ----
    function recomputeDilemma() {
      if (!monographState.dilemmaSimOpt) {
        monographState.dilemmaSimOpt = engine.runTwinSimulation(
          {
            scaleL: 5,
            controlMode: "closed_loop",
            muTarget: 0.26,
            manualFeedRate: 0.08,
            rpm: 450,
            vvm: 1.2,
            disturbanceMode: "none",
          },
          bundle
        );
      }
      monographState.dilemmaSimOvf = engine.runTwinSimulation(
        {
          scaleL: 5,
          controlMode: "manual",
          muTarget: 0.26,
          manualFeedRate: monographState.dilemmaFeed,
          rpm: 450,
          vvm: 1.2,
          disturbanceMode: "none",
        },
        bundle
      );

      const isOpt = monographState.dilemmaMode === "optimal";
      const activeSim = isOpt ? monographState.dilemmaSimOpt : monographState.dilemmaSimOvf;
      const cStar = activeSim.C_star || 8.2;
      let peakAce = 0;
      let minDoPct = 100;
      for (let k = 0; k < activeSim.trueStates.length; k++) {
        const st = activeSim.trueStates[k];
        if (st[4] > peakAce) peakAce = st[4];
        const doPct = (100 * st[3]) / cStar;
        if (doPct < minDoPct) minDoPct = doPct;
      }
      const finalXv = activeSim.trueStates[activeSim.trueStates.length - 1][0];

      setTextById("val-dilemma-feed", fmt(monographState.dilemmaFeed, 2) + " L/h");
      setTextById("dilemma-kpi-ace", fmt(peakAce, 2) + " g/L");
      setTextById("dilemma-kpi-do", fmt(minDoPct, 1) + "%");
      setTextById("dilemma-kpi-xv", fmt(finalXv, 1) + " g/L");

      if (isOpt) {
        setTextById("dilemma-regime-badge", "Regime: Optimal Closed-Loop (DO & Acetate Guarded)");
      } else {
        const tag = minDoPct < 20 || peakAce > 1.0 ? "Overflow + Hypoxia Fault" : "Manual Constant Feed";
        setTextById("dilemma-regime-badge", "Regime: " + tag + " (F = " + fmt(monographState.dilemmaFeed, 2) + " L/h)");
      }

      const optBtns = [
        document.getElementById("btn-dilemma-optimal"),
        document.getElementById("seg-dilemma-optimal"),
      ];
      const ovfBtns = [
        document.getElementById("btn-dilemma-overflow"),
        document.getElementById("seg-dilemma-overflow"),
      ];
      optBtns.forEach((b) => {
        if (b) {
          b.classList.toggle("active", isOpt);
          b.setAttribute("aria-pressed", isOpt ? "true" : "false");
        }
      });
      ovfBtns.forEach((b) => {
        if (b) {
          b.classList.toggle("active", !isOpt);
          b.setAttribute("aria-pressed", !isOpt ? "true" : "false");
        }
      });
      drawDilemmaCanvas();
    }

    function drawDilemmaCanvas() {
      const canvas = document.getElementById("canvas-dilemma");
      const optSim = monographState.dilemmaSimOpt;
      const ovfSim = monographState.dilemmaSimOvf;
      if (!canvas || !optSim || !ovfSim) return;

      const C = getThemeColors();
      const { ctx, w, h } = setupHiDpiCanvas(canvas);
      ctx.clearRect(0, 0, w, h);

      const isOpt = monographState.dilemmaMode === "optimal";
      const activeSim = isOpt ? optSim : ovfSim;
      const phase = clampVal(monographState.dilemmaPhase, 0, 1);
      const kIdx = Math.min(activeSim.t.length - 1, Math.floor(phase * (activeSim.t.length - 1)));
      const st = activeSim.trueStates[kIdx];
      const cStar = activeSim.C_star || 8.2;
      const curDoPct = clampVal((100 * st[3]) / cStar, 0, 105);
      const curAce = st[4];
      const xTot = st[0] + st[1];
      const etaKla = activeSim.klaHyb[kIdx] / Math.max(1e-6, activeSim.klaNom[kIdx]);

      const stacked = w < 560;
      const splitX = stacked ? w : Math.round(w * 0.42);
      const diagW = splitX;
      const diagH = stacked ? Math.round(h * 0.44) : h;

      // Left Panel: Live Metabolic Pathway + Viscosity-Damped Oxygen Bubbles
      ctx.fillStyle = C.panel;
      ctx.fillRect(0, 0, diagW, diagH);
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      if (stacked) {
        ctx.beginPath();
        ctx.moveTo(0, diagH + 0.5);
        ctx.lineTo(w, diagH + 0.5);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.moveTo(diagW + 0.5, 0);
        ctx.lineTo(diagW + 0.5, h);
        ctx.stroke();
      }

      const padD = 14;
      const boxW = diagW - padD * 2;
      const boxH = diagH - padD * 2;
      const cxCell = padD + boxW * 0.52;
      const cyCell = padD + boxH * 0.44;
      const rxCell = clampVal(boxW * 0.34, 48, 128);
      const ryCell = clampVal(boxH * 0.25, 26, 62);

      // Sparger rising oxygen bubbles behind cell
      const nBubbles = 12;
      const bubbleColor = curDoPct < 22 ? C.alert : C.gt;
      ctx.save();
      ctx.globalAlpha = 0.38;
      ctx.fillStyle = bubbleColor;
      for (let b = 0; b < nBubbles; b++) {
        const bx = padD + ((b + 0.6) / nBubbles) * boxW;
        const speed = 0.35 + 0.65 * clampVal(etaKla, 0.25, 1.0);
        const rawY = 1.0 - ((phase * speed * 2.2 + b * 0.23) % 1.0);
        const by = padD + rawY * boxH;
        const rad = curDoPct < 22 ? 2.2 : 3.4;
        ctx.beginPath();
        ctx.arc(bx, by, rad, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Header text in left diagram
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("INTRACELLULAR FLUX (t = " + fmt(activeSim.t[kIdx], 1) + " h)", padD, padD + 10);

      // Cell envelope
      ctx.fillStyle = C.bg;
      ctx.strokeStyle = isOpt ? C.gtInk : curAce > 0.8 || curDoPct < 22 ? C.alertInk : C.hybInk;
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(cxCell - rxCell, cyCell - ryCell, rxCell * 2, ryCell * 2, ryCell);
      } else {
        ctx.rect(cxCell - rxCell, cyCell - ryCell, rxCell * 2, ryCell * 2);
      }
      ctx.fill();
      ctx.stroke();

      // Glucose -> Pyruvate -> TCA / Overflow Arrows
      const inX = cxCell - rxCell - 22;
      const pyrX = cxCell - rxCell * 0.32;
      const tcaX = cxCell + rxCell * 0.38;
      const tcaY = cyCell - ryCell * 0.36;
      const aceY = cyCell + ryCell * 0.44;
      const overflowFrac = clampVal(curAce / 2.2 + (st[2] > 0.35 ? 0.45 : 0.05), 0.08, 0.95);

      ctx.lineCap = "round";
      // Feed inlet arrow
      ctx.strokeStyle = st[2] > 0.35 ? C.alert : C.gt;
      ctx.lineWidth = st[2] > 0.35 ? 3.5 : 2.2;
      ctx.beginPath();
      ctx.moveTo(Math.max(padD, inX), cyCell);
      ctx.lineTo(pyrX - 6, cyCell);
      ctx.stroke();

      // TCA biomass branch
      ctx.strokeStyle = C.gt;
      ctx.lineWidth = 3.2 - 1.6 * overflowFrac;
      ctx.beginPath();
      ctx.moveTo(pyrX + 6, cyCell);
      ctx.lineTo(tcaX, tcaY);
      ctx.stroke();

      // Overflow Acetate branch
      ctx.strokeStyle = overflowFrac > 0.42 ? C.alert : C.hyb;
      ctx.lineWidth = 1.2 + 3.2 * overflowFrac;
      ctx.beginPath();
      ctx.moveTo(pyrX + 6, cyCell);
      ctx.lineTo(tcaX, aceY);
      ctx.stroke();

      // Animated flux pulse dots
      const pulse = (phase * 5) % 1;
      ctx.fillStyle = C.gt;
      ctx.beginPath();
      ctx.arc(pyrX + 6 + (tcaX - pyrX - 6) * pulse, cyCell + (tcaY - cyCell) * pulse, 3.2, 0, Math.PI * 2);
      ctx.fill();
      if (overflowFrac > 0.28) {
        ctx.fillStyle = C.alert;
        ctx.beginPath();
        ctx.arc(pyrX + 6 + (tcaX - pyrX - 6) * pulse, cyCell + (aceY - cyCell) * pulse, 3.6, 0, Math.PI * 2);
        ctx.fill();
      }

      // Node labels inside cell
      ctx.font = '600 10.5px "IBM Plex Sans", sans-serif';
      ctx.fillStyle = C.ink;
      ctx.textAlign = "right";
      ctx.fillText("S=" + fmt(st[2], 2), Math.max(padD + 44, cxCell - rxCell - 6), cyCell - 12);
      ctx.textAlign = "center";
      ctx.fillText("Pyr", pyrX, cyCell - 8);
      ctx.fillStyle = C.gtInk;
      ctx.fillText("TCA -> X_v", tcaX + 4, tcaY - 5);
      ctx.fillStyle = overflowFrac > 0.42 ? C.alertInk : C.hybInk;
      ctx.fillText("Ace " + fmt(curAce, 2) + " g/L", tcaX + 4, aceY + 13);

      // Footer broth state pill inside diagram
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.textAlign = "left";
      ctx.fillStyle = curDoPct < 20 ? C.alertInk : C.ink;
      ctx.fillText(
        "DO=" + fmt(curDoPct, 0) + "% | X_tot=" + fmt(xTot, 1) + " g/L | eta_kLa=" + fmt(etaKla, 2),
        padD,
        diagH - 12
      );

      // Right Panel: 16h Trajectories (DO% + Acetate)
      const chartX = stacked ? 0 : splitX;
      const chartY = stacked ? diagH : 0;
      const chartW = stacked ? w : w - splitX;
      const chartH = stacked ? h - diagH : h;
      const pad = { l: 46, r: 44, t: 24, b: 28 };
      const pw = Math.max(60, chartW - pad.l - pad.r);
      const ph = Math.max(60, chartH - pad.t - pad.b);
      const tMax = optSim.t[optSim.t.length - 1] || 16;

      let maxAce = 1.6;
      for (const s of [optSim, ovfSim]) {
        for (let i = 0; i < s.trueStates.length; i++) {
          if (s.trueStates[i][4] * 1.15 > maxAce) maxAce = s.trueStates[i][4] * 1.15;
        }
      }

      const px = (t) => chartX + pad.l + (t / tMax) * pw;
      const pyDo = (pct) => chartY + pad.t + ph * (1 - clampVal(pct, 0, 105) / 105);
      const pyAce = (a) => chartY + pad.t + ph * (1 - clampVal(a, 0, maxAce) / maxAce);

      // Grid lines
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const y = Math.round(chartY + pad.t + (ph * g) / 4) + 0.5;
        ctx.beginPath();
        ctx.moveTo(chartX + pad.l, y);
        ctx.lineTo(chartX + chartW - pad.r, y);
        ctx.stroke();

        ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.fillStyle = C.muted;
        ctx.textAlign = "right";
        ctx.fillText(String(Math.round(105 * (1 - g / 4))) + "%", chartX + pad.l - 5, y + 3);
        ctx.textAlign = "left";
        ctx.fillText(fmt(maxAce * (1 - g / 4), 1), chartX + chartW - pad.r + 5, y + 3);
      }

      // Hypoxia floor at 20% DO
      const yFloor = pyDo(20);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = C.alert;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(chartX + pad.l, yFloor);
      ctx.lineTo(chartX + chartW - pad.r, yFloor);
      ctx.stroke();
      ctx.restore();
      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.alertInk;
      ctx.textAlign = "left";
      ctx.fillText("20% DO Floor", chartX + pad.l + 4, yFloor - 4);

      function traceSeries(simObj, yFn, valFn, color, width, dashed, alpha) {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (let k = 0; k < simObj.t.length; k++) {
          const x = px(simObj.t[k]);
          const y = yFn(valFn(simObj.trueStates[k], simObj.C_star));
          if (k === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }

      // Comparison curves + active curve
      traceSeries(optSim, pyDo, (s, cs) => (100 * s[3]) / cs, C.gt, isOpt ? 2.4 : 1.4, !isOpt, isOpt ? 1 : 0.45);
      traceSeries(ovfSim, pyDo, (s, cs) => (100 * s[3]) / cs, C.alert, isOpt ? 1.3 : 2.5, isOpt, isOpt ? 0.35 : 0.95);
      traceSeries(ovfSim, pyAce, (s) => s[4], C.hyb, isOpt ? 1.2 : 2.3, true, isOpt ? 0.3 : 0.95);
      if (isOpt) {
        traceSeries(optSim, pyAce, (s) => s[4], C.hybInk, 1.8, true, 0.85);
      }

      // Playhead cursor
      const cx = px(activeSim.t[kIdx]);
      ctx.strokeStyle = C.borderStrong;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx, chartY + pad.t);
      ctx.lineTo(cx, chartY + pad.t + ph);
      ctx.stroke();

      // Axis titles & legend
      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.gtInk;
      ctx.textAlign = "left";
      ctx.fillText("DO (% sat, solid)", chartX + pad.l, chartY + 14);
      ctx.fillStyle = C.hybInk;
      ctx.textAlign = "right";
      ctx.fillText("Acetate (g/L, dashed)", chartX + chartW - pad.r, chartY + 14);
      ctx.fillStyle = C.muted;
      ctx.textAlign = "center";
      ctx.fillText("Batch Time (0 to 16 h)", chartX + pad.l + pw * 0.5, chartY + chartH - 8);
    }

    function stepDilemmaAnim(ts) {
      if (monographState.dilemmaVisible && !reducedMotion) {
        if (!monographState.dilemmaLastTs) monographState.dilemmaLastTs = ts;
        const dt = Math.min(0.05, (ts - monographState.dilemmaLastTs) / 1000);
        monographState.dilemmaLastTs = ts;
        monographState.dilemmaPhase = (monographState.dilemmaPhase + dt * 0.11) % 1.0;
        drawDilemmaCanvas();
        global.requestAnimationFrame(stepDilemmaAnim);
      } else {
        monographState.dilemmaLastTs = 0;
      }
    }

    function bindSection1Dilemma() {
      const setMode = (mode) => {
        monographState.dilemmaMode = mode;
        recomputeDilemma();
      };
      ["btn-dilemma-optimal", "seg-dilemma-optimal"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("click", () => setMode("optimal"));
      });
      ["btn-dilemma-overflow", "seg-dilemma-overflow"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("click", () => setMode("overflow"));
      });
      const sl = document.getElementById("slider-dilemma-feed");
      if (sl) {
        sl.addEventListener("input", function () {
          monographState.dilemmaFeed = Number(this.value);
          monographState.dilemmaMode = "overflow";
          recomputeDilemma();
        });
      }
      const plate = document.getElementById("fig-dilemma-plate");
      if (plate && global.IntersectionObserver && !reducedMotion) {
        const obs = new IntersectionObserver(
          (entries) => {
            const vis = entries.some((e) => e.isIntersecting);
            if (vis && !monographState.dilemmaVisible) {
              monographState.dilemmaVisible = true;
              global.requestAnimationFrame(stepDilemmaAnim);
            } else if (!vis) {
              monographState.dilemmaVisible = false;
            }
          },
          { threshold: 0.05 }
        );
        obs.observe(plate);
      }
      recomputeDilemma();
    }

    // ---- Section 2: Equation Explorer + 3D Zone Highlighter + Patankar Demo ----
    const SEC2_ZONE_SPECS = {
      all: {
        badge: "Active Zone: Full Vessel Assembly",
        title: "Full Serial Hybrid Mass Balance (7-State Coupled Bioreactor)",
        formula: "\\frac{dX_v}{dt} = (\\mu - k_d - D)\\,X_v",
        desc: "Hover or click any equation term above to aim the camera and spotlight that physical sub-assembly inside the vessel.",
      },
      growth: {
        badge: "Active Zone: Broth Bulk & Cellular Metabolism",
        title: "Specific Growth Rate & Overflow Efficiency",
        formula: "\\mu = \\mu_{\\text{mech}}(S, C_L)\\cdot \\eta_\\mu(S, A)",
        desc: "Mechanistic Monod substrate and oxygen affinity multiplied by a bounded neural correction inside (0.05, 1.05) that captures overflow inhibition when glucose or acetate rises.",
      },
      sparger: {
        badge: "Active Zone: Sintered Sparger Ring & Plume",
        title: "Oxygen Mass Transfer & Broth Viscosity Attenuation",
        formula: "k_L a = k_{L}a_{\\text{nom}}(P_g/V, v_s)\\cdot \\eta_{k_L a}(X_{\\text{tot}})",
        desc: "Rising sparger gas bubbles encounter progressively thicker cell suspension as biomass accumulates, dampening mass transfer toward 0.40x near 25 g/L and below 0.10x above 80 g/L.",
      },
      impeller: {
        badge: "Active Zone: Dual Rushton Impellers (Tip Shear)",
        title: "Hydrodynamic Shear & Cellular Stress Lysis",
        formula: "k_d = k_{d,0} + \\Delta k_d(S, C_L, A, v_{\\text{tip}})",
        desc: "High impeller tip speed (v_tip = pi N D_i) inflicts hydrodynamic shear lysis at the turbine blade tips, especially when cells are concurrently starved or acetate-stressed.",
      },
      feed: {
        badge: "Active Zone: Headplate Sterile Glucose Feed",
        title: "Fed-Batch Substrate Addition & Volume Dilution",
        formula: "D = \\frac{F}{V}, \\qquad \\frac{dV}{dt} = F",
        desc: "Concentrated sterile glucose (300 g/L) enters from the headplate port. Growing liquid volume V dilutes every soluble metabolite and viable cell concentration equally.",
      },
      patankar: {
        badge: "Active Zone: Exponential-Rosenbrock Positive Lock",
        title: "Exact Integrating Factor for Stiff Oxygen & Decay",
        formula: "\\phi_1(z) = \\frac{1 - e^{-z}}{z}, \\qquad x_{n+1} > 0",
        desc: "Replaces unstable explicit linear steps with exact exponential decay fractions so stiff oxygen transfer (k_La > 400 1/h) and low glucose concentrations stay strictly non-negative.",
      },
    };

    function setSection2Zone(zone) {
      const z = SEC2_ZONE_SPECS[zone] ? zone : "all";
      monographState.sec2Zone = z;
      const spec = SEC2_ZONE_SPECS[z];

      setTextById("sec2-zone-badge", spec.badge);
      document.querySelectorAll(".eq-term-btn[data-zone]").forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("data-zone") === z);
      });

      const detail = document.getElementById("eq-zone-detail");
      if (detail) {
        detail.innerHTML =
          '<div class="eq-detail-title">' +
          escapeHtml(spec.title) +
          ' (<span data-katex="' +
          escapeHtml(spec.formula) +
          '"></span>)</div>' +
          "<p>" +
          escapeHtml(spec.desc) +
          "</p>";
        renderKaTeXMath(detail);
      }

      if (appState.sec2Reactor3D && typeof appState.sec2Reactor3D.setHighlightZone === "function") {
        appState.sec2Reactor3D.setHighlightZone(z);
      }
    }

    function drawPatankarDemo() {
      const canvas = document.getElementById("canvas-patankar");
      if (!canvas) return;
      const C = getThemeColors();
      const { ctx, w, h } = setupHiDpiCanvas(canvas);
      ctx.clearRect(0, 0, w, h);

      const dt = clampVal(monographState.patankarDt, 0.02, 0.5);
      const tEnd = 1.4;
      const kSink = 6.5;
      const prod = 0.08;
      const s0 = 1.2;

      const tPts = [0];
      const eulerPts = [s0];
      const patPts = [s0];
      let t = 0;
      let sE = s0;
      let sP = s0;
      let minEuler = s0;
      let minPat = s0;
      const steps = Math.min(200, Math.ceil(tEnd / dt));
      for (let k = 0; k < steps; k++) {
        t += dt;
        const nextEuler = sE + dt * (prod - kSink * sE);
        const z = kSink * dt;
        const p1 = z < 1e-6 ? 1 - 0.5 * z : (1 - Math.exp(-z)) / z;
        const nextPat = Math.max(0, sP + dt * p1 * (prod - kSink * sP));
        sE = clampVal(nextEuler, -1.6, 2.0);
        sP = nextPat;
        if (nextEuler < minEuler) minEuler = nextEuler;
        if (nextPat < minPat) minPat = nextPat;
        tPts.push(t);
        eulerPts.push(sE);
        patPts.push(sP);
      }

      setTextById("val-patankar-dt", fmt(dt, 2) + " h");
      setTextById(
        "patankar-euler-min",
        minEuler < 0
          ? "UNPHYSICAL CRASH at dt=" + fmt(dt, 2) + " h: Explicit Euler min S = " + fmt(minEuler, 3) + " g/L (< 0) vs Patankar min S = +" + fmt(minPat, 4) + " g/L"
          : "Below stability limit dt <= 0.154 h: Euler min S = +" + fmt(minEuler, 4) + " g/L | Patankar min S = +" + fmt(minPat, 4) + " g/L"
      );

      const pad = { l: 44, r: 16, t: 18, b: 24 };
      const pw = Math.max(60, w - pad.l - pad.r);
      const ph = Math.max(50, h - pad.t - pad.b);
      const yMin = Math.min(-0.6, minEuler * 1.12);
      const yMax = 1.32;
      const px = (tv) => pad.l + (tv / Math.max(tEnd, tPts[tPts.length - 1])) * pw;
      const py = (sv) => pad.t + ph * (1 - (clampVal(sv, yMin, yMax) - yMin) / (yMax - yMin));
      const yZero = py(0);

      // Shaded unphysical negative hazard region
      ctx.fillStyle = C.alertWash;
      ctx.fillRect(pad.l, yZero, pw, pad.t + ph - yZero);
      ctx.strokeStyle = C.alert;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, Math.round(yZero) + 0.5);
      ctx.lineTo(w - pad.r, Math.round(yZero) + 0.5);
      ctx.stroke();

      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.alertInk;
      ctx.textAlign = "right";
      ctx.fillText("UNPHYSICAL NEGATIVE ZONE (S < 0)", w - pad.r - 4, Math.min(h - 6, yZero + 13));

      // Zero and max axis ticks
      ctx.fillStyle = C.muted;
      ctx.textAlign = "right";
      ctx.fillText("1.2", pad.l - 5, py(1.2) + 3);
      ctx.fillText("0.0", pad.l - 5, yZero + 3);
      if (yMin < -0.2) ctx.fillText(fmt(yMin, 1), pad.l - 5, pad.t + ph);

      // Exact solution baseline
      ctx.strokeStyle = C.borderStrong;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let i = 0; i <= 50; i++) {
        const ti = (tEnd * i) / 50;
        const sEx = prod / kSink + (s0 - prod / kSink) * Math.exp(-kSink * ti);
        if (i === 0) ctx.moveTo(px(ti), py(sEx));
        else ctx.lineTo(px(ti), py(sEx));
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Explicit Euler trace
      ctx.strokeStyle = C.alert;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < tPts.length; i++) {
        if (i === 0) ctx.moveTo(px(tPts[i]), py(eulerPts[i]));
        else ctx.lineTo(px(tPts[i]), py(eulerPts[i]));
      }
      ctx.stroke();

      // Patankar phi1 trace
      ctx.strokeStyle = C.gt;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let i = 0; i < tPts.length; i++) {
        if (i === 0) ctx.moveTo(px(tPts[i]), py(patPts[i]));
        else ctx.lineTo(px(tPts[i]), py(patPts[i]));
      }
      ctx.stroke();

      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.textAlign = "left";
      ctx.fillStyle = C.gtInk;
      ctx.fillText("Patankar phi_1 (positive lock)", pad.l + 6, 12);
      ctx.fillStyle = C.alertInk;
      ctx.textAlign = "right";
      ctx.fillText("Explicit Euler (step dt=" + fmt(dt, 2) + " h)", w - pad.r, 12);
    }

    function bindSection2Explorer() {
      document.querySelectorAll(".eq-term-btn[data-zone]").forEach((btn) => {
        const z = btn.getAttribute("data-zone");
        btn.addEventListener("mouseenter", () => setSection2Zone(z));
        btn.addEventListener("focus", () => setSection2Zone(z));
        btn.addEventListener("click", () => setSection2Zone(z));
      });
      const resetBtn = document.getElementById("sec2-reset-zone-btn");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => setSection2Zone("all"));
      }
      const slDt = document.getElementById("slider-patankar-dt");
      if (slDt) {
        slDt.addEventListener("input", function () {
          monographState.patankarDt = Number(this.value);
          drawPatankarDemo();
        });
      }
      setSection2Zone("growth");
      drawPatankarDemo();
    }

    // ---- Section 3: Literature-Grounded Verification Stepper & Trajectory Compare ----
    const AUDIT_STEP_DIFFS = [
      {
        flawTitle: "Iteration 0 Flaws: Sensor Leakage & Volume Lock-In (P0 x3, P1 x5)",
        flawHtml:
          "<strong>1. Dead-Cell Sensor Leakage:</strong> Assumed offline instruments measure dead cells (<span data-katex=\"X_d\">X_d</span>) directly inside EKF updates.<br />" +
          "<strong>2. Extensive Inputs:</strong> Passed reactor volume (<span data-katex=\"V\">V</span>) to the neural network, blocking 5 L to 1000 L transfer.<br />" +
          "<strong>3. Negative Step Risk:</strong> Fixed-step explicit solver could undershoot glucose below zero.",
        fixTitle: "Governance Mandate: Physical Realism & Observability Lock",
        fixHtml:
          "Strip <span data-katex=\"X_d\">X_d</span> from the sensor observation matrix <span data-katex=\"H\">H</span>, enforce intensive fluid physics (<span data-katex=\"P_g/V, v_s, v_{\\text{tip}}\">P_g/V, v_s, v_tip</span>), and lock non-negativity with Patankar-Rosenbrock <span data-katex=\"\\phi_1(z)\">phi_1(z)</span>.",
      },
      {
        flawTitle: "Iteration 1 Flaws: Parameter Collinearity & Black-Box Creep (P0 x2, P1 x6)",
        flawHtml:
          "<strong>1. Growth/Death Collinearity:</strong> Single-feed calibration failed to decouple <span data-katex=\"\\mu\">mu</span> from lysis <span data-katex=\"k_d\">k_d</span> (<span data-katex=\"\\kappa_{\\text{rel}} > 45\">kappa_rel &gt; 45</span>).<br />" +
          "<strong>2. No Extrapolation Guard:</strong> Unpruned neural ensemble extrapolated unpredictably outside training broth densities.",
        fixTitle: "Governance Mandate: iDoE Excitation & Mahalanobis Guard",
        fixHtml:
          "Deploy 3-batch iDoE calibration reaching <span data-katex=\"\\kappa_{\\text{rel}} = 38.18\">kappa_rel = 38.18</span> (seed 42; 14-seed max &le; 38.18), select GAM-86 via 0.2 h convergence &amp; 5-regime gate, and add smooth <span data-katex=\"D_M\">D_M</span> relaxation toward a power-law prior past <span data-katex=\"p_{99} = 5.01\">p99 = 5.01</span>.",
      },
      {
        flawTitle: "Iteration 2 Verified Architecture: Portela Level 1 Digital Model / Level 2-Ready EKF (0 P0, 0 P1)",
        flawHtml:
          "All P0/P1 structural defects resolved. Offline Stage-2 computes exact discrete reverse-mode VJP gradients (<code>max_rel = 6.92e-7</code>) with zero <span data-katex=\"X_d\">X_d</span> supervision; runtime EKF propagates central finite-difference Jacobians with unclipped Pad&eacute;(6,6) matrix exponentials.",
        fixTitle: "8/8 Contract Suite & Multi-Regime Error Reduction Proof",
        fixHtml:
          "Held-out pooled <span data-katex=\"R^2(\\mu) = 0.9951\">0.9951</span>, <span data-katex=\"R^2(k_L a) = 0.9699\">0.9699</span>. Live 5 L Air-Fault Drop: <strong>-69.2% (0.308x)</strong> | 5 L Benchmark (-55% vvm): <strong>-59.1%</strong> | Held-Out 5-Regime: <strong>-85.8%</strong> | 1000 L: <strong>-83.8%</strong> (<code>0.0553</code> vs <code>0.3411</code>).",
      },
    ];

    function renderAuditStep(stepIdx) {
      monographState.auditStep = ((stepIdx % 3) + 3) % 3;
      const idx = monographState.auditStep;
      const prog = bundle.audit_progression[idx];
      const diff = AUDIT_STEP_DIFFS[idx];

      if (prog) {
        setTextById(
          "audit-step-verdict",
          prog.iteration + ": " + prog.verdict + " | " + prog.p0_count + " P0, " + prog.p1_count + " P1"
        );
      }
      document.querySelectorAll(".audit-step-btn[data-audit-step]").forEach((btn) => {
        btn.classList.toggle("active", Number(btn.getAttribute("data-audit-step")) === idx);
      });

      const card = document.getElementById("audit-step-card");
      if (card && diff) {
        const topBoxCls = idx === 2 ? "audit-diff-box resolved" : "audit-diff-box flaw";
        card.innerHTML =
          '<div class="' +
          topBoxCls +
          '">' +
          '<div class="audit-diff-label">' +
          escapeHtml(diff.flawTitle) +
          "</div>" +
          "<div>" +
          diff.flawHtml +
          "</div>" +
          "</div>" +
          '<div class="audit-diff-box fix">' +
          '<div class="audit-diff-label">' +
          escapeHtml(diff.fixTitle) +
          "</div>" +
          "<div>" +
          diff.fixHtml +
          "</div>" +
          "</div>";
        renderKaTeXMath(card);
      }
    }

    function drawAuditCompare() {
      const canvas = document.getElementById("canvas-audit-compare");
      if (!canvas) return;
      if (!monographState.auditSim) {
        monographState.auditSim = engine.runTwinSimulation(
          {
            scaleL: 5,
            controlMode: "closed_loop",
            muTarget: 0.28,
            manualFeedRate: 0.08,
            rpm: 450,
            vvm: 1.2,
            disturbanceMode: "aeration_drop",
          },
          bundle
        );
      }
      const sim = monographState.auditSim;
      const mode = monographState.auditCompare;
      const C = getThemeColors();
      const { ctx, w, h } = setupHiDpiCanvas(canvas);
      ctx.clearRect(0, 0, w, h);

      const stacked = w < 560;
      const subW = stacked ? w : w * 0.5;
      const subH = stacked ? h * 0.5 : h;
      const showNaive = mode === "both" || mode === "naive";
      const showAudit = mode === "both" || mode === "audited";
      const tMax = sim.t[sim.t.length - 1] || 16;

      function renderSub(ox, oy, sw, sh, title, unitLabel, maxVal, getterTrue, getterNom, getterHyb, getterStd) {
        const pad = { l: 44, r: 26, t: 26, b: 26 };
        const pw = Math.max(50, sw - pad.l - pad.r);
        const ph = Math.max(50, sh - pad.t - pad.b);
        const px = (k) => ox + pad.l + (sim.t[k] / tMax) * pw;
        const py = (v) => oy + pad.t + ph * (1 - clampVal(v, 0, maxVal) / maxVal);

        // Highlight aeration drop fault window 6.0..9.5h (-55% vvm)
        const fx0 = ox + pad.l + (6.0 / tMax) * pw;
        const fx1 = ox + pad.l + (9.5 / tMax) * pw;
        ctx.fillStyle = C.alertWash;
        ctx.fillRect(fx0, oy + pad.t, fx1 - fx0, ph);
        ctx.font = '9.5px "IBM Plex Mono", monospace';
        ctx.fillStyle = C.alertInk;
        ctx.textAlign = "center";
        ctx.fillText(sw < 360 ? "-55% Air" : "-55% Air Fault (6-9.5h)", (fx0 + fx1) * 0.5, oy + pad.t + 12);

        // Horizontal ticks
        ctx.strokeStyle = C.border;
        ctx.lineWidth = 1;
        for (let g = 0; g <= 3; g++) {
          const y = Math.round(oy + pad.t + (ph * g) / 3) + 0.5;
          ctx.beginPath();
          ctx.moveTo(ox + pad.l, y);
          ctx.lineTo(ox + sw - pad.r, y);
          ctx.stroke();
          ctx.font = '10px "IBM Plex Mono", monospace';
          ctx.fillStyle = C.muted;
          ctx.textAlign = "right";
          ctx.fillText(fmt(maxVal * (1 - g / 3), 1), ox + pad.l - 5, y + 3);
        }

        // Bottom time ticks (0h, 4h, 8h, 12h, 16h)
        ctx.font = '9.5px "IBM Plex Mono", monospace';
        ctx.fillStyle = C.muted;
        for (const hr of [0, 4, 8, 12, 16]) {
          const tx = ox + pad.l + (hr / tMax) * pw;
          ctx.textAlign = hr === 0 ? "left" : hr === 16 ? "right" : "center";
          ctx.fillText(hr + "h", tx, oy + pad.t + ph + 14);
        }

        // +/- 2 sigma band for Verified EKF
        if (showAudit && getterStd) {
          ctx.fillStyle = C.hybBand;
          ctx.beginPath();
          for (let k = 0; k < sim.t.length; k++) {
            const x = px(k);
            const y = py(getterHyb(k) + 2 * getterStd(k));
            if (k === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          for (let k = sim.t.length - 1; k >= 0; k--) {
            ctx.lineTo(px(k), py(Math.max(0, getterHyb(k) - 2 * getterStd(k))));
          }
          ctx.closePath();
          ctx.fill();
        }

        // Plant truth solid teal
        ctx.strokeStyle = C.gt;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        for (let k = 0; k < sim.t.length; k++) {
          if (k === 0) ctx.moveTo(px(k), py(getterTrue(k)));
          else ctx.lineTo(px(k), py(getterTrue(k)));
        }
        ctx.stroke();

        // Naive dashed slate
        if (showNaive) {
          ctx.save();
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = C.nom;
          ctx.lineWidth = 2.1;
          ctx.beginPath();
          for (let k = 0; k < sim.t.length; k++) {
            if (k === 0) ctx.moveTo(px(k), py(getterNom(k)));
            else ctx.lineTo(px(k), py(getterNom(k)));
          }
          ctx.stroke();
          ctx.restore();
        }

        // Verified EKF solid amber
        if (showAudit) {
          ctx.strokeStyle = C.hyb;
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          for (let k = 0; k < sim.t.length; k++) {
            if (k === 0) ctx.moveTo(px(k), py(getterHyb(k)));
            else ctx.lineTo(px(k), py(getterHyb(k)));
          }
          ctx.stroke();
        }

        ctx.font = '600 11px "IBM Plex Sans", sans-serif';
        ctx.fillStyle = C.ink;
        ctx.textAlign = "left";
        ctx.fillText(title, ox + pad.l, oy + 15);
        ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.fillStyle = C.muted;
        ctx.textAlign = "right";
        ctx.fillText(unitLabel, ox + sw - pad.r, oy + 15);
      }

      let maxXd = 2.5;
      for (let k = 0; k < sim.t.length; k++) {
        maxXd = Math.max(maxXd, sim.trueStates[k][1] * 1.18, sim.estNom[k][1] * 1.18, sim.estHyb[k][1] * 1.18);
      }

      renderSub(
        0,
        0,
        subW,
        subH,
        subW < 420 ? "Dead Biomass X_d" : "Unmeasured Dead Biomass X_d (Teal=True)",
        "g/L",
        maxXd,
        (k) => sim.trueStates[k][1],
        (k) => sim.estNom[k][1],
        (k) => sim.estHyb[k][1],
        (k) => sim.stdHyb[k][1]
      );
      renderSub(
        stacked ? 0 : subW,
        stacked ? subH : 0,
        subW,
        subH,
        subW < 420 ? "Dissolved Oxygen C_L (Amber=EKF)" : "Dissolved Oxygen C_L (Amber=Audited, Slate=Naive)",
        "mg/L",
        (sim.C_star || 8.2) * 1.08,
        (k) => sim.trueStates[k][3],
        (k) => sim.estNom[k][3],
        (k) => sim.estHyb[k][3],
        (k) => sim.stdHyb[k][3]
      );
    }

    function bindSection3Audit() {
      document.querySelectorAll(".audit-step-btn[data-audit-step]").forEach((btn) => {
        btn.addEventListener("click", () => renderAuditStep(Number(btn.getAttribute("data-audit-step"))));
      });
      const nextBtn = document.getElementById("audit-next-btn");
      if (nextBtn) {
        nextBtn.addEventListener("click", () => renderAuditStep((monographState.auditStep + 1) % 3));
      }
      document.querySelectorAll(".audit-cmp-btn[data-cmp]").forEach((btn) => {
        btn.addEventListener("click", () => {
          monographState.auditCompare = btn.getAttribute("data-cmp");
          document.querySelectorAll(".audit-cmp-btn[data-cmp]").forEach((b) => {
            b.classList.toggle("active", b === btn);
          });
          drawAuditCompare();
        });
      });
      renderAuditStep(2);
      drawAuditCompare();
    }

    // ---- Section 4: Geometric Scale-Up & Live B=15 Bootstrap UQ Fan ----
    function getScaleScanData(vTip) {
      const members = bundle.gam86.ensemble_weights || [];
      const B = members.length;
      const xVals = [];
      const meanEta = [];
      const stdEta = [];
      const dmVals = [];
      const memCurves = Array.from({ length: B }, () => []);
      const muM =
        bundle.p_rec.mu_max *
        (0.3 / (bundle.p_rec.K_S + 0.3)) *
        (4.5 / (bundle.p_rec.K_O2 + 4.5));

      for (let xt = 5; xt <= 95; xt += 3) {
        xVals.push(xt);
        const xv = xt * 0.95;
        const xd = xt * 0.05;
        const p = engine.predictGamWithUQ(xv, xd, 0.3, 4.5, 0.2, vTip, muM, bundle);
        meanEta.push(p.eta_kla);
        stdEta.push(p.std_kla);
        dmVals.push(p.d_m);
        for (let b = 0; b < B; b++) {
          memCurves[b].push(p.member_eta_kla[b]);
        }
      }
      return { xVals, meanEta, stdEta, dmVals, memCurves };
    }

    function drawScaleupExplorer() {
      const canvas = document.getElementById("canvas-scaleup");
      if (!canvas) return;
      const C = getThemeColors();
      const { ctx, w, h } = setupHiDpiCanvas(canvas);
      ctx.clearRect(0, 0, w, h);

      const xt = monographState.scaleXtot;
      const vTip = monographState.scaleVtip;
      const muM =
        bundle.p_rec.mu_max *
        (0.3 / (bundle.p_rec.K_S + 0.3)) *
        (4.5 / (bundle.p_rec.K_O2 + 4.5));
      const curUQ = engine.predictGamWithUQ(0.95 * xt, 0.05 * xt, 0.3, 4.5, 0.2, vTip, muM, bundle);
      const p99 = bundle.gam86.dov_stats.global.p99 || 5.0118;

      setTextById("val-scale-xtot", fmt(xt, 0) + " g/L");
      setTextById("val-scale-vtip", fmt(vTip, 3) + " m/s");
      const badge = document.getElementById("scaleup-dm-badge");
      if (badge) {
        if (curUQ.in_domain) {
          badge.textContent = "IN-DOMAIN (D_M = " + fmt(curUQ.d_m, 2) + " <= " + fmt(p99, 2) + ")";
          badge.classList.remove("ood");
        } else {
          badge.textContent = "OOD GUARD ACTIVE (D_M = " + fmt(curUQ.d_m, 2) + " > " + fmt(p99, 2) + ")";
          badge.classList.add("ood");
        }
      }

      if (!monographState.scaleEnsCache || Math.abs(monographState.scaleEnsCache.vTip - vTip) > 1e-4) {
        monographState.scaleEnsCache = { vTip: vTip, data: getScaleScanData(vTip) };
      }
      const scan = monographState.scaleEnsCache.data;

      const stacked = w < 580;
      const leftW = stacked ? w : Math.round(w * 0.39);
      const leftH = stacked ? Math.round(h * 0.42) : h;

      // Left Panel: 5 L vs 1000 L Geometric Comparison
      ctx.fillStyle = C.panel;
      ctx.fillRect(0, 0, leftW, leftH);
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      if (stacked) {
        ctx.beginPath();
        ctx.moveTo(0, leftH + 0.5);
        ctx.lineTo(w, leftH + 0.5);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.moveTo(leftW + 0.5, 0);
        ctx.lineTo(leftW + 0.5, h);
        ctx.stroke();
      }

      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("SCALE INVARIANCE (200x VOLUME JUMP)", 12, 18);

      // Draw 5 L vessel (smaller) & 1000 L vessel (larger)
      const baseY = leftH - 36;
      const v5W = 28;
      const v5H = Math.min(54, leftH * 0.28);
      const v5X = leftW * 0.22 - v5W * 0.5;
      ctx.fillStyle = C.gtWash;
      ctx.strokeStyle = C.gtInk;
      ctx.lineWidth = 1.6;
      ctx.fillRect(v5X, baseY - v5H * 0.72, v5W, v5H * 0.72);
      ctx.strokeRect(v5X, baseY - v5H, v5W, v5H);
      ctx.font = '600 10.5px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.ink;
      ctx.textAlign = "center";
      ctx.fillText("5 L", leftW * 0.22, baseY - v5H - 6);
      ctx.font = '10px "IBM Plex Sans", sans-serif';
      ctx.fillStyle = C.muted;
      ctx.fillText("450 RPM", leftW * 0.22, baseY + 14);
      ctx.fillText("v_tip 1.21", leftW * 0.22, baseY + 26);

      const v1kW = clampVal(leftW * 0.34, 72, 118);
      const v1kH = clampVal(leftH * 0.52, 86, 154);
      const v1kX = leftW * 0.68 - v1kW * 0.5;
      ctx.fillStyle = curUQ.in_domain ? C.hybWash : C.alertWash;
      ctx.strokeStyle = curUQ.in_domain ? C.hybInk : C.alertInk;
      ctx.lineWidth = 2;
      ctx.fillRect(v1kX, baseY - v1kH * 0.75, v1kW, v1kH * 0.75);
      ctx.strokeRect(v1kX, baseY - v1kH, v1kW, v1kH);

      // Impeller tip-speed vector bar inside 1000 L vessel
      const impY = baseY - v1kH * 0.28;
      const impW = v1kW * 0.56;
      ctx.strokeStyle = vTip > 2.6 ? C.alert : C.ink;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(leftW * 0.68 - impW * 0.5, impY);
      ctx.lineTo(leftW * 0.68 + impW * 0.5, impY);
      ctx.stroke();

      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.ink;
      ctx.fillText("1000 L", leftW * 0.68, baseY - v1kH - 6);
      ctx.font = '10px "IBM Plex Sans", sans-serif';
      ctx.fillStyle = vTip > 2.6 ? C.alertInk : C.muted;
      const d1000 = engine.computeHydrodynamics(135, 0.2, 400, { ...bundle.p_rec, V_vessel: 1000 }).D_i_m;
      const eqRpm = (vTip * 60) / (Math.PI * d1000);
      ctx.fillText(fmt(eqRpm, 1) + " RPM (v_tip " + fmt(vTip, 3) + ")", leftW * 0.68, baseY + 14);
      ctx.fillStyle = C.muted;
      ctx.fillText("delta_kd = " + fmt(curUQ.delta_kd, 3) + " +/- " + fmt(2 * curUQ.std_kd, 3), leftW * 0.68, baseY + 26);

      // Arrow 5 L -> 1000 L
      ctx.strokeStyle = C.borderStrong;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(v5X + v5W + 8, baseY - v5H * 0.5);
      ctx.lineTo(v1kX - 8, baseY - v5H * 0.5);
      ctx.stroke();

      // Right Panel: Live B=15 Bootstrap Ensemble eta_kLa(X_tot) + +/- 2 sigma DoV Ribbon
      const rx = stacked ? 0 : leftW;
      const ry = stacked ? leftH : 0;
      const rw = stacked ? w : w - leftW;
      const rh = stacked ? h - leftH : h;
      const pad = { l: 48, r: 18, t: 28, b: 28 };
      const pw = Math.max(60, rw - pad.l - pad.r);
      const ph = Math.max(60, rh - pad.t - pad.b);
      const px = (xVal) => rx + pad.l + ((xVal - 5) / 90) * pw;
      const yNorm = (eta) => ry + pad.t + ph * (1 - clampVal(eta, 0.0, 1.05) / 1.05);

      // Shade sparse OOD regime above p99 transition
      for (let i = 0; i < scan.xVals.length; i++) {
        if (scan.dmVals[i] > p99) {
          const xStart = px(scan.xVals[i]);
          ctx.fillStyle = C.alertWash;
          ctx.fillRect(xStart, ry + pad.t, rx + pad.l + pw - xStart, ph);
          ctx.font = '10px "IBM Plex Mono", monospace';
          ctx.fillStyle = C.alertInk;
          ctx.textAlign = "right";
          ctx.fillText("OOD Zone (D_M > p99)", rx + rw - pad.r - 4, ry + pad.t + 12);
          break;
        }
      }

      // Horizontal grid
      ctx.strokeStyle = C.border;
      ctx.lineWidth = 1;
      for (const tick of [0.0, 0.25, 0.5, 0.75, 1.0]) {
        const y = Math.round(yNorm(tick)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(rx + pad.l, y);
        ctx.lineTo(rx + rw - pad.r, y);
        ctx.stroke();
        ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.fillStyle = C.muted;
        ctx.textAlign = "right";
        ctx.fillText(fmt(tick, 2), rx + pad.l - 6, y + 3);
      }

      // Shaded +/- 2 sigma band
      ctx.fillStyle = C.hybBand;
      ctx.beginPath();
      for (let i = 0; i < scan.xVals.length; i++) {
        const x = px(scan.xVals[i]);
        const y = yNorm(scan.meanEta[i] + 2 * scan.stdEta[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      for (let i = scan.xVals.length - 1; i >= 0; i--) {
        ctx.lineTo(px(scan.xVals[i]), yNorm(scan.meanEta[i] - 2 * scan.stdEta[i]));
      }
      ctx.closePath();
      ctx.fill();

      // All B=15 individual ensemble member curves
      ctx.save();
      ctx.globalAlpha = 0.34;
      ctx.strokeStyle = C.nom;
      ctx.lineWidth = 1;
      for (const curve of scan.memCurves) {
        ctx.beginPath();
        for (let i = 0; i < scan.xVals.length; i++) {
          const x = px(scan.xVals[i]);
          const y = yNorm(curve[i]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.restore();

      // Mean eta_kLa trace
      ctx.strokeStyle = C.hyb;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let i = 0; i < scan.xVals.length; i++) {
        const x = px(scan.xVals[i]);
        const y = yNorm(scan.meanEta[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Interactive cursor dot at selected X_tot
      const cx = px(xt);
      const cy = yNorm(curUQ.eta_kla);
      ctx.strokeStyle = C.borderStrong;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx, ry + pad.t);
      ctx.lineTo(cx, ry + pad.t + ph);
      ctx.stroke();

      ctx.fillStyle = curUQ.in_domain ? C.gt : C.alert;
      ctx.strokeStyle = C.bg;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Labels & live readout
      ctx.font = '600 10.5px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.ink;
      ctx.textAlign = "left";
      ctx.fillText(
        "B=15 k_La Attenuation eta_kLa = " + fmt(curUQ.eta_kla, 3) + " +/- " + fmt(2 * curUQ.std_kla, 3),
        rx + pad.l,
        ry + 16
      );
      ctx.font = '10px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.muted;
      ctx.textAlign = "center";
      ctx.fillText("Total Broth Biomass X_tot (5 to 95 g/L)", rx + pad.l + pw * 0.5, ry + rh - 8);
    }

    function bindSection4ScaleUp() {
      const presets = {
        matched: { xtot: 25, vtip: 2.125 },
        naive: { xtot: 42, vtip: 3.30 },
        ood: { xtot: 84, vtip: 2.95 },
      };
      const slX = document.getElementById("slider-scale-xtot");
      const slV = document.getElementById("slider-scale-vtip");

      function syncPresetBtns(mode) {
        document.querySelectorAll(".scale-mode-btn[data-scale-mode]").forEach((b) => {
          b.classList.toggle("active", b.getAttribute("data-scale-mode") === mode);
        });
      }

      document.querySelectorAll(".scale-mode-btn[data-scale-mode]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const mode = btn.getAttribute("data-scale-mode");
          const p = presets[mode];
          if (!p) return;
          monographState.scaleMode = mode;
          monographState.scaleXtot = p.xtot;
          monographState.scaleVtip = p.vtip;
          if (slX) slX.value = String(p.xtot);
          if (slV) slV.value = String(p.vtip);
          syncPresetBtns(mode);
          drawScaleupExplorer();
        });
      });

      if (slX) {
        slX.addEventListener("input", function () {
          monographState.scaleXtot = Number(this.value);
          syncPresetBtns("");
          drawScaleupExplorer();
        });
      }
      if (slV) {
        slV.addEventListener("input", function () {
          monographState.scaleVtip = Number(this.value);
          syncPresetBtns("");
          drawScaleupExplorer();
        });
      }
      drawScaleupExplorer();
    }

    // =========================================================================
    // SECTION 6: WET-LAB CHO-K1 VALIDATION & TRAJECTORY EXPLORER
    // =========================================================================
    function drawWetlabVolumeChart() {
      const cv = document.getElementById("chart-wetlab-volume");
      const val = global.UHLENDORFF_2026_VALIDATION;
      if (!cv || !val || !val.mass_balance_discovery || !val.mass_balance_discovery.volume_series) return;
      const d = val.mass_balance_discovery.volume_series.EXP010;
      if (!d) return;

      const { ctx, w, h } = setupHiDpiCanvas(cv);
      if (!w || !h) return;
      const theme = getThemeColors();
      ctx.clearRect(0, 0, w, h);

      const pad = { l: 48, r: 16, t: 22, b: 30 };
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;
      const tMax = d.t[d.t.length - 1];
      const yMin = 0.4, yMax = 1.5;
      const xOf = (t) => pad.l + (t / tMax) * pw;
      const yOf = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

      ctx.strokeStyle = theme.grid;
      ctx.lineWidth = 1;
      [0.5, 0.8, 1.1, 1.4].forEach((yv) => {
        const y = yOf(yv);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
        ctx.fillStyle = theme.axis;
        ctx.font = "10px 'IBM Plex Mono', monospace";
        ctx.textAlign = "right";
        ctx.fillText(yv.toFixed(1) + "L", pad.l - 6, y + 3);
      });

      // Naive volume
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      d.t.forEach((t, i) => (i === 0 ? ctx.moveTo(xOf(t), yOf(d.v_naive[i])) : ctx.lineTo(xOf(t), yOf(d.v_naive[i]))));
      ctx.stroke();
      ctx.setLineDash([]);

      // True inline volume
      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      d.t.forEach((t, i) => (i === 0 ? ctx.moveTo(xOf(t), yOf(d.v_true[i])) : ctx.lineTo(xOf(t), yOf(d.v_true[i]))));
      ctx.stroke();

      ctx.font = "11px 'IBM Plex Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "#dc2626";
      ctx.fillText("Naive V0 + m_feed/1000 (1.402 L)", pad.l + 6, pad.t + 12);
      ctx.fillStyle = "#0f766e";
      ctx.fillText("True Inline Volume (0.647 L)", pad.l + 6, pad.t + 26);
    }

    function drawWetlabStateCanvas(canvasId, expId, stateIdx) {
      const cv = document.getElementById(canvasId);
      const val = global.UHLENDORFF_2026_VALIDATION;
      if (!cv || !val || !val.held_out_trajectories) return;
      const tr = val.held_out_trajectories[expId];
      if (!tr) return;

      const { ctx, w, h } = setupHiDpiCanvas(cv);
      if (!w || !h) return;
      const theme = getThemeColors();
      ctx.clearRect(0, 0, w, h);

      const pad = { l: 44, r: 16, t: 24, b: 30 };
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;

      const r1Vals = tr.obs_r01.map((row) => row[stateIdx]);
      const r2Vals = tr.obs_r02.map((row) => row[stateIdx]);
      const htMean = tr.ht_mean.map((row) => row[stateIdx]);
      const htUpper = tr.ht_mean.map((row, i) => row[stateIdx] + 1.96 * tr.ht_std[i][stateIdx]);
      const htLower = tr.ht_mean.map((row, i) => Math.max(0, row[stateIdx] - 1.96 * tr.ht_std[i][stateIdx]));
      const vsMean = tr.vs_mean.map((row) => row[stateIdx]);
      const ekfVals = stateIdx === 0 && tr.ekf_r01 ? tr.ekf_r01.vcd : [];

      const tMax = Math.max(tr.t_off_r01[tr.t_off_r01.length - 1], tr.t_off_r02[tr.t_off_r02.length - 1]);
      const yMax = Math.max(1e-3, ...r1Vals, ...r2Vals, ...htUpper, ...vsMean, ...ekfVals) * 1.12;
      const xOf = (t) => pad.l + (t / tMax) * pw;
      const yOf = (v) => pad.t + ph - (v / yMax) * ph;

      // Grid
      ctx.strokeStyle = theme.grid;
      ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const yv = (yMax * g) / 4;
        const y = yOf(yv);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
        ctx.fillStyle = theme.axis;
        ctx.font = "10px 'IBM Plex Mono', monospace";
        ctx.textAlign = "right";
        ctx.fillText(yv.toFixed(1), pad.l - 6, y + 3);
      }

      // Feed start line if fed-batch
      if (tr.t_feed_start !== null && tr.t_feed_start < tMax) {
        const xf = xOf(tr.t_feed_start);
        ctx.strokeStyle = "rgba(180, 83, 9, 0.45)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(xf, pad.t);
        ctx.lineTo(xf, pad.t + ph);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 95% CI band
      ctx.fillStyle = "rgba(37, 99, 235, 0.14)";
      ctx.beginPath();
      tr.t_off_r01.forEach((t, i) => (i === 0 ? ctx.moveTo(xOf(t), yOf(htUpper[i])) : ctx.lineTo(xOf(t), yOf(htUpper[i]))));
      for (let i = tr.t_off_r01.length - 1; i >= 0; i--) {
        ctx.lineTo(xOf(tr.t_off_r01[i]), yOf(htLower[i]));
      }
      ctx.closePath();
      ctx.fill();

      // @von_stosch_twin baseline
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      tr.t_off_r01.forEach((t, i) => (i === 0 ? ctx.moveTo(xOf(t), yOf(vsMean[i])) : ctx.lineTo(xOf(t), yOf(vsMean[i]))));
      ctx.stroke();
      ctx.setLineDash([]);

      // Open-loop hybrid-fedbatch-twin mean
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 2;
      ctx.beginPath();
      tr.t_off_r01.forEach((t, i) => (i === 0 ? ctx.moveTo(xOf(t), yOf(htMean[i])) : ctx.lineTo(xOf(t), yOf(htMean[i]))));
      ctx.stroke();

      // Closed-loop Pade EKF on VCD (stateIdx === 0)
      if (stateIdx === 0 && tr.ekf_r01) {
        ctx.strokeStyle = "#0f766e";
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        tr.ekf_r01.t.forEach((t, i) => (i === 0 ? ctx.moveTo(xOf(t), yOf(tr.ekf_r01.vcd[i])) : ctx.lineTo(xOf(t), yOf(tr.ekf_r01.vcd[i]))));
        ctx.stroke();

        // Mark PAT dropout steps along bottom axis
        ctx.fillStyle = "#d97706";
        tr.ekf_r01.t.forEach((t, i) => {
          if (tr.ekf_r01.dropout[i] === 1) {
            ctx.fillRect(xOf(t) - 1.5, pad.t + ph - 5, 3, 5);
          }
        });
      }

      // Observations R01 & R02
      const drawDots = (ts, vals, color) => {
        ctx.fillStyle = color;
        ts.forEach((t, i) => {
          ctx.beginPath();
          ctx.arc(xOf(t), yOf(vals[i]), 3.2, 0, Math.PI * 2);
          ctx.fill();
        });
      };
      drawDots(tr.t_off_r01, r1Vals, "#111827");
      drawDots(tr.t_off_r02, r2Vals, "#7c3aed");

      // Legend
      ctx.font = "10px 'IBM Plex Sans', sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "#111827";
      ctx.fillText("Obs R01/R02", pad.l + 4, pad.t + 10);
      ctx.fillStyle = "#2563eb";
      ctx.fillText("Open-Loop HT + 95% CI", pad.l + 72, pad.t + 10);
      if (stateIdx === 0) {
        ctx.fillStyle = "#0f766e";
        ctx.fillText("Pade EKF (PAT)", pad.l + 196, pad.t + 10);
      } else {
        ctx.fillStyle = "#64748b";
        ctx.fillText("@von_stosch_twin", pad.l + 196, pad.t + 10);
      }

      // X-axis ticks
      ctx.fillStyle = theme.axis;
      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      [0, Math.round(tMax / 2), Math.round(tMax)].forEach((tv) => {
        ctx.fillText(tv + "h", xOf(tv), h - 8);
      });
    }

    function drawWetlabTrajectories() {
      const expId = monographState.wetlabExp || "EXP008";
      drawWetlabVolumeChart();
      drawWetlabStateCanvas("chart-wetlab-vcd", expId, 0);
      drawWetlabStateCanvas("chart-wetlab-glc", expId, 1);
      drawWetlabStateCanvas("chart-wetlab-lac", expId, 2);
      drawWetlabStateCanvas("chart-wetlab-amm", expId, 3);
    }

    function bindSection6Wetlab() {
      monographState.wetlabExp = "EXP008";
      const btns = document.querySelectorAll(".wetlab-exp-btn");
      const sub = document.getElementById("wetlab-campaign-subtitle");
      const labels = {
        EXP008: "EXP008 (Batch) · Open-Loop B=15 95% CI vs. Closed-Loop Padé EKF + Causal PAT",
        EXP009: "EXP009 (Shifted Preculture Batch) · Open-Loop B=15 95% CI vs. Closed-Loop Padé EKF + Causal PAT",
        EXP012: "EXP012 (Fed-Batch t_feed=48h) · Open-Loop B=15 95% CI vs. Closed-Loop Padé EKF + Causal PAT",
      };
      btns.forEach((btn) => {
        btn.addEventListener("click", () => {
          const expId = btn.getAttribute("data-wetlab-exp");
          monographState.wetlabExp = expId;
          btns.forEach((b) => b.classList.toggle("btn-primary", b === btn));
          if (sub && labels[expId]) sub.textContent = labels[expId];
          drawWetlabTrajectories();
        });
      });
      drawWetlabTrajectories();
    }

    function drawAllMonographCanvases() {
      drawDilemmaCanvas();
      drawPatankarDemo();
      drawAuditCompare();
      drawScaleupExplorer();
      drawWetlabTrajectories();
    }

    appState.monographRedraw = drawAllMonographCanvases;

    global.addEventListener("resize", function () {
      drawAllCharts();
      drawAllMonographCanvases();
    });
    if (global.ResizeObserver) {
      const chartsGrid = document.querySelector(".charts-grid");
      if (chartsGrid) {
        new ResizeObserver(drawAllCharts).observe(chartsGrid);
      }
      const mainPaper = document.getElementById("main-content");
      if (mainPaper) {
        new ResizeObserver(drawAllMonographCanvases).observe(mainPaper);
      }
    }

    renderKaTeXMath(document);
    global.addEventListener("load", () => renderKaTeXMath(document));
    initTocScrollSpy();
    bindSection1Dilemma();
    bindSection2Explorer();
    bindSection3Audit();
    bindSection4ScaleUp();
    bindSection6Wetlab();

    hydrateCaseStudy();
    syncControlUI();
    runAndRenderSimulation();
  });
})(window);
