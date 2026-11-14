/**
 * hybrid-fedbatch-twin: Zero-Backend Client-Side Bioprocess Simulation & EKF Engine
 * Faithfully ports hybrid_fedbatch_twin/{mechanistic,neural_ode,estimator,controller}.py:
 *   - Geometric cylindrical hydrodynamics (Pg/V, v_s, v_tip, Van't Riet k_L a) across 5 L <-> 1000 L
 *   - Serial Monod/Luedeking-Piret stoichiometry (Y_AS, Y_AO, maintenance, overflow)
 *   - GAM-86 (B=15 bootstrap ensemble) + Subspace Mahalanobis DoV + OOD prior relaxation
 *   - Exponential-Rosenbrock (phi1) positivity-preserving ODE integration
 *   - Continuous-Discrete EKF with Padé scaling-and-squaring matrix exponential expm(F*dt),
 *     7x5 stoichiometric epistemic process noise Q_epistemic, and optional delayed-LIMS replay
 */

(function (global) {
  "use strict";

  const STATE_NAMES = ["X_v", "X_d", "S", "C_L", "A", "P", "V"];
  const STATE_NORM_SCALES = [15.0, 2.0, 2.0, 8.2, 2.5, 5.0, 3.0];
  const Q_DEFAULT = [0.04, 5e-4, 0.04, 0.18, 0.04, 0.04, 1e-4];
  const P0_DEFAULT = [0.04, 0.01, 0.05, 0.10, 0.01, 0.02, 1e-4];

  function sigmoid(x) {
    const xc = x < -35 ? -35 : (x > 35 ? 35 : x);
    return 1.0 / (1.0 + Math.exp(-xc));
  }

  function softplus(x) {
    return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0.0);
  }

  function phi1(z) {
    if (Math.abs(z) <= 1e-5) {
      return 1.0 - 0.5 * z + (z * z) / 6.0 - (z * z * z) / 24.0;
    }
    const zc = z < -50 ? -50 : (z > 80 ? 80 : z);
    return (1.0 - Math.exp(-zc)) / z;
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  }

  function vesselGeometry(params) {
    const vVesM3 = 1e-3 * Math.max(1e-4, params.V_vessel);
    const tankD = Math.pow((4.0 * vVesM3) / (Math.PI * Math.max(0.5, params.H_to_T_nominal)), 1.0 / 3.0);
    const impellerD = params.D_to_T * tankD;
    const areaTank = 0.25 * Math.PI * tankD * tankD;
    return { V_ves_m3: vVesM3, T_m: tankD, D_i_m: impellerD, A_tank_m2: areaTank };
  }

  function computeHydrodynamics(rpm, vvm, V, params) {
    const geom = vesselGeometry(params);
    const nRps = Math.max(50.0, rpm) / 60.0;
    const vM3 = 1e-3 * Math.max(0.05, V);
    const qG = (Math.max(0.01, vvm) * vM3) / 60.0;
    const dI = geom.D_i_m;
    const nAe = qG / (nRps * dI * dI * dI + 1e-12);

    const p0 = params.N_p * params.broth_density * Math.pow(nRps, 3) * Math.pow(dI, 5);
    const pg = p0 / (1.0 + 3.2 * Math.pow(nAe, 0.7));
    const pgV = pg / vM3;
    const vS = qG / geom.A_tank_m2;
    const vTip = Math.PI * nRps * dI;
    return {
      N_rps: nRps,
      Q_g_m3_s: qG,
      N_Ae: nAe,
      P_g_W: pg,
      Pg_V: pgV,
      v_s: vS,
      v_tip: vTip,
      T_m: geom.T_m,
      D_i_m: dI,
    };
  }

  // Reference 5 L vessel baseline (400 RPM, 1.0 vvm, V0 = 2.0 L) matching mechanistic.py:_REF_HYDRO_5L
  const REF_5L_BASE = {
    V_vessel: 5.0,
    H_to_T_nominal: 2.0,
    D_to_T: 0.35,
    N_p: 10.0,
    broth_density: 1020.0,
  };
  const REF_HYDRO_5L = computeHydrodynamics(400.0, 1.0, 2.0, REF_5L_BASE);

  function nominalKla(rpm, vvm, V, params) {
    const hyd = computeHydrodynamics(rpm, vvm, V, params);
    const pgRatio = Math.max(1e-6, hyd.Pg_V / REF_HYDRO_5L.Pg_V);
    const vsRatio = Math.max(1e-6, hyd.v_s / REF_HYDRO_5L.v_s);
    return params.k_La0 * Math.pow(pgRatio, 0.40) * Math.pow(vsRatio, 0.50);
  }

  function trueKlaViscosityFactor(X_total) {
    const xPos = Math.max(0.0, X_total);
    return 1.0 / (1.0 + 0.85 * Math.pow(xPos / 18.0, 1.6));
  }

  function trueDynamicDeathRate(S, C_L, A, vTip, params) {
    const sPos = Math.max(0.0, S);
    const clPos = Math.max(0.0, C_L);
    const aPos = Math.max(0.0, A);
    const shearExcess = Math.max(0.0, (vTip - params.v_tip_crit_ref) / Math.max(1e-6, params.v_tip_span_ref));
    const kdAcetate = (0.045 * aPos * aPos) / (3.5 + aPos * aPos);
    const kdHypoxia = 0.055 * Math.exp(-clPos / 0.40);
    const kdShear = 0.030 * shearExcess * shearExcess;
    const kdStarvation = 0.020 * Math.exp(-sPos / 0.020);
    return params.k_d0 + kdAcetate + kdHypoxia + kdShear + kdStarvation;
  }

  function trueAcetateRates(mu, S, C_L, A) {
    const sPos = Math.max(0.0, S);
    const clPos = Math.max(0.0, C_L);
    const aPos = Math.max(0.0, A);
    const sAvail = sPos / (0.002 + sPos);
    const o2Avail = clPos / (0.005 + clPos);
    const aAvail = aPos / (0.25 + aPos);
    const qProd =
      (0.38 * Math.max(0.0, mu - 0.30) + 0.055 * (sPos / (0.5 + sPos)) * (0.6 / (0.6 + clPos))) * sAvail;
    const qCons =
      0.040 * (aPos / (0.5 + aPos)) * (0.25 / (0.25 + sPos)) * (clPos / (0.5 + clPos)) * aAvail * o2Avail;
    return [qProd, qCons];
  }

  function disturbanceProfile(distMode, t, rpmNom, vvmNom, S_feedNom, scaleFactor) {
    let rpm = rpmNom;
    let vvm = vvmNom;
    let sFeed = S_feedNom;
    const sf = scaleFactor || 1.0;
    const m = (distMode || "none").toLowerCase();

    if (m === "none") {
      return [rpm, vvm, sFeed];
    }
    if ((m === "aeration_drop" || m === "do_drop") && t >= 6.0 && t <= 9.5) {
      vvm = Math.max(0.35 / Math.pow(sf, 1.0 / 3.0), 0.45 * vvmNom);
    } else if ((m === "stirrer_drop" || m === "agitation_drop") && t >= 7.0 && t <= 10.5) {
      rpm = Math.max(180.0 / Math.pow(sf, 2.0 / 9.0), 0.55 * rpmNom);
    } else if (m === "shear_spike" && t >= 8.0 && t <= 12.0) {
      const spikeDelta = 260.0 / Math.pow(sf, 2.0 / 9.0);
      rpm = Math.min(900.0 / Math.pow(sf, 2.0 / 9.0), rpmNom + spikeDelta);
    } else if ((m === "feed_step" || m === "feed_dilution") && t >= 7.0) {
      sFeed = 0.70 * S_feedNom;
    } else if (m === "combo") {
      if (t >= 5.5 && t <= 8.5) {
        vvm = Math.max(0.40 / Math.pow(sf, 1.0 / 3.0), 0.55 * vvmNom);
      }
      if (t >= 9.0) {
        sFeed = 0.80 * S_feedNom;
      }
    }
    return [rpm, vvm, sFeed];
  }

  // Exact GAM-86 Head Evaluator reading unpacked flat weights (length 86)
  // Layout: kd_w1(8), kd_b1(8), kd_w2(8), kla_c(2), kla_p(2),
  //         ace_W1(20), ace_b1(5), ace_W2(10), ace_b2(2),
  //         mu_W1(12), mu_b1(4), mu_w2(4), mu_b2(1)
  function evalGamHeads(theta, X_tot, S, C_L, A, vTip, muMech, vtipOnset, klaScales) {
    const sPos = Math.max(0.0, S);
    const clPos = Math.max(0.0, C_L);
    const aPos = Math.max(0.0, A);
    const xtPos = Math.max(0.0, X_tot);

    const sAvail = sPos / (0.002 + sPos);
    const o2Avail = clPos / (0.005 + clPos);
    const aAvail = aPos / (0.25 + aPos);

    // 1. Dynamic Death Rate delta_kd (4x2 additive sigmoids anchored at basal zero)
    const aRat = aPos / (1.5 + aPos);
    const zk0 = 1.0 / (1.0 + sPos / 0.05);
    const zk1 = 1.0 / (1.0 + clPos / 0.50);
    const zk2 = 2.0 * aRat * aRat;
    const zk3 = Math.max(0.0, vTip - vtipOnset);
    const zkd = [zk0, zk1, zk2, zk3];

    let dkdSum = 0.0;
    for (let i = 0; i < 4; i++) {
      const zv = zkd[i];
      for (let j = 0; j < 2; j++) {
        const idx = i * 2 + j;
        const w1 = theta[idx];
        const b1 = theta[8 + idx];
        const w2 = theta[16 + idx];
        dkdSum += (sigmoid(zv * w1 + b1) - sigmoid(b1)) * w2;
      }
    }
    const deltaKd = Math.max(0.0, 0.05 * dkdSum);

    // 2. Power-Law Broth Rheology Attenuation eta_kla
    const u0 = 1.0 + xtPos / klaScales[0];
    const u1 = 1.0 + xtPos / klaScales[1];
    const psiKla = theta[24] * (Math.pow(u0, theta[26]) - 1.0) + theta[25] * (Math.pow(u1, theta[27]) - 1.0);
    const etaKla = 1.0 / (1.0 + Math.max(0.0, psiKla));

    // 3. Specific Growth Rate Multiplier eta_mu (h_mu = 4, z_mu dim = 3)
    const zm0 = sPos / (0.5 + sPos);
    const zm1 = Math.log1p(sPos * 0.5) * 0.4;
    const zm2 = aPos / (3.0 + aPos);
    let muLogit = theta[85]; // mu_b2[0]
    for (let k = 0; k < 4; k++) {
      const off = 65 + k * 3;
      const act = Math.tanh(theta[off] * zm0 + theta[off + 1] * zm1 + theta[off + 2] * zm2 + theta[77 + k]);
      muLogit += theta[81 + k] * act;
    }
    const etaMu = 0.05 + sigmoid(muLogit);
    const muHyb = muMech * etaMu;

    // 4. Overflow Acetate Production & Consumption (h_ace = 5, z_ace dim = 4)
    const za0 = muHyb / 0.45;
    const za1 = Math.log1p(sPos * 10.0) / 3.0;
    const za2 = Math.log1p(clPos * 2.0) / 3.0;
    const za3 = Math.log1p(aPos * 2.0) / 3.0;
    let ra0 = theta[63]; // ace_b2[0]
    let ra1 = theta[64]; // ace_b2[1]
    for (let k = 0; k < 5; k++) {
      const off = 28 + k * 4;
      const h = Math.tanh(
        theta[off] * za0 + theta[off + 1] * za1 + theta[off + 2] * za2 + theta[off + 3] * za3 + theta[48 + k]
      );
      ra0 += theta[53 + k] * h;
      ra1 += theta[58 + k] * h;
    }
    const qAProd = 0.08 * softplus(ra0) * sAvail;
    const qACons = 0.08 * softplus(ra1) * aAvail * o2Avail;

    return [deltaKd, etaKla, qAProd, qACons, etaMu];
  }

  function mahalanobisDist(z, stats) {
    const mu = stats.mean;
    const sd = stats.std;
    const ic = stats.cov_inv;
    const d = z.length;
    const diff = new Array(d);
    for (let i = 0; i < d; i++) {
      diff[i] = (z[i] - mu[i]) / sd[i];
    }
    let d2 = 0.0;
    for (let i = 0; i < d; i++) {
      let rowDot = 0.0;
      const row = ic[i];
      for (let j = 0; j < d; j++) {
        rowDot += row[j] * diff[j];
      }
      d2 += diff[i] * rowDot;
    }
    return Math.sqrt(Math.max(0.0, d2));
  }

  function computeSubspaceDoV(X_tot, S, C_L, A, vTip, bundle) {
    const dov = bundle.gam86.dov_stats;
    const vtipOnset = bundle.constants.vtip_onset;
    const sPos = Math.max(0.0, S);
    const clPos = Math.max(0.0, C_L);
    const aPos = Math.max(0.0, A);
    const aRat = aPos / (1.5 + aPos);

    const zKd = [
      1.0 / (1.0 + sPos / 0.05),
      1.0 / (1.0 + clPos / 0.50),
      2.0 * aRat * aRat,
      Math.max(0.0, vTip - vtipOnset),
    ];
    const zKla = [Math.max(0.0, X_tot)];
    const zMet = [
      Math.log1p(sPos / 0.1) / 3.0,
      Math.log1p(clPos / 0.5) / 3.0,
      Math.log1p(aPos / 0.5) / 3.0,
      sPos / 5.0,
      aPos / 3.0,
    ];

    const dmKd = mahalanobisDist(zKd, dov.kd);
    const dmKla = mahalanobisDist(zKla, dov.kla);
    const dmMet = mahalanobisDist(zMet, dov.met);
    const dm = Math.max(dmKd, dmKla, dmMet);
    const p99Global = dov.global.p99;
    return {
      d_m: dm,
      d_m_kd: dmKd,
      d_m_kla: dmKla,
      d_m_met: dmMet,
      in_domain: dm <= p99Global,
      p99_kd: dov.kd.p99,
      p99_kla: dov.kla.p99,
      p99_met: dov.met.p99,
      p99_global: p99Global,
    };
  }

  function predictGamPointFast(X_v, X_d, S, C_L, A, vTip, muMech, bundle) {
    const theta = bundle.gam86.weights;
    const vtipOnset = bundle.constants.vtip_onset;
    const klaScales = bundle.constants.kla_scales;
    const xtPos = Math.max(0.0, X_v) + Math.max(0.0, X_d);
    const res = evalGamHeads(theta, xtPos, S, C_L, A, vTip, muMech, vtipOnset, klaScales);
    let dkd = res[0];
    const etaKla = res[1];
    let qp = res[2];
    let qc = res[3];
    let emu = res[4];

    const dov = computeSubspaceDoV(xtPos, S, C_L, A, vTip, bundle);
    const exKd = Math.max(0.0, dov.d_m_kd - dov.p99_kd);
    const exMet = Math.max(0.0, dov.d_m_met - dov.p99_met);
    const wKd = Math.exp(-0.5 * Math.pow(exKd / Math.max(0.5, 0.5 * dov.p99_kd), 2));
    const wMet = Math.exp(-0.5 * Math.pow(exMet / Math.max(0.35, 0.35 * dov.p99_met), 2));

    dkd *= wKd;
    qp *= wMet;
    qc *= wMet;
    emu = wMet * emu + (1.0 - wMet);

    return {
      delta_kd: dkd,
      eta_kla: etaKla,
      q_A_prod: qp,
      q_A_cons: qc,
      eta_mu: emu,
    };
  }

  function predictGamWithUQ(X_v, X_d, S, C_L, A, vTip, muMech, bundle) {
    const thetaMain = bundle.gam86.weights;
    const members = bundle.gam86.ensemble_weights;
    const vtipOnset = bundle.constants.vtip_onset;
    const klaScales = bundle.constants.kla_scales;
    const xtPos = Math.max(0.0, X_v) + Math.max(0.0, X_d);
    const B = members.length;

    const pt = evalGamHeads(thetaMain, xtPos, S, C_L, A, vTip, muMech, vtipOnset, klaScales);
    let dkdM = pt[0];
    const ekM = pt[1];
    let qpM = pt[2];
    let qcM = pt[3];
    let emuM = pt[4];

    const kdArr = new Float64Array(B);
    const ekArr = new Float64Array(B);
    const qpArr = new Float64Array(B);
    const qcArr = new Float64Array(B);
    const muArr = new Float64Array(B);

    let sKd = 0, sEk = 0, sQp = 0, sQc = 0, sMu = 0;
    for (let b = 0; b < B; b++) {
      const out = evalGamHeads(members[b], xtPos, S, C_L, A, vTip, muMech, vtipOnset, klaScales);
      kdArr[b] = out[0]; sKd += out[0];
      ekArr[b] = out[1]; sEk += out[1];
      qpArr[b] = out[2]; sQp += out[2];
      qcArr[b] = out[3]; sQc += out[3];
      muArr[b] = out[4]; sMu += out[4];
    }

    const mKd = sKd / B, mEk = sEk / B, mQp = sQp / B, mQc = sQc / B, mMu = sMu / B;
    let vKd = 0, vEk = 0, vQp = 0, vQc = 0, vMu = 0;
    if (B > 1) {
      for (let b = 0; b < B; b++) {
        vKd += (kdArr[b] - mKd) * (kdArr[b] - mKd);
        vEk += (ekArr[b] - mEk) * (ekArr[b] - mEk);
        vQp += (qpArr[b] - mQp) * (qpArr[b] - mQp);
        vQc += (qcArr[b] - mQc) * (qcArr[b] - mQc);
        vMu += (muArr[b] - mMu) * (muArr[b] - mMu);
      }
      vKd /= B; vEk /= B; vQp /= B; vQc /= B; vMu /= B;
    }

    const dov = computeSubspaceDoV(xtPos, S, C_L, A, vTip, bundle);
    const exKd = Math.max(0.0, dov.d_m_kd - dov.p99_kd);
    const exKla = Math.max(0.0, dov.d_m_kla - dov.p99_kla);
    const exMet = Math.max(0.0, dov.d_m_met - dov.p99_met);

    const wKd = Math.exp(-0.5 * Math.pow(exKd / Math.max(0.5, 0.5 * dov.p99_kd), 2));
    const wMet = Math.exp(-0.5 * Math.pow(exMet / Math.max(0.35, 0.35 * dov.p99_met), 2));

    dkdM *= wKd;
    qpM *= wMet;
    qcM *= wMet;
    emuM = wMet * emuM + (1.0 - wMet);

    const rKd = clamp(exKd / Math.max(1.0, dov.p99_kd), 0.0, 2.0);
    const rKla = clamp(exKla / Math.max(1.0, dov.p99_kla), 0.0, 2.0);
    const rMet = clamp(exMet / Math.max(1.0, dov.p99_met), 0.0, 2.0);

    return {
      delta_kd: dkdM,
      eta_kla: ekM,
      q_A_prod: qpM,
      q_A_cons: qcM,
      eta_mu: emuM,
      std_kd: Math.sqrt(vKd + Math.pow(0.002 + 0.015 * rKd, 2)),
      std_kla: Math.sqrt(vEk + Math.pow(0.025 + 0.065 * Math.min(1.0, rKla) * (1.0 - ekM), 2)),
      std_qp: Math.sqrt(vQp + Math.pow(0.002 + 0.015 * rMet, 2)),
      std_qc: Math.sqrt(vQc + Math.pow(0.002 + 0.015 * rMet, 2)),
      std_mu: Math.sqrt(vMu + Math.pow(0.015 + 0.12 * rMet, 2)),
      d_m: dov.d_m,
      d_m_kd: dov.d_m_kd,
      d_m_kla: dov.d_m_kla,
      d_m_met: dov.d_m_met,
      in_domain: dov.in_domain,
    };
  }

  function computeRates(x, u, params, mode, bundle, includeUQ) {
    const X_v = Math.max(0.0, x[0]);
    const X_d = Math.max(0.0, x[1]);
    const S = Math.max(0.0, x[2]);
    const C_L = Math.max(0.0, x[3]);
    const A = Math.max(0.0, x[4]);
    const V = Math.max(0.2, x[6]);

    const hyd = computeHydrodynamics(u.rpm, u.vvm, V, params);
    const klaNom = nominalKla(u.rpm, u.vvm, V, params);
    const sAvail = S / (0.002 + S);
    const o2Avail = C_L / (0.005 + C_L);
    const muMech = params.mu_max * (S / (params.K_S + S)) * (C_L / (params.K_O2 + C_L));

    let mu = muMech;
    let k_d = params.k_d0;
    let k_La = klaNom;
    let qAProd = 0.0;
    let qACons = 0.0;
    let etaMu = 1.0;
    let stdKd = 0, stdKla = 0, stdQp = 0, stdQc = 0, stdMu = 0;
    let d_m = 0, d_m_kla = 0, inDomain = true;

    if (mode === "nominal") {
      // Pure Monod + nominal k_d0 + nominal k_La
    } else if (mode === "ground_truth") {
      mu =
        params.mu_max *
        (S / (params.K_S + S + (S * S) / 60.0)) *
        (C_L / (params.K_O2 + C_L)) *
        (8.0 / (8.0 + A));
      etaMu = muMech > 1e-12 ? mu / muMech : 1.0;
      k_d = trueDynamicDeathRate(S, C_L, A, hyd.v_tip, params);
      k_La = klaNom * trueKlaViscosityFactor(X_v + X_d);
      const qPair = trueAcetateRates(mu, S, C_L, A);
      qAProd = qPair[0];
      qACons = qPair[1];
    } else if (mode === "hybrid") {
      const out = includeUQ
        ? predictGamWithUQ(X_v, X_d, S, C_L, A, hyd.v_tip, muMech, bundle)
        : predictGamPointFast(X_v, X_d, S, C_L, A, hyd.v_tip, muMech, bundle);
      etaMu = out.eta_mu;
      mu = Math.max(0.0, muMech * etaMu);
      k_d = Math.max(0.0, params.k_d0 + out.delta_kd);
      k_La = Math.max(5.0, klaNom * out.eta_kla);
      qAProd = Math.max(0.0, out.q_A_prod);
      qACons = Math.max(0.0, out.q_A_cons);
      if (includeUQ) {
        stdKd = out.std_kd;
        stdKla = out.std_kla;
        stdQp = out.std_qp;
        stdQc = out.std_qc;
        stdMu = out.std_mu;
        d_m = out.d_m;
        d_m_kla = out.d_m_kla;
        inDomain = out.in_domain;
      }
    }

    const qP = (params.alpha * mu + params.beta) * sAvail * o2Avail;
    return {
      mu: mu,
      mu_mech: muMech,
      eta_mu: etaMu,
      k_d: k_d,
      k_La: k_La,
      k_La_nom: klaNom,
      q_A_prod: qAProd,
      q_A_cons: qACons,
      q_A_net: qAProd - qACons,
      q_P: qP,
      std_kd: stdKd,
      std_kla: stdKla,
      std_qp: stdQp,
      std_qc: stdQc,
      std_mu: stdMu,
      d_m: d_m,
      d_m_kla: d_m_kla,
      in_domain: inDomain,
      Pg_V: hyd.Pg_V,
      v_s: hyd.v_s,
      v_tip: hyd.v_tip,
    };
  }

  function bioreactorRhs(x, u, params, mode, bundle) {
    const X_v = Math.max(0.0, x[0]);
    const X_d = Math.max(0.0, x[1]);
    const S = Math.max(0.0, x[2]);
    const C_L = Math.max(0.0, x[3]);
    const A = Math.max(0.0, x[4]);
    const P = Math.max(0.0, x[5]);
    const V = Math.max(0.2, x[6]);

    const F = V >= params.V_max ? 0.0 : Math.max(0.0, u.F);
    const D = F / V;
    const S_feed = u.S_feed == null ? params.S_feed : u.S_feed;

    const r = computeRates(x, u, params, mode, bundle, false);
    const sAvail = S / (0.002 + S);
    const o2Avail = C_L / (0.005 + C_L);

    const dXv = (r.mu - r.k_d - D) * X_v;
    const dXd = r.k_d * X_v - D * X_d;
    const qS = r.mu / params.Y_XS + params.m_s * sAvail + r.q_P / params.Y_PS + r.q_A_prod / params.Y_AS;
    const dS = D * (S_feed - S) - qS * X_v;
    const our = 1000.0 * (r.mu / params.Y_XO + params.m_o * o2Avail + r.q_A_cons / params.Y_AO) * X_v;
    const otr = r.k_La * (params.C_star - C_L);
    const dCL = otr - our - D * C_L;
    const dA = (r.q_A_prod - r.q_A_cons) * X_v - D * A;
    const dP = r.q_P * X_v - D * P;
    const dV = F;

    return [dXv, dXd, dS, dCL, dA, dP, dV];
  }

  // Positivity-preserving Exponential-Rosenbrock Patankar integration step
  function integrateStep(dtTotal, x0, u, params, mode, bundle, nSubsteps) {
    const steps = nSubsteps || 10;
    const dt = dtTotal / steps;
    const x = [
      Math.max(0.0, x0[0]),
      Math.max(0.0, x0[1]),
      Math.max(0.0, x0[2]),
      Math.max(0.0, x0[3]),
      Math.max(0.0, x0[4]),
      Math.max(0.0, x0[5]),
      Math.max(0.2, x0[6]),
    ];
    const S_feed = u.S_feed == null ? params.S_feed : u.S_feed;

    for (let s = 0; s < steps; s++) {
      const X_v = x[0], X_d = x[1], S = x[2], C_L = x[3], A = x[4], P = x[5], V = Math.max(0.2, x[6]);
      const F = V >= params.V_max ? 0.0 : Math.max(0.0, u.F);
      const D = F / V;
      const r = computeRates(x, u, params, mode, bundle, false);

      const sAvail = S / (0.002 + S);
      const dsa = 0.002 / ((0.002 + S) * (0.002 + S));
      const o2Avail = C_L / (0.005 + C_L);
      const doa = 0.005 / ((0.005 + C_L) * (0.005 + C_L));

      const dSm = params.K_S / Math.pow(params.K_S + S, 2);
      const dOm = params.K_O2 / Math.pow(params.K_O2 + C_L, 2);
      const sM = S / (params.K_S + S);
      const oM = C_L / (params.K_O2 + C_L);

      const dMu_dS = params.mu_max * dSm * oM * r.eta_mu;
      const dMu_dCL = params.mu_max * sM * dOm * r.eta_mu;
      const dqP_dS = params.alpha * dMu_dS * sAvail * o2Avail + (params.alpha * r.mu + params.beta) * dsa * o2Avail;

      const qS = r.mu / params.Y_XS + params.m_s * sAvail + r.q_P / params.Y_PS + r.q_A_prod / params.Y_AS;
      const dqS_dS = dMu_dS / params.Y_XS + params.m_s * dsa + dqP_dS / params.Y_PS;
      const our = 1000.0 * (r.mu / params.Y_XO + params.m_o * o2Avail + r.q_A_cons / params.Y_AO) * X_v;
      const dOur_dCL = 1000.0 * (dMu_dCL / params.Y_XO + params.m_o * doa) * X_v;

      const z2 = (D + X_v * dqS_dS) * dt;
      const p2 = phi1(z2);
      const z3 = (r.k_La + D + dOur_dCL) * dt;
      const p3 = phi1(z3);

      // Patankar positive form for Xv, Xd, A, P
      const xvNext = (X_v + dt * r.mu * X_v) / (1.0 + dt * (r.k_d + D));
      const xdNext = (X_d + dt * r.k_d * X_v) / (1.0 + dt * D);
      const sNext = Math.max(0.0, S + dt * p2 * (D * (S_feed - S) - qS * X_v));
      const clNext = clamp(
        C_L + dt * p3 * (r.k_La * (params.C_star - C_L) - our - D * C_L),
        0.0,
        params.C_star * 1.05
      );
      const aNext = Math.max(0.0, (A + dt * r.q_A_prod * X_v) / (1.0 + dt * (D + (r.q_A_cons * X_v) / (A + 1e-6))));
      const pNext = (P + dt * r.q_P * X_v) / (1.0 + dt * D);
      const vNext = Math.min(params.V_max + 0.05, V + dt * F);

      x[0] = Math.max(0.0, xvNext);
      x[1] = Math.max(0.0, xdNext);
      x[2] = sNext;
      x[3] = clNext;
      x[4] = aNext;
      x[5] = Math.max(0.0, pNext);
      x[6] = vNext;
    }
    return x;
  }

  // Dense N x N linear algebra helpers for 7x7 Matrix Exponential Padé(6,6) + EKF update
  function eye(n) {
    const m = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) m[i][i] = 1.0;
    return m;
  }

  function matMul(A, B) {
    const n = A.length;
    const m = B[0].length;
    const p = B.length;
    const C = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) {
      const aRow = A[i];
      const cRow = C[i];
      for (let k = 0; k < p; k++) {
        const aik = aRow[k];
        if (aik === 0.0) continue;
        const bRow = B[k];
        for (let j = 0; j < m; j++) {
          cRow[j] += aik * bRow[j];
        }
      }
    }
    return C;
  }

  function matScaleAdd(cA, A, cB, B) {
    const n = A.length;
    const m = A[0].length;
    const C = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        C[i][j] = cA * A[i][j] + cB * B[i][j];
      }
    }
    return C;
  }

  function matOneNorm(A) {
    const n = A.length;
    let maxCol = 0.0;
    for (let j = 0; j < n; j++) {
      let colSum = 0.0;
      for (let i = 0; i < n; i++) colSum += Math.abs(A[i][j]);
      if (colSum > maxCol) maxCol = colSum;
    }
    return maxCol;
  }

  function solveLinearSystem(M_in, B_in) {
    const n = M_in.length;
    const rhsCols = B_in[0].length;
    const A = M_in.map((r) => Float64Array.from(r));
    const B = B_in.map((r) => Float64Array.from(r));

    for (let col = 0; col < n; col++) {
      let pivRow = col;
      let pivMax = Math.abs(A[col][col]);
      for (let r = col + 1; r < n; r++) {
        const val = Math.abs(A[r][col]);
        if (val > pivMax) {
          pivMax = val;
          pivRow = r;
        }
      }
      if (pivRow !== col) {
        const tmpA = A[col]; A[col] = A[pivRow]; A[pivRow] = tmpA;
        const tmpB = B[col]; B[col] = B[pivRow]; B[pivRow] = tmpB;
      }
      const diag = Math.abs(A[col][col]) < 1e-14 ? (A[col][col] >= 0 ? 1e-14 : -1e-14) : A[col][col];
      for (let r = col + 1; r < n; r++) {
        const f = A[r][col] / diag;
        if (f === 0.0) continue;
        for (let k = col; k < n; k++) A[r][k] -= f * A[col][k];
        for (let k = 0; k < rhsCols; k++) B[r][k] -= f * B[col][k];
      }
    }
    for (let r = n - 1; r >= 0; r--) {
      const diag = Math.abs(A[r][r]) < 1e-14 ? (A[r][r] >= 0 ? 1e-14 : -1e-14) : A[r][r];
      for (let k = 0; k < rhsCols; k++) {
        let sum = B[r][k];
        for (let c = r + 1; c < n; c++) sum -= A[r][c] * B[c][k];
        B[r][k] = sum / diag;
      }
    }
    return B;
  }

  // Scaling-and-squaring Padé(6,6) matrix exponential for stiff 7x7 EKF Jacobian (zero eigenvalue clipping)
  function expm7x7(A_in) {
    const n = 7;
    const norm1 = matOneNorm(A_in);
    const s = Math.max(0, Math.ceil(Math.log2(Math.max(1e-12, norm1 / 0.25))));
    const scale = Math.pow(0.5, s);
    const A = A_in.map((r) => {
      const nr = new Float64Array(n);
      for (let j = 0; j < n; j++) nr[j] = r[j] * scale;
      return nr;
    });

    // Padé(6,6) coefficients: b = [1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280]
    const b = [1.0, 0.5, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0];
    const I = eye(n);
    const A2 = matMul(A, A);
    const A4 = matMul(A2, A2);
    const A6 = matMul(A4, A2);

    // U = A * (b1*I + b3*A2 + b5*A4)
    const U_inner = Array.from({ length: n }, () => new Float64Array(n));
    const V = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        U_inner[i][j] = b[1] * I[i][j] + b[3] * A2[i][j] + b[5] * A4[i][j];
        V[i][j] = b[0] * I[i][j] + b[2] * A2[i][j] + b[4] * A4[i][j] + b[6] * A6[i][j];
      }
    }
    const U = matMul(A, U_inner);
    const N = matScaleAdd(1.0, V, 1.0, U);
    const D = matScaleAdd(1.0, V, -1.0, U);
    let R = solveLinearSystem(D, N);
    for (let k = 0; k < s; k++) {
      R = matMul(R, R);
    }
    return R;
  }

  function makeDiag(vec) {
    const n = vec.length;
    const M = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) M[i][i] = vec[i];
    return M;
  }

  function symmetrize(P) {
    const n = P.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const s = 0.5 * (P[i][j] + P[j][i]);
        P[i][j] = s;
        P[j][i] = s;
      }
      if (P[i][i] < 0) P[i][i] = 0.0;
    }
    return P;
  }

  function createEKF(x0, params, mode, bundle) {
    return {
      x: x0.slice(),
      P: makeDiag(P0_DEFAULT),
      Q_base: makeDiag(Q_DEFAULT),
      params: params,
      mode: mode,
      bundle: bundle,
      lastRates: null,
      lastEpCov: Array.from({ length: 7 }, () => new Float64Array(7)),
    };
  }

  function ekfJacobian(ekf, x, u) {
    const Fmat = Array.from({ length: 7 }, () => new Float64Array(7));
    for (let j = 0; j < 7; j++) {
      const eps = 1e-5 * Math.max(1.0, Math.abs(x[j]));
      const xp = x.slice();
      const xm = x.slice();
      xp[j] += eps;
      xm[j] = Math.max(0.0, xm[j] - eps);
      const denom = xp[j] - xm[j];
      if (denom <= 0.0) continue;
      const fp = bioreactorRhs(xp, u, ekf.params, ekf.mode, ekf.bundle);
      const fm = bioreactorRhs(xm, u, ekf.params, ekf.mode, ekf.bundle);
      for (let i = 0; i < 7; i++) {
        Fmat[i][j] = (fp[i] - fm[i]) / denom;
      }
    }
    return Fmat;
  }

  function ekfEpistemicCov(ekf, x, rates) {
    const Qep = Array.from({ length: 7 }, () => new Float64Array(7));
    if (ekf.mode !== "hybrid") return Qep;

    const X_v = Math.max(0.0, x[0]);
    const S = Math.max(0.0, x[2]);
    const C_L = Math.max(0.0, x[3]);
    const sAv = S / (0.002 + S);
    const o2Av = C_L / (0.005 + C_L);
    const muM = rates.mu_mech;
    const klaNom = rates.k_La_nom;
    const p = ekf.params;

    // 7x5 stoichiometric projection G matching estimator.py:119-148
    const G = Array.from({ length: 7 }, () => new Float64Array(5));
    G[0][0] = -X_v;
    G[1][0] = +X_v;
    G[3][1] = klaNom * (p.C_star - C_L);
    G[2][2] = -X_v / p.Y_AS;
    G[4][2] = +X_v;
    G[3][3] = (-1000.0 * X_v) / p.Y_AO;
    G[4][3] = -X_v;
    G[0][4] = muM * X_v;
    G[2][4] = -(muM / p.Y_XS + (p.alpha * muM * sAv * o2Av) / p.Y_PS) * X_v;
    G[3][4] = (-1000.0 * (muM / p.Y_XO)) * X_v;
    G[5][4] = p.alpha * muM * sAv * o2Av * X_v;

    const vars = [
      rates.std_kd * rates.std_kd,
      rates.std_kla * rates.std_kla,
      rates.std_qp * rates.std_qp,
      rates.std_qc * rates.std_qc,
      rates.std_mu * rates.std_mu,
    ];
    for (let i = 0; i < 7; i++) {
      for (let j = i; j < 7; j++) {
        let sum = 0.0;
        for (let k = 0; k < 5; k++) {
          sum += G[i][k] * vars[k] * G[j][k];
        }
        Qep[i][j] = sum;
        Qep[j][i] = sum;
      }
    }
    return Qep;
  }

  function ekfPredict(ekf, u, dt) {
    if (dt <= 0.0) return;
    const xPred = integrateStep(dt, ekf.x, u, ekf.params, ekf.mode, ekf.bundle, 10);
    const xMid = ekf.x.map((v, i) => 0.5 * (v + xPred[i]));
    const Fmat = ekfJacobian(ekf, xMid, u);
    const Fdt = Fmat.map((r) => r.map((v) => v * dt));
    const Phi = expm7x7(Fdt);
    const ratesMid = computeRates(xMid, u, ekf.params, ekf.mode, ekf.bundle, true);
    const Qep = ekfEpistemicCov(ekf, xMid, ratesMid);

    // P = Phi * P * Phi^T + (Q_base + Q_ep) * dt
    const PhiP = matMul(Phi, ekf.P);
    const PhiT = Array.from({ length: 7 }, (_, i) => {
      const r = new Float64Array(7);
      for (let j = 0; j < 7; j++) r[j] = Phi[j][i];
      return r;
    });
    const Pprop = matMul(PhiP, PhiT);
    const dtQ = Math.max(dt, 1e-4);
    for (let i = 0; i < 7; i++) {
      for (let j = 0; j < 7; j++) {
        Pprop[i][j] += (ekf.Q_base[i][j] + Qep[i][j]) * dtQ;
      }
    }
    ekf.P = symmetrize(Pprop);
    ekf.x = xPred;
    ekf.lastRates = ratesMid;
    ekf.lastEpCov = Qep;
  }

  function ekfUpdate(ekf, y, obsIdx, rDiag) {
    const m = obsIdx.length;
    const n = 7;
    // S_cov = H P H^T + R (m x m)
    const Scov = Array.from({ length: m }, () => new Float64Array(m));
    const HP = Array.from({ length: m }, () => new Float64Array(n));
    for (let r = 0; r < m; r++) {
      const si = obsIdx[r];
      for (let j = 0; j < n; j++) HP[r][j] = ekf.P[si][j];
      for (let c = 0; c < m; c++) {
        const sj = obsIdx[c];
        Scov[r][c] = ekf.P[si][sj] + (r === c ? rDiag[r] : 0.0);
      }
    }
    // K = (S_cov^{-1} * H P)^T -> (7 x m)
    const K_T = solveLinearSystem(Scov, HP);
    const K = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(m);
      for (let c = 0; c < m; c++) row[c] = K_T[c][i];
      return row;
    });

    const innov = new Float64Array(m);
    for (let r = 0; r < m; r++) {
      innov[r] = y[r] - ekf.x[obsIdx[r]];
    }
    for (let i = 0; i < n; i++) {
      let dx = 0.0;
      for (let r = 0; r < m; r++) dx += K[i][r] * innov[r];
      ekf.x[i] = Math.max(0.0, ekf.x[i] + dx);
    }
    ekf.x[3] = Math.min(ekf.x[3], ekf.params.C_star * 1.05);
    ekf.x[6] = clamp(ekf.x[6], 0.2, ekf.params.V_max + 0.05);

    // Joseph-stabilized covariance update: P = (I - K H) P (I - K H)^T + K R K^T
    const IKH = eye(n);
    for (let i = 0; i < n; i++) {
      for (let r = 0; r < m; r++) {
        IKH[i][obsIdx[r]] -= K[i][r];
      }
    }
    const IKH_T = Array.from({ length: n }, (_, i) => {
      const r = new Float64Array(n);
      for (let j = 0; j < n; j++) r[j] = IKH[j][i];
      return r;
    });
    const Pnew = matMul(matMul(IKH, ekf.P), IKH_T);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let krk = 0.0;
        for (let r = 0; r < m; r++) krk += K[i][r] * rDiag[r] * K[j][r];
        Pnew[i][j] += krk;
        if (i !== j) Pnew[j][i] += krk;
      }
    }
    ekf.P = symmetrize(Pnew);
  }

  // Deterministic Mulberry32 PRNG for reproducible sensor noise in client UI
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randNormal(rng) {
    const u1 = Math.max(1e-12, rng());
    const u2 = rng();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  function closedLoopControl(t, xHat, cfg, params, bundle, scaleFactor) {
    const [rpmNow, vvmNow] = disturbanceProfile(
      cfg.disturbanceMode,
      t,
      cfg.rpm,
      cfg.vvm,
      params.S_feed,
      scaleFactor
    );

    const X_v = Math.max(0.05, xHat[0]);
    const S = Math.max(0.0, xHat[2]);
    const C_L = Math.max(0.0, xHat[3]);
    const A = Math.max(0.0, xHat[4]);
    const V = Math.max(0.2, xHat[6]);

    if (V >= params.V_max * 0.995) {
      return { F: 0.0, rpm: rpmNow, vvm: vvmNow, S_feed: null };
    }

    if (cfg.controlMode === "manual") {
      return {
        F: clamp(cfg.manualFeedRate, 0.0, cfg.F_max),
        rpm: rpmNow,
        vvm: vvmNow,
        S_feed: null,
      };
    }

    const uProbe = { F: 0.0, rpm: rpmNow, vvm: vvmNow, S_feed: null };
    const rHat = computeRates(xHat, uProbe, params, "hybrid", bundle, true);
    const muEff = rHat.mu;
    const etaMuHat = rHat.eta_mu;

    const inhibO2 = C_L / (params.K_O2 + C_L);
    const muCap = Math.max(muEff + 0.04, params.mu_max * inhibO2 * etaMuHat);

    const aLimit = 1.8;
    const sSet = 0.25;
    const aceThrottle = Math.max(0.35, 1.0 - Math.max(0.0, A - 0.55 * aLimit) / Math.max(0.25, aLimit));
    const o2Guard = clamp((C_L - 1.0) / 1.8, 0.35, 1.0);
    const o2Av = C_L / (0.005 + C_L);
    const otrAvail = (rHat.k_La * Math.max(0.0, params.C_star - Math.max(C_L, 2.2))) / 1000.0;
    const muOtrCap = Math.max(0.03, params.Y_XO * (otrAvail / Math.max(0.1, X_v) - params.m_o * o2Av));
    const muSp = Math.min(cfg.muTarget * aceThrottle * o2Guard, muCap, muOtrCap);

    const sSp = Math.max(0.08, sSet * Math.max(0.35, 1.0 - Math.max(0.0, A - 0.7 * aLimit) / Math.max(0.5, aLimit)));
    const sAvSp = sSp / (0.002 + sSp);
    const qPSp = (params.alpha * muSp + params.beta) * sAvSp * o2Av;
    const qSSp =
      muSp / params.Y_XS +
      params.m_s * sAvSp +
      qPSp / params.Y_PS +
      rHat.q_A_prod / params.Y_AS;

    const gainS = S <= sSp ? 1.6 : 2.4;
    const demandGh = qSSp * X_v * V + gainS * (sSp - S) * V;
    const Fcmd = clamp(demandGh / Math.max(10.0, params.S_feed - S), 0.0, cfg.F_max);
    return { F: Fcmd, rpm: rpmNow, vvm: vvmNow, S_feed: null };
  }

  /**
   * Run full 16 h fed-batch simulation + sensor-parity Hybrid EKF vs Nominal EKF
   */
  function runTwinSimulation(options, bundle) {
    const scaleL = options.scaleL || 5;
    const scaleFactor = scaleL / 5.0;

    // Model uses Stage-1 recovered parameters p_rec; plant uses ground-truth p_true
    const pModel = {
      ...bundle.p_rec,
      V_vessel: scaleL,
      V_0: 2.0 * scaleFactor,
      V_max: scaleL,
    };
    const pPlant = {
      ...bundle.p_true,
      V_vessel: scaleL,
      V_0: 2.0 * scaleFactor,
      V_max: scaleL,
    };

    const durationH = 16.0;
    const dtStep = 0.1;
    const nSteps = Math.round(durationH / dtStep);
    const atlineEvery = 10; // every 1.0 h

    const cfg = {
      controlMode: options.controlMode || "closed_loop",
      muTarget: options.muTarget != null ? options.muTarget : 0.28,
      manualFeedRate: options.manualFeedRate != null ? options.manualFeedRate : 0.08 * scaleFactor,
      rpm: options.rpm != null ? options.rpm : 450.0,
      vvm: options.vvm != null ? options.vvm : 1.2,
      F_max: 0.30 * scaleFactor,
      disturbanceMode: options.disturbanceMode || "none",
    };

    const x0 = [1.5, 0.02, 1.2, pPlant.C_star * 0.95, 0.05, 0.0, pPlant.V_0];
    const rng = mulberry32(101);

    const onlineIdx = [0, 3, 6];
    const sigOnline = [0.12, 0.08, 0.015 * scaleFactor];
    const rOnline = sigOnline.map((s) => s * s);
    const atlineIdx = [0, 2, 3, 4, 6];
    const sigAtline = [0.12, 0.06, 0.08, 0.05, 0.015 * scaleFactor];
    const rAtline = sigAtline.map((s) => s * s);

    const ekfHyb = createEKF(x0, pModel, "hybrid", bundle);
    const ekfNom = createEKF(x0, pModel, "nominal", bundle);

    const t = new Float64Array(nSteps + 1);
    const trueStates = Array.from({ length: nSteps + 1 });
    const estHyb = Array.from({ length: nSteps + 1 });
    const estNom = Array.from({ length: nSteps + 1 });
    const stdHyb = Array.from({ length: nSteps + 1 });
    const measStates = Array.from({ length: nSteps + 1 });
    const inputs = Array.from({ length: nSteps + 1 });
    const obsAtline = new Uint8Array(nSteps + 1);

    const muTrue = new Float64Array(nSteps + 1);
    const muHyb = new Float64Array(nSteps + 1);
    const muNom = new Float64Array(nSteps + 1);
    const kdTrue = new Float64Array(nSteps + 1);
    const kdHyb = new Float64Array(nSteps + 1);
    const klaTrue = new Float64Array(nSteps + 1);
    const klaHyb = new Float64Array(nSteps + 1);
    const klaNom = new Float64Array(nSteps + 1);
    const dmArr = new Float64Array(nSteps + 1);
    const dmKlaArr = new Float64Array(nSteps + 1);
    const vTipArr = new Float64Array(nSteps + 1);
    const pgVArr = new Float64Array(nSteps + 1);
    const vsArr = new Float64Array(nSteps + 1);
    const inDomain = new Uint8Array(nSteps + 1);

    let xTrue = x0.slice();
    trueStates[0] = xTrue.slice();
    estHyb[0] = ekfHyb.x.slice();
    estNom[0] = ekfNom.x.slice();
    stdHyb[0] = Array.from({ length: 7 }, (_, j) => Math.sqrt(Math.max(0.0, ekfHyb.P[j][j])));

    const m0 = [NaN, NaN, NaN, NaN, NaN, NaN, NaN];
    for (let k = 0; k < atlineIdx.length; k++) {
      const si = atlineIdx[k];
      m0[si] = Math.max(0.0, xTrue[si] + randNormal(rng) * sigAtline[k]);
    }
    measStates[0] = m0;
    obsAtline[0] = 1;

    for (let i = 0; i < nSteps; i++) {
      const ti = Number((i * dtStep).toFixed(2));
      t[i] = ti;
      const uCmd = closedLoopControl(ti, ekfHyb.x, cfg, pModel, bundle, scaleFactor);
      inputs[i] = [uCmd.F, uCmd.rpm, uCmd.vvm];

      const [, , sFeedPlant] = disturbanceProfile(
        cfg.disturbanceMode,
        ti,
        uCmd.rpm,
        uCmd.vvm,
        pPlant.S_feed,
        scaleFactor
      );
      const uPlant = { F: uCmd.F, rpm: uCmd.rpm, vvm: uCmd.vvm, S_feed: sFeedPlant };
      const uModel = { F: uCmd.F, rpm: uCmd.rpm, vvm: uCmd.vvm, S_feed: null };

      const rt = computeRates(xTrue, uPlant, pPlant, "ground_truth", bundle, false);
      const rh = computeRates(ekfHyb.x, uModel, pModel, "hybrid", bundle, true);
      const rn = computeRates(ekfNom.x, uModel, pModel, "nominal", bundle, false);

      muTrue[i] = rt.mu;
      muHyb[i] = rh.mu;
      muNom[i] = rn.mu;
      kdTrue[i] = rt.k_d;
      kdHyb[i] = rh.k_d;
      klaTrue[i] = rt.k_La;
      klaHyb[i] = rh.k_La;
      klaNom[i] = rh.k_La_nom;
      dmArr[i] = rh.d_m;
      dmKlaArr[i] = rh.d_m_kla;
      vTipArr[i] = rh.v_tip;
      pgVArr[i] = rh.Pg_V;
      vsArr[i] = rh.v_s;
      inDomain[i] = rh.in_domain ? 1 : 0;

      xTrue = integrateStep(dtStep, xTrue, uPlant, pPlant, "ground_truth", bundle, 10);
      ekfPredict(ekfHyb, uModel, dtStep);
      ekfPredict(ekfNom, uModel, dtStep);

      const isAt = (i + 1) % atlineEvery === 0;
      obsAtline[i + 1] = isAt ? 1 : 0;
      const mk = [NaN, NaN, NaN, NaN, NaN, NaN, NaN];

      if (isAt) {
        const yVec = [];
        for (let k = 0; k < atlineIdx.length; k++) {
          const si = atlineIdx[k];
          const val = Math.max(0.0, xTrue[si] + randNormal(rng) * sigAtline[k]);
          yVec.push(val);
          mk[si] = val;
        }
        ekfUpdate(ekfHyb, yVec, atlineIdx, rAtline);
        ekfUpdate(ekfNom, yVec, atlineIdx, rAtline);
      } else {
        const yVec = [];
        for (let k = 0; k < onlineIdx.length; k++) {
          const si = onlineIdx[k];
          const val = Math.max(0.0, xTrue[si] + randNormal(rng) * sigOnline[k]);
          yVec.push(val);
          mk[si] = val;
        }
        ekfUpdate(ekfHyb, yVec, onlineIdx, rOnline);
        ekfUpdate(ekfNom, yVec, onlineIdx, rOnline);
      }

      measStates[i + 1] = mk;
      trueStates[i + 1] = xTrue.slice();
      estHyb[i + 1] = ekfHyb.x.slice();
      estNom[i + 1] = ekfNom.x.slice();
      stdHyb[i + 1] = Array.from({ length: 7 }, (_, j) => Math.sqrt(Math.max(0.0, ekfHyb.P[j][j])));
    }

    // Final point rates
    const tEnd = durationH;
    t[nSteps] = tEnd;
    const uEnd = closedLoopControl(tEnd, ekfHyb.x, cfg, pModel, bundle, scaleFactor);
    inputs[nSteps] = [uEnd.F, uEnd.rpm, uEnd.vvm];
    const [, , sFeedEnd] = disturbanceProfile(
      cfg.disturbanceMode,
      tEnd,
      uEnd.rpm,
      uEnd.vvm,
      pPlant.S_feed,
      scaleFactor
    );
    const rtE = computeRates(xTrue, { F: uEnd.F, rpm: uEnd.rpm, vvm: uEnd.vvm, S_feed: sFeedEnd }, pPlant, "ground_truth", bundle, false);
    const rhE = computeRates(ekfHyb.x, uEnd, pModel, "hybrid", bundle, true);
    const rnE = computeRates(ekfNom.x, uEnd, pModel, "nominal", bundle, false);

    muTrue[nSteps] = rtE.mu;
    muHyb[nSteps] = rhE.mu;
    muNom[nSteps] = rnE.mu;
    kdTrue[nSteps] = rtE.k_d;
    kdHyb[nSteps] = rhE.k_d;
    klaTrue[nSteps] = rtE.k_La;
    klaHyb[nSteps] = rhE.k_La;
    klaNom[nSteps] = rhE.k_La_nom;
    dmArr[nSteps] = rhE.d_m;
    dmKlaArr[nSteps] = rhE.d_m_kla;
    vTipArr[nSteps] = rhE.v_tip;
    pgVArr[nSteps] = rhE.Pg_V;
    vsArr[nSteps] = rhE.v_s;
    inDomain[nSteps] = rhE.in_domain ? 1 : 0;

    // Summary KPIs with scale-adjusted volume normalization
    const normScales = STATE_NORM_SCALES.slice();
    normScales[6] = 3.0 * scaleFactor;
    let sumSqHyb = 0.0;
    let sumSqNom = 0.0;
    let maxDm = 0.0;
    let oodCount = 0;
    let peakAceTrue = 0.0;
    let peakAceEst = 0.0;
    let minDOTrue = 100.0;
    let minDOEst = 100.0;
    for (let i = 0; i <= nSteps; i++) {
      for (let j = 0; j < 7; j++) {
        const eh = (estHyb[i][j] - trueStates[i][j]) / normScales[j];
        const en = (estNom[i][j] - trueStates[i][j]) / normScales[j];
        sumSqHyb += eh * eh;
        sumSqNom += en * en;
      }
      if (dmArr[i] > maxDm) maxDm = dmArr[i];
      if (!inDomain[i]) oodCount++;
      if (trueStates[i][4] > peakAceTrue) peakAceTrue = trueStates[i][4];
      if (estHyb[i][4] > peakAceEst) peakAceEst = estHyb[i][4];
      const doT = (100.0 * trueStates[i][3]) / pPlant.C_star;
      const doE = (100.0 * estHyb[i][3]) / pModel.C_star;
      if (doT < minDOTrue) minDOTrue = doT;
      if (doE < minDOEst) minDOEst = doE;
    }
    const nTotal = (nSteps + 1) * 7;
    const hybNormRmse = Math.sqrt(sumSqHyb / nTotal);
    const nomNormRmse = Math.sqrt(sumSqNom / nTotal);

    return {
      scaleL: scaleL,
      scaleFactor: scaleFactor,
      C_star: pModel.C_star,
      V_max: pModel.V_max,
      p99Global: bundle.gam86.dov_stats.global.p99,
      t: t,
      trueStates: trueStates,
      estHyb: estHyb,
      estNom: estNom,
      stdHyb: stdHyb,
      measStates: measStates,
      inputs: inputs,
      obsAtline: obsAtline,
      muTrue: muTrue,
      muHyb: muHyb,
      muNom: muNom,
      kdTrue: kdTrue,
      kdHyb: kdHyb,
      klaTrue: klaTrue,
      klaHyb: klaHyb,
      klaNom: klaNom,
      d_m: dmArr,
      d_m_kla: dmKlaArr,
      v_tip: vTipArr,
      Pg_V: pgVArr,
      v_s: vsArr,
      inDomain: inDomain,
      summary: {
        hyb_norm_rmse: hybNormRmse,
        nom_norm_rmse: nomNormRmse,
        rmse_ratio: hybNormRmse / Math.max(1e-9, nomNormRmse),
        final_Xv_true: trueStates[nSteps][0],
        final_Xv_est: estHyb[nSteps][0],
        final_Xd_true: trueStates[nSteps][1],
        final_Xd_est: estHyb[nSteps][1],
        final_P_true: trueStates[nSteps][5],
        final_P_est: estHyb[nSteps][5],
        peak_A_true: peakAceTrue,
        peak_A_est: peakAceEst,
        min_DO_true: minDOTrue,
        min_DO_est: minDOEst,
        max_d_m: maxDm,
        ood_fraction: oodCount / (nSteps + 1),
      },
    };
  }

  /**
   * Compute matched (RPM, vvm) when toggling between 5 L and 1000 L at equal (Pg/V, v_s)
   */
  function matchOperatingPointForScale(rpmFrom, vvmFrom, fromScaleL, toScaleL, params) {
    const pFrom = { ...params, V_vessel: fromScaleL, V_0: 0.4 * fromScaleL, V_max: fromScaleL };
    const pTo = { ...params, V_vessel: toScaleL, V_0: 0.4 * toScaleL, V_max: toScaleL };
    const hydTarget = computeHydrodynamics(rpmFrom, vvmFrom, pFrom.V_0, pFrom);
    const vvmTo = clamp(vvmFrom * Math.pow(fromScaleL / toScaleL, 1.0 / 3.0), 0.1, 3.0);

    // Bisection for rpmTo matching Pg/V
    let lo = 30.0;
    let hi = 1200.0;
    for (let it = 0; it < 40; it++) {
      const mid = 0.5 * (lo + hi);
      const pg = computeHydrodynamics(mid, vvmTo, pTo.V_0, pTo).Pg_V;
      if (pg < hydTarget.Pg_V) lo = mid;
      else hi = mid;
    }
    return { rpm: Math.round(0.5 * (lo + hi)), vvm: Number(vvmTo.toFixed(2)) };
  }

  global.HybridTwinEngine = {
    STATE_NAMES: STATE_NAMES,
    computeHydrodynamics: computeHydrodynamics,
    nominalKla: nominalKla,
    computeRates: computeRates,
    predictGamWithUQ: predictGamWithUQ,
    runTwinSimulation: runTwinSimulation,
    matchOperatingPointForScale: matchOperatingPointForScale,
  };
})(window);
