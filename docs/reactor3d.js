/**
 * hybrid-fedbatch-twin: Interactive Three.js 3D Stirred-Tank Bioreactor Viewport
 * Renders an orbitable 5 L benchtop borosilicate vessel or 1000 L industrial cutaway vessel
 * coupled in real time to volume V(t), biomass X_v(t), DO%, RPM shear tip speed v_tip,
 * aeration sparger vvm, and substrate feed rate F(t), with Canvas 2.5D fallback.
 */

(function (global) {
  "use strict";

  function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  }

  class Bioreactor3DView {
    constructor(containerEl, opts) {
      this.container = containerEl;
      this.opts = opts || {};
      this.theme = this.opts.theme || "dark";
      this.scaleL = 5;
      this.motionPaused = Boolean(
        global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches
      );

      // Live physical state fed from twin simulation
      this.state = {
        t_h: 0.0,
        X_v: 1.5,
        X_d: 0.02,
        S: 1.2,
        C_L: 7.79,
        DO_pct: 95.0,
        A: 0.05,
        P: 0.0,
        V: 2.0,
        V_max: 5.0,
        F: 0.02,
        rpm: 450.0,
        vvm: 1.2,
        v_tip: 1.21,
        Pg_V: 2680.0,
        k_La: 650.0,
        in_domain: true,
        d_m: 1.0,
      };

      // Camera orbit spherical coordinates
      this.orbit = {
        theta: 0.55,
        phi: 1.22,
        radius: 6.2,
        targetY: 0.1,
        dragging: false,
        lastX: 0,
        lastY: 0,
      };

      this._rafId = null;
      this._lastTime = performance.now();
      this._bubbleSeeds = [];
      this._dropSeeds = [];
      this.backend = "none";

      this._init();
    }

    _init() {
      const T = global.THREE;
      if (T) {
        try {
          this._initThreeJS(T);
          this.backend = "webgl";
        } catch (_err) {
          this._initCanvasFallback();
          this.backend = "canvas2d";
        }
      } else {
        this._initCanvasFallback();
        this.backend = "canvas2d";
      }
      this._bindInteractions();
      this._onResize();
      if (global.ResizeObserver) {
        this._ro = new ResizeObserver(() => this._onResize());
        this._ro.observe(this.container);
      } else {
        global.addEventListener("resize", () => this._onResize());
      }
      this._loop(performance.now());
    }

    _initThreeJS(THREE) {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
      this.renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
      this.renderer.domElement.style.width = "100%";
      this.renderer.domElement.style.height = "100%";
      this.renderer.domElement.style.display = "block";
      this.renderer.domElement.setAttribute("aria-label", "Interactive 3D Stirred-Tank Bioreactor Model");
      this.container.appendChild(this.renderer.domElement);

      this.scene = new THREE.Scene();
      const w = Math.max(200, this.container.clientWidth);
      const h = Math.max(200, this.container.clientHeight);
      this.camera = new THREE.PerspectiveCamera(36, w / h, 0.1, 100);

      // Clean scientific studio lighting
      this.ambientLight = new THREE.AmbientLight(0xdce6e4, 0.72);
      this.scene.add(this.ambientLight);
      this.keyLight = new THREE.DirectionalLight(0xffffff, 0.92);
      this.keyLight.position.set(5.0, 7.5, 5.5);
      this.scene.add(this.keyLight);
      this.rimLight = new THREE.DirectionalLight(0x2dd4bf, 0.45);
      this.rimLight.position.set(-5.0, 3.0, -4.0);
      this.scene.add(this.rimLight);

      this.root = new THREE.Group();
      this.scene.add(this.root);

      // Tank dimensions in scene units (normalized height ~ 3.2, radius ~ 0.95)
      this.tankR = 0.96;
      this.tankH = 3.1;
      this.tankBottomY = -1.55;

      // 1. Structural Pedestal / Load-Cell Base (V weight measurement)
      const metalDark = new THREE.MeshStandardMaterial({ color: 0x38434a, metalness: 0.75, roughness: 0.35 });
      const metalBright = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.82, roughness: 0.22 });
      const amberMat = new THREE.MeshStandardMaterial({ color: 0xf59e0b, metalness: 0.3, roughness: 0.4, emissive: 0x78350f, emissiveIntensity: 0.35 });
      const tealMat = new THREE.MeshStandardMaterial({ color: 0x2dd4bf, metalness: 0.25, roughness: 0.35, emissive: 0x0f766e, emissiveIntensity: 0.3 });

      const baseDisk = new THREE.Mesh(new THREE.CylinderGeometry(1.16, 1.24, 0.16, 48), metalDark);
      baseDisk.position.y = this.tankBottomY - 0.14;
      this.root.add(baseDisk);

      // Load-cell foot indicators
      for (let i = 0; i < 4; i++) {
        const ang = (i * Math.PI) / 2 + Math.PI / 4;
        const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.12, 0.10, 16), amberMat);
        foot.position.set(Math.cos(ang) * 0.95, this.tankBottomY - 0.26, Math.sin(ang) * 0.95);
        this.root.add(foot);
      }

      // 2. Vessel Body (5L Glass vs 1000L Industrial Cutaway)
      const glassMat = new THREE.MeshPhysicalMaterial({
        color: 0xbde4df,
        transparent: true,
        opacity: 0.18,
        roughness: 0.12,
        metalness: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      this.glassVessel = new THREE.Mesh(new THREE.CylinderGeometry(this.tankR, this.tankR, this.tankH, 48, 1, true), glassMat);
      this.glassVessel.position.y = 0;
      this.root.add(this.glassVessel);

      // Vessel bottom dish
      const bottomDish = new THREE.Mesh(
        new THREE.CylinderGeometry(this.tankR, this.tankR * 0.92, 0.14, 48),
        metalBright
      );
      bottomDish.position.y = this.tankBottomY;
      this.root.add(bottomDish);

      // 1000 L Industrial Stainless Cutaway Jacket (visible only when scaleL === 1000)
      this.industrialGroup = new THREE.Group();
      const steelJacketMat = new THREE.MeshStandardMaterial({
        color: 0x64748b,
        metalness: 0.78,
        roughness: 0.32,
        side: THREE.DoubleSide,
      });
      // Back 210-degree stainless cutaway cylinder so broth & impellers stay clearly visible from front
      const jacketShell = new THREE.Mesh(
        new THREE.CylinderGeometry(this.tankR + 0.04, this.tankR + 0.04, this.tankH * 0.92, 48, 1, true, Math.PI * 0.58, Math.PI * 1.25),
        steelJacketMat
      );
      this.industrialGroup.add(jacketShell);
      for (const ry of [-1.05, -0.35, 0.35, 1.05]) {
        const rib = new THREE.Mesh(new THREE.TorusGeometry(this.tankR + 0.055, 0.025, 12, 48), metalBright);
        rib.rotation.x = Math.PI / 2;
        rib.position.y = ry;
        this.industrialGroup.add(rib);
      }
      this.industrialGroup.visible = false;
      this.root.add(this.industrialGroup);

      // Headplate + Drive Motor Housing
      const headplate = new THREE.Mesh(new THREE.CylinderGeometry(this.tankR + 0.06, this.tankR + 0.06, 0.14, 48), metalBright);
      headplate.position.y = this.tankBottomY + this.tankH + 0.06;
      this.root.add(headplate);

      const motorHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.28, 0.55, 24), metalDark);
      motorHousing.position.y = this.tankBottomY + this.tankH + 0.42;
      this.root.add(motorHousing);

      // 4 Internal Baffles
      for (let b = 0; b < 4; b++) {
        const ang = (b * Math.PI) / 2 + Math.PI / 4;
        const baffle = new THREE.Mesh(new THREE.BoxGeometry(0.08, this.tankH * 0.88, 0.02), metalBright);
        baffle.position.set(Math.cos(ang) * (this.tankR - 0.06), 0, Math.sin(ang) * (this.tankR - 0.06));
        baffle.rotation.y = -ang;
        this.root.add(baffle);
      }

      // 3. Dynamic Working Volume Broth Cylinder + Meniscus Top
      this.brothMat = new THREE.MeshPhysicalMaterial({
        color: 0x1e6b63,
        transparent: true,
        opacity: 0.48,
        roughness: 0.25,
        metalness: 0.02,
        depthWrite: false,
      });
      this.brothCyl = new THREE.Mesh(new THREE.CylinderGeometry(this.tankR - 0.025, this.tankR - 0.025, 1.0, 44), this.brothMat);
      this.root.add(this.brothCyl);

      this.meniscusMat = new THREE.MeshBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.55, side: THREE.DoubleSide });
      this.meniscusDisk = new THREE.Mesh(new THREE.RingGeometry(0.06, this.tankR - 0.025, 44), this.meniscusMat);
      this.meniscusDisk.rotation.x = -Math.PI / 2;
      this.root.add(this.meniscusDisk);

      // 4. Central Shaft + Dual 6-Blade Rushton Turbines
      this.impellerGroup = new THREE.Group();
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, this.tankH + 0.15, 16), metalBright);
      shaft.position.y = 0.05;
      this.impellerGroup.add(shaft);

      const bladeGeo = new THREE.BoxGeometry(0.18, 0.14, 0.022);
      for (const yPos of [-0.95, -0.15]) {
        const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.05, 24), metalBright);
        hub.position.y = yPos;
        this.impellerGroup.add(hub);

        const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.30, 0.30, 0.018, 24), metalDark);
        disc.position.y = yPos;
        this.impellerGroup.add(disc);

        for (let k = 0; k < 6; k++) {
          const ang = (k * Math.PI) / 3;
          const blade = new THREE.Mesh(bladeGeo, metalBright);
          blade.position.set(Math.cos(ang) * 0.28, yPos, Math.sin(ang) * 0.28);
          blade.rotation.y = -ang;
          this.impellerGroup.add(blade);
        }
      }
      this.root.add(this.impellerGroup);

      // Shear Stress Halo Ring around lower impeller (reacts to v_tip)
      this.shearMat = new THREE.MeshBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.48, side: THREE.DoubleSide });
      this.shearRing = new THREE.Mesh(new THREE.RingGeometry(0.40, 0.54, 48), this.shearMat);
      this.shearRing.rotation.x = -Math.PI / 2;
      this.shearRing.position.y = -0.95;
      this.root.add(this.shearRing);

      // 5. Ring Sparger & Gas Inlet Pipe
      const spargerRing = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.032, 12, 36), metalBright);
      spargerRing.rotation.x = Math.PI / 2;
      spargerRing.position.y = this.tankBottomY + 0.22;
      this.root.add(spargerRing);

      // 6. Top Feed Nozzle & Nutrient Droplets
      const feedPipe = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.55, 12), amberMat);
      feedPipe.position.set(0.42, this.tankBottomY + this.tankH + 0.16, 0.18);
      this.root.add(feedPipe);

      this.dropGroup = new THREE.Group();
      const dropGeo = new THREE.SphereGeometry(0.030, 8, 8);
      for (let i = 0; i < 12; i++) {
        const drop = new THREE.Mesh(dropGeo, amberMat);
        this.dropGroup.add(drop);
        this._dropSeeds.push({ phase: i / 12 });
      }
      this.root.add(this.dropGroup);

      // 7. In-Situ PAT Sensor Probes (Dielectric Capacitance X_v + Optical DO C_L)
      const capProbe = new THREE.Mesh(new THREE.CylinderGeometry(0.030, 0.030, 2.35, 12), tealMat);
      capProbe.position.set(-0.52, 0.52, 0.28);
      this.root.add(capProbe);

      this.doTipMat = new THREE.MeshStandardMaterial({ color: 0x2dd4bf, emissive: 0x2dd4bf, emissiveIntensity: 0.65 });
      const doProbeBody = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 2.25, 12), metalBright);
      doProbeBody.position.set(-0.28, 0.56, -0.48);
      this.root.add(doProbeBody);
      const doProbeTip = new THREE.Mesh(new THREE.SphereGeometry(0.048, 12, 12), this.doTipMat);
      doProbeTip.position.set(-0.28, -0.58, -0.48);
      this.root.add(doProbeTip);

      // 8. Instanced Rising Sparger Bubbles (max 220 bubbles)
      this.maxBubbles = 220;
      const bubGeo = new THREE.SphereGeometry(0.024, 8, 8);
      const bubMat = new THREE.MeshBasicMaterial({ color: 0xdcfcf7, transparent: true, opacity: 0.72 });
      this.bubbleMesh = new THREE.InstancedMesh(bubGeo, bubMat, this.maxBubbles);
      this.bubbleDummy = new THREE.Object3D();
      for (let i = 0; i < this.maxBubbles; i++) {
        const ang = (i * 2.39996) % (Math.PI * 2);
        const rad = 0.25 + 0.58 * ((i * 17) % 100) / 100;
        this._bubbleSeeds.push({
          yFrac: Math.random(),
          ang: ang,
          rad: rad,
          speedScale: 0.75 + 0.5 * Math.random(),
          size: 0.65 + 0.7 * Math.random(),
        });
      }
      this.root.add(this.bubbleMesh);

      this._updateCamera();
      this.setTheme(this.theme);
    }

    _initCanvasFallback() {
      this.canvas = document.createElement("canvas");
      this.canvas.style.width = "100%";
      this.canvas.style.height = "100%";
      this.canvas.style.display = "block";
      this.canvas.setAttribute("aria-label", "Interactive 2.5D Stirred-Tank Bioreactor Schematic");
      this.container.appendChild(this.canvas);
      this.ctx = this.canvas.getContext("2d");
      this._impellerAngle = 0;
      for (let i = 0; i < 80; i++) {
        this._bubbleSeeds.push({
          yFrac: Math.random(),
          xNorm: (Math.random() - 0.5) * 1.4,
          speedScale: 0.7 + 0.6 * Math.random(),
          r: 1.8 + 2.2 * Math.random(),
        });
      }
    }

    _bindInteractions() {
      const dom = this.backend === "webgl" ? this.renderer.domElement : this.canvas;
      dom.style.cursor = "grab";

      dom.addEventListener("pointerdown", (e) => {
        this.orbit.dragging = true;
        this.orbit.lastX = e.clientX;
        this.orbit.lastY = e.clientY;
        dom.style.cursor = "grabbing";
        if (dom.setPointerCapture) dom.setPointerCapture(e.pointerId);
      });

      dom.addEventListener("pointermove", (e) => {
        if (!this.orbit.dragging) return;
        const dx = e.clientX - this.orbit.lastX;
        const dy = e.clientY - this.orbit.lastY;
        this.orbit.lastX = e.clientX;
        this.orbit.lastY = e.clientY;
        this.orbit.theta -= dx * 0.009;
        this.orbit.phi = clamp(this.orbit.phi - dy * 0.008, 0.35, 2.25);
        this._updateCamera();
      });

      const endDrag = () => {
        this.orbit.dragging = false;
        dom.style.cursor = "grab";
      };
      dom.addEventListener("pointerup", endDrag);
      dom.addEventListener("pointercancel", endDrag);

      dom.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          this.orbit.radius = clamp(this.orbit.radius + e.deltaY * 0.004, 3.8, 9.5);
          this._updateCamera();
        },
        { passive: false }
      );
    }

    _updateCamera() {
      if (this.backend !== "webgl") return;
      const { theta, phi, radius, targetY } = this.orbit;
      const x = radius * Math.sin(phi) * Math.sin(theta);
      const y = targetY + radius * Math.cos(phi);
      const z = radius * Math.sin(phi) * Math.cos(theta);
      this.camera.position.set(x, y, z);
      this.camera.lookAt(0, targetY, 0);
    }

    resetCamera() {
      this.orbit.theta = 0.55;
      this.orbit.phi = 1.22;
      this.orbit.radius = 6.2;
      this._updateCamera();
    }

    setScale(scaleL) {
      this.scaleL = scaleL;
      if (this.backend === "webgl" && this.industrialGroup) {
        this.industrialGroup.visible = scaleL >= 500;
      }
    }

    setTheme(theme) {
      this.theme = theme;
      if (this.backend === "webgl" && this.ambientLight) {
        if (theme === "light") {
          this.ambientLight.intensity = 0.92;
          this.keyLight.intensity = 1.05;
        } else {
          this.ambientLight.intensity = 0.72;
          this.keyLight.intensity = 0.92;
        }
      }
    }

    setMotionPaused(paused) {
      this.motionPaused = Boolean(paused);
    }

    updateState(nextState) {
      Object.assign(this.state, nextState);
    }

    _onResize() {
      const w = Math.max(200, this.container.clientWidth);
      const h = Math.max(200, this.container.clientHeight);
      if (this.backend === "webgl") {
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h, false);
      } else if (this.canvas) {
        const dpr = Math.min(global.devicePixelRatio || 1, 2);
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    }

    _loop(now) {
      const dt = Math.min(0.05, Math.max(0.001, (now - this._lastTime) / 1000.0));
      this._lastTime = now;
      this._renderStep(dt);
      this._rafId = global.requestAnimationFrame((t) => this._loop(t));
    }

    _renderStep(dt) {
      const s = this.state;
      const fillFrac = clamp(s.V / Math.max(0.1, s.V_max), 0.12, 0.96);
      const rpmNorm = clamp(s.rpm / (this.scaleL >= 500 ? 189.0 : 450.0), 0.2, 2.5);
      const vvmNorm = clamp(s.vvm / (this.scaleL >= 500 ? 0.21 : 1.2), 0.15, 2.5);
      const feedNorm = clamp(s.F / (0.15 * (this.scaleL / 5.0)), 0.0, 2.0);
      const xvNorm = clamp(s.X_v / 35.0, 0.0, 1.0);
      const doLow = s.DO_pct < 20.0;
      const highShear = s.v_tip > 1.75;

      if (this.backend === "webgl") {
        const brothH = this.tankH * fillFrac;
        const brothCenterY = this.tankBottomY + 0.5 * brothH + 0.04;
        this.brothCyl.scale.set(1, brothH, 1);
        this.brothCyl.position.y = brothCenterY;
        this.meniscusDisk.position.y = this.tankBottomY + brothH + 0.04;

        // Broth color deepens with X_v density, warms toward amber under high acetate/DO stress
        const rCol = 0.08 + 0.42 * xvNorm + (doLow ? 0.18 : 0.0);
        const gCol = 0.44 - 0.10 * xvNorm + (doLow ? -0.10 : 0.08);
        const bCol = 0.41 - 0.24 * xvNorm;
        this.brothMat.color.setRGB(clamp(rCol, 0, 1), clamp(gCol, 0, 1), clamp(bCol, 0, 1));
        this.brothMat.opacity = 0.36 + 0.28 * xvNorm;

        // DO probe emissive tip warns on hypoxia
        if (doLow) {
          this.doTipMat.color.setHex(0xf87171);
          this.doTipMat.emissive.setHex(0xef4444);
        } else {
          this.doTipMat.color.setHex(0x2dd4bf);
          this.doTipMat.emissive.setHex(0x0f766e);
        }

        // Shear stress ring color: teal (safe) -> amber (moderate shear) -> crimson (high lysis)
        if (s.v_tip > 2.15) {
          this.shearMat.color.setHex(0xf87171);
          this.shearMat.opacity = 0.65;
        } else if (highShear) {
          this.shearMat.color.setHex(0xf59e0b);
          this.shearMat.opacity = 0.55;
        } else {
          this.shearMat.color.setHex(0x2dd4bf);
          this.shearMat.opacity = 0.32;
        }

        if (!this.motionPaused) {
          this.impellerGroup.rotation.y += dt * 7.5 * rpmNorm;
        }

        // Sparger bubbles rising through active liquid height
        const activeCount = Math.round(clamp(45 + 95 * vvmNorm, 25, this.maxBubbles));
        const liqBot = this.tankBottomY + 0.24;
        const liqTop = this.tankBottomY + brothH + 0.02;
        const spanY = Math.max(0.3, liqTop - liqBot);
        for (let i = 0; i < this.maxBubbles; i++) {
          const seed = this._bubbleSeeds[i];
          if (i >= activeCount) {
            this.bubbleDummy.scale.set(0, 0, 0);
            this.bubbleDummy.updateMatrix();
            this.bubbleMesh.setMatrixAt(i, this.bubbleDummy.matrix);
            continue;
          }
          if (!this.motionPaused) {
            seed.yFrac += dt * (0.28 + 0.25 * vvmNorm) * seed.speedScale;
            if (seed.yFrac > 1.0) seed.yFrac -= 1.0;
          }
          const swirl = seed.ang + seed.yFrac * 2.2 * rpmNorm;
          const rad = clamp(seed.rad * (0.55 + 0.5 * seed.yFrac), 0.14, this.tankR - 0.12);
          this.bubbleDummy.position.set(
            Math.cos(swirl) * rad,
            liqBot + seed.yFrac * spanY,
            Math.sin(swirl) * rad
          );
          const sc = seed.size * (0.75 + 0.4 * vvmNorm);
          this.bubbleDummy.scale.set(sc, sc, sc);
          this.bubbleDummy.updateMatrix();
          this.bubbleMesh.setMatrixAt(i, this.bubbleDummy.matrix);
        }
        this.bubbleMesh.instanceMatrix.needsUpdate = true;

        // Top feed nutrient droplet stream
        const topNozzleY = this.tankBottomY + this.tankH - 0.12;
        const dropSpan = Math.max(0.2, topNozzleY - liqTop);
        const visibleDrops = feedNorm > 0.02 ? Math.max(2, Math.round(6 * clamp(feedNorm, 0, 2))) : 0;
        for (let i = 0; i < this.dropGroup.children.length; i++) {
          const drop = this.dropGroup.children[i];
          const ds = this._dropSeeds[i];
          if (i >= visibleDrops) {
            drop.visible = false;
            continue;
          }
          drop.visible = true;
          if (!this.motionPaused) {
            ds.phase += dt * (1.2 + 0.8 * feedNorm);
            if (ds.phase > 1.0) ds.phase -= 1.0;
          }
          drop.position.set(0.42, topNozzleY - ds.phase * dropSpan, 0.18);
        }

        this.renderer.render(this.scene, this.camera);
      } else {
        this._renderCanvas2D(dt, fillFrac, rpmNorm, vvmNorm, feedNorm, xvNorm, doLow, highShear);
      }
    }

    _renderCanvas2D(dt, fillFrac, rpmNorm, vvmNorm, feedNorm, xvNorm, doLow, highShear) {
      const ctx = this.ctx;
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      ctx.clearRect(0, 0, w, h);

      const isLight = this.theme === "light";
      const bgLine = isLight ? "#9a958b" : "#3a4650";
      const steelCol = isLight ? "#4a5568" : "#94a3b8";
      const textCol = isLight ? "#181a1b" : "#e2e8f0";

      const tankW = Math.min(w * 0.44, 175);
      const tankH = Math.min(h * 0.66, 260);
      const cx = w * 0.5;
      const topY = h * 0.16;
      const leftX = cx - tankW / 2;
      const rightX = cx + tankW / 2;
      const botY = topY + tankH;

      // Industrial cutaway framing when 1000L
      if (this.scaleL >= 500) {
        ctx.fillStyle = isLight ? "rgba(100,116,139,0.22)" : "rgba(100,116,139,0.28)";
        ctx.fillRect(leftX - 8, topY + 6, tankW + 16, tankH - 12);
      }

      // Dynamic Broth Fill
      const liqH = tankH * fillFrac;
      const liqY = botY - liqH;
      const grad = ctx.createLinearGradient(cx, liqY, cx, botY);
      grad.addColorStop(0, doLow ? "rgba(248, 113, 113, 0.45)" : "rgba(45, 212, 191, 0.36)");
      grad.addColorStop(1, "rgba(245, 158, 11, " + (0.20 + 0.40 * xvNorm).toFixed(2) + ")");
      ctx.fillStyle = grad;
      ctx.fillRect(leftX + 2, liqY, tankW - 4, liqH);

      // Meniscus arc
      ctx.beginPath();
      ctx.ellipse(cx, liqY, tankW / 2 - 2, 10, 0, 0, Math.PI * 2);
      ctx.fillStyle = doLow ? "rgba(248,113,113,0.45)" : "rgba(45,212,191,0.42)";
      ctx.fill();

      // Vessel Glass Outline
      ctx.strokeStyle = steelCol;
      ctx.lineWidth = 2;
      ctx.strokeRect(leftX, topY, tankW, tankH);

      // Headplate & Motor
      ctx.fillStyle = steelCol;
      ctx.fillRect(leftX - 10, topY - 12, tankW + 20, 12);
      ctx.fillRect(cx - 18, topY - 40, 36, 28);

      // Central Shaft & Dual Impellers
      ctx.beginPath();
      ctx.moveTo(cx, topY - 12);
      ctx.lineTo(cx, botY - 16);
      ctx.strokeStyle = steelCol;
      ctx.lineWidth = 4;
      ctx.stroke();

      if (!this.motionPaused) this._impellerAngle += dt * 7.5 * rpmNorm;
      const bladeSpan = (tankW * 0.28) * Math.max(0.25, Math.abs(Math.cos(this._impellerAngle)));
      for (const iy of [botY - tankH * 0.25, botY - tankH * 0.56]) {
        ctx.fillStyle = highShear ? "#f59e0b" : "#cbd5e1";
        ctx.fillRect(cx - bladeSpan, iy - 6, bladeSpan * 2, 12);
      }

      // Sparger & Bubbles
      ctx.strokeStyle = "#2dd4bf";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - tankW * 0.25, botY - 14);
      ctx.lineTo(cx + tankW * 0.25, botY - 14);
      ctx.stroke();

      const activeBubbles = Math.round(clamp(16 + 32 * vvmNorm, 8, this._bubbleSeeds.length));
      ctx.fillStyle = isLight ? "rgba(15, 118, 110, 0.55)" : "rgba(220, 252, 247, 0.65)";
      for (let i = 0; i < activeBubbles; i++) {
        const bs = this._bubbleSeeds[i];
        if (!this.motionPaused) {
          bs.yFrac += dt * (0.28 + 0.25 * vvmNorm) * bs.speedScale;
          if (bs.yFrac > 1.0) bs.yFrac -= 1.0;
        }
        const bx = cx + bs.xNorm * (tankW * 0.36);
        const by = (botY - 16) - bs.yFrac * Math.max(10, liqH - 18);
        ctx.beginPath();
        ctx.arc(bx, by, bs.r, 0, Math.PI * 2);
        ctx.fill();
      }

      // Feed Line
      if (feedNorm > 0.02) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx + tankW * 0.28, topY);
        ctx.lineTo(cx + tankW * 0.28, liqY);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Probe indicators
      ctx.strokeStyle = bgLine;
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.fillStyle = textCol;
      ctx.fillText(this.scaleL + " L Vessel", leftX, topY - 18);
    }

    destroy() {
      if (this._rafId) global.cancelAnimationFrame(this._rafId);
      if (this._ro) this._ro.disconnect();
      if (this.renderer && this.renderer.dispose) this.renderer.dispose();
    }
  }

  global.Bioreactor3DView = Bioreactor3DView;
})(window);
