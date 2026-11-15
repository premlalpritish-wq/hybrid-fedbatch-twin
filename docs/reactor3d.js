/**
 * Interactive Three.js 3D Stirred-Tank Bioreactor Assembly.
 * Couples physical vessel geometry (5 L borosilicate or 1000 L cutaway) to live simulation
 * state: V(t), X_v(t), DO%, v_tip, aeration vvm, and feed rate F(t), with Canvas 2D fallback.
 */

(function (global) {
  "use strict";

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  class Bioreactor3DView {
    constructor(containerEl, opts) {
      this.container = containerEl;
      this.opts = opts || {};
      this.theme = this.opts.theme || "light";
      this.scaleL = 5;
      this.motionPaused = Boolean(
        global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches
      );

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

      this.orbit = {
        theta: 0.54,
        phi: 1.24,
        radius: 8.0,
        targetY: 0.24,
        dragging: false,
        lastX: 0,
        lastY: 0,
      };
      this.orbitTarget = {
        theta: 0.54,
        phi: 1.24,
        radius: 8.0,
        targetY: 0.24,
      };
      this.highlightZone = "all";
      this._highlightPulse = 0.0;
      this._isVisible = true;

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
      if (global.IntersectionObserver) {
        this._io = new IntersectionObserver(
          (entries) => {
            const visible = entries.some((entry) => entry.isIntersecting);
            this._isVisible = visible;
            if (visible && !this._rafId) {
              this._lastTime = performance.now();
              this._loop(this._lastTime);
            }
          },
          { threshold: 0.02 }
        );
        this._io.observe(this.container);
      }
      this._renderStep(0.016);
      this._loop(performance.now());
    }

    _buildStudioEnvMap(THREE) {
      // Procedural studio lightboxes so borosilicate glass and 316L steel show crisp reflections.
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      pmrem.compileEquirectangularShader();
      const envScene = new THREE.Scene();
      envScene.background = new THREE.Color(0x283138);

      const panelGeo = new THREE.PlaneGeometry(8, 8);
      const overheadMat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
      const overhead = new THREE.Mesh(panelGeo, overheadMat);
      overhead.position.set(0, 7, 0);
      overhead.rotation.x = Math.PI / 2;
      envScene.add(overhead);

      const keyMat = new THREE.MeshBasicMaterial({ color: 0xfff8ed, side: THREE.DoubleSide });
      const keyPanel = new THREE.Mesh(panelGeo, keyMat);
      keyPanel.position.set(6.5, 3, 5);
      keyPanel.lookAt(0, 0, 0);
      envScene.add(keyPanel);

      const fillMat = new THREE.MeshBasicMaterial({ color: 0xd9ece8, side: THREE.DoubleSide });
      const fillPanel = new THREE.Mesh(panelGeo, fillMat);
      fillPanel.position.set(-7, 2, -4);
      fillPanel.lookAt(0, 0, 0);
      envScene.add(fillPanel);

      const target = pmrem.fromScene(envScene, 0.04);
      panelGeo.dispose();
      overheadMat.dispose();
      keyMat.dispose();
      fillMat.dispose();
      pmrem.dispose();
      return target;
    }

    _createContactShadowTexture(THREE) {
      const cv = document.createElement("canvas");
      cv.width = 128;
      cv.height = 128;
      const ctx = cv.getContext("2d");
      const g = ctx.createRadialGradient(64, 64, 6, 64, 64, 62);
      g.addColorStop(0, "rgba(17, 21, 24, 0.42)");
      g.addColorStop(0.55, "rgba(17, 21, 24, 0.16)");
      g.addColorStop(1, "rgba(17, 21, 24, 0.0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 128, 128);
      const tex = new THREE.CanvasTexture(cv);
      return tex;
    }

    _initThreeJS(THREE) {
      this.renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
      this.renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.05;
      if (THREE.sRGBEncoding !== undefined) {
        this.renderer.outputEncoding = THREE.sRGBEncoding;
      }
      this.renderer.domElement.style.width = "100%";
      this.renderer.domElement.style.height = "100%";
      this.renderer.domElement.style.display = "block";
      this.renderer.domElement.setAttribute("aria-label", "Interactive 3D Stirred-Tank Bioreactor Assembly");
      this.container.appendChild(this.renderer.domElement);
      this.backend = "webgl";

      this.scene = new THREE.Scene();
      this._envRT = this._buildStudioEnvMap(THREE);
      this.scene.environment = this._envRT.texture;

      const w = Math.max(200, this.container.clientWidth);
      const h = Math.max(200, this.container.clientHeight);
      this.camera = new THREE.PerspectiveCamera(35, w / h, 0.1, 100);

      // Ambient + key + cool fill lights required for PBR stainless and borosilicate glass.
      this.ambientLight = new THREE.AmbientLight(0xf1f5f4, 0.55);
      this.scene.add(this.ambientLight);
      this.keyLight = new THREE.DirectionalLight(0xfffdf8, 1.15);
      this.keyLight.position.set(4.8, 7.2, 5.2);
      this.scene.add(this.keyLight);
      this.rimLight = new THREE.DirectionalLight(0xbbe7e2, 0.45);
      this.rimLight.position.set(-5.0, 2.5, -4.5);
      this.scene.add(this.rimLight);

      this.root = new THREE.Group();
      this.scene.add(this.root);

      this.tankR = 0.94;
      this.tankH = 3.0;
      this.tankBottomY = -1.5;

      // Machined 316L stainless, anodized dark steel, EPDM gasket, and PTFE insulator materials.
      const steel316 = new THREE.MeshStandardMaterial({
        color: 0xb8c2cc,
        metalness: 0.88,
        roughness: 0.24,
      });
      const steelMachined = new THREE.MeshStandardMaterial({
        color: 0x8b98a6,
        metalness: 0.84,
        roughness: 0.32,
      });
      const metalDark = new THREE.MeshStandardMaterial({
        color: 0x333c44,
        metalness: 0.72,
        roughness: 0.42,
      });
      const epdmSeal = new THREE.MeshStandardMaterial({
        color: 0x1e2226,
        metalness: 0.05,
        roughness: 0.82,
      });
      const ptfeMat = new THREE.MeshStandardMaterial({
        color: 0xedeae2,
        metalness: 0.04,
        roughness: 0.48,
      });
      const copperFitting = new THREE.MeshStandardMaterial({
        color: 0xb45309,
        metalness: 0.58,
        roughness: 0.34,
      });

      // Soft ground contact shadow under pedestal
      this.shadowTex = this._createContactShadowTexture(THREE);
      this.shadowMat = new THREE.MeshBasicMaterial({
        map: this.shadowTex,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
      });
      const shadowDisk = new THREE.Mesh(new THREE.CircleGeometry(1.72, 48), this.shadowMat);
      shadowDisk.rotation.x = -Math.PI / 2;
      shadowDisk.position.y = this.tankBottomY - 0.34;
      this.root.add(shadowDisk);

      // 1. Machined Pedestal Base, 4 Elastomer Feet, and 4 Load-Cell Flexure Blocks
      const baseDeck = new THREE.Mesh(new THREE.CylinderGeometry(1.16, 1.24, 0.14, 48), metalDark);
      baseDeck.position.y = this.tankBottomY - 0.15;
      this.root.add(baseDeck);

      const baseTrim = new THREE.Mesh(new THREE.TorusGeometry(1.18, 0.018, 12, 48), steelMachined);
      baseTrim.rotation.x = Math.PI / 2;
      baseTrim.position.y = this.tankBottomY - 0.08;
      this.root.add(baseTrim);

      for (let i = 0; i < 4; i++) {
        const ang = (i * Math.PI) / 2 + Math.PI / 4;
        const fx = Math.cos(ang) * 0.96;
        const fz = Math.sin(ang) * 0.96;

        const footPad = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.13, 0.08, 20), epdmSeal);
        footPad.position.set(fx, this.tankBottomY - 0.28, fz);
        this.root.add(footPad);

        const loadCell = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.07, 0.16), copperFitting);
        loadCell.position.set(fx * 0.85, this.tankBottomY - 0.06, fz * 0.85);
        loadCell.rotation.y = -ang;
        this.root.add(loadCell);
      }

      // 2. Lower Torispherical Dish & Bottom Harvest Valve
      const bottomDish = new THREE.Mesh(
        new THREE.CylinderGeometry(this.tankR + 0.03, this.tankR * 0.86, 0.16, 48),
        steelMachined
      );
      bottomDish.position.y = this.tankBottomY;
      this.root.add(bottomDish);

      const harvestStem = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.16, 16), steel316);
      harvestStem.position.y = this.tankBottomY - 0.13;
      this.root.add(harvestStem);

      // 3. Borosilicate Glass Vessel Cylinder + Graduated Volume Scale Ticks
      const glassMat = new THREE.MeshPhysicalMaterial({
        color: 0xe6f2f0,
        transparent: true,
        opacity: 0.22,
        roughness: 0.06,
        metalness: 0.02,
        clearcoat: 1.0,
        clearcoatRoughness: 0.05,
        ior: 1.47,
        reflectivity: 0.9,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      this.glassVessel = new THREE.Mesh(
        new THREE.CylinderGeometry(this.tankR, this.tankR, this.tankH, 56, 1, true),
        glassMat
      );
      this.root.add(this.glassVessel);

      // Graduated volume markings on front outer wall
      this.tickMat = new THREE.MeshBasicMaterial({ color: 0x475569, transparent: true, opacity: 0.55 });
      for (let g = 1; g <= 5; g++) {
        const ty = this.tankBottomY + 0.1 + (g / 5.5) * (this.tankH - 0.25);
        const tickArc = new THREE.Mesh(
          new THREE.CylinderGeometry(this.tankR + 0.006, this.tankR + 0.006, 0.014, 16, 1, true, 0.32, 0.16),
          this.tickMat
        );
        tickArc.position.y = ty;
        this.root.add(tickArc);
      }

      // 4. 1000 L Stainless Steel Cutaway Jacket (visible only when scaleL >= 500)
      this.industrialGroup = new THREE.Group();
      const jacketMat = new THREE.MeshStandardMaterial({
        color: 0x748290,
        metalness: 0.84,
        roughness: 0.28,
        side: THREE.DoubleSide,
      });
      const jacketShell = new THREE.Mesh(
        new THREE.CylinderGeometry(
          this.tankR + 0.045,
          this.tankR + 0.045,
          this.tankH * 0.92,
          52,
          1,
          true,
          Math.PI * 0.62,
          Math.PI * 1.24
        ),
        jacketMat
      );
      this.industrialGroup.add(jacketShell);

      for (const ry of [-1.05, -0.35, 0.35, 1.05]) {
        const rib = new THREE.Mesh(new THREE.TorusGeometry(this.tankR + 0.06, 0.028, 14, 52), steel316);
        rib.rotation.x = Math.PI / 2;
        rib.position.y = ry;
        this.industrialGroup.add(rib);
      }

      // Side CIP / Jacket sanitary nozzles on industrial shell
      for (const ny of [-0.85, 0.85]) {
        const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 0.22, 16), steel316);
        nozzle.rotation.z = Math.PI / 2;
        nozzle.position.set(-(this.tankR + 0.12), ny, -0.2);
        this.industrialGroup.add(nozzle);

        const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.03, 18), steelMachined);
        flange.rotation.z = Math.PI / 2;
        flange.position.set(-(this.tankR + 0.22), ny, -0.2);
        this.industrialGroup.add(flange);
      }
      this.industrialGroup.visible = false;
      this.root.add(this.industrialGroup);

      // 5. Sanitary Headplate, EPDM Seal, 12 Perimeter Flange Bolts, & Agitator Motor
      const headY = this.tankBottomY + this.tankH + 0.06;
      const epdmRing = new THREE.Mesh(new THREE.TorusGeometry(this.tankR + 0.01, 0.025, 12, 48), epdmSeal);
      epdmRing.rotation.x = Math.PI / 2;
      epdmRing.position.y = headY - 0.06;
      this.root.add(epdmRing);

      const headplate = new THREE.Mesh(
        new THREE.CylinderGeometry(this.tankR + 0.08, this.tankR + 0.08, 0.12, 52),
        steel316
      );
      headplate.position.y = headY;
      this.root.add(headplate);

      const headBevel = new THREE.Mesh(
        new THREE.CylinderGeometry(this.tankR + 0.04, this.tankR + 0.08, 0.04, 52),
        steelMachined
      );
      headBevel.position.y = headY + 0.08;
      this.root.add(headBevel);

      // 12 Hex Cap Bolts around headplate flange
      const boltGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.06, 6);
      for (let i = 0; i < 12; i++) {
        const ang = (i * Math.PI * 2) / 12;
        const bolt = new THREE.Mesh(boltGeo, metalDark);
        bolt.position.set(
          Math.cos(ang) * (this.tankR + 0.02),
          headY + 0.08,
          Math.sin(ang) * (this.tankR + 0.02)
        );
        this.root.add(bolt);
      }

      // Mechanical Seal Pedestal & Ribbed Servo Drive Motor
      const sealHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.22, 0.16, 24), steelMachined);
      sealHousing.position.y = headY + 0.16;
      this.root.add(sealHousing);

      const motorBody = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.25, 0.46, 24), metalDark);
      motorBody.position.y = headY + 0.46;
      this.root.add(motorBody);

      for (let fy = 0; fy < 4; fy++) {
        const fin = new THREE.Mesh(new THREE.TorusGeometry(0.255, 0.012, 8, 28), steelMachined);
        fin.rotation.x = Math.PI / 2;
        fin.position.y = headY + 0.32 + fy * 0.09;
        this.root.add(fin);
      }

      const encoderCap = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.2, 0.1, 20), steel316);
      encoderCap.position.y = headY + 0.72;
      this.root.add(encoderCap);

      // Headplate Instrumentation Ports: Exhaust Condenser, Pressure Gauge, Feed Port
      const condenserCol = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.48, 16), steel316);
      condenserCol.position.set(-0.48, headY + 0.32, -0.32);
      this.root.add(condenserCol);
      const ventFilter = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.14, 18), ptfeMat);
      ventFilter.position.set(-0.48, headY + 0.6, -0.32);
      this.root.add(ventFilter);

      // Diaphragm Pressure Gauge (front right of headplate)
      const gaugeStem = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.18, 12), steel316);
      gaugeStem.position.set(0.54, headY + 0.16, 0.34);
      this.root.add(gaugeStem);
      const gaugeCase = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.05, 24), steel316);
      gaugeCase.rotation.x = Math.PI / 2;
      gaugeCase.rotation.z = -0.35;
      gaugeCase.position.set(0.54, headY + 0.28, 0.34);
      this.root.add(gaugeCase);
      const gaugeFace = new THREE.Mesh(
        new THREE.CircleGeometry(0.095, 24),
        new THREE.MeshBasicMaterial({ color: 0xf7f6f2 })
      );
      gaugeFace.rotation.y = 0.35;
      gaugeFace.position.set(0.54 + Math.sin(0.35) * 0.028, headY + 0.28, 0.34 + Math.cos(0.35) * 0.028);
      this.root.add(gaugeFace);

      // 6. 4 Internal Wall Baffles with Mounting Standoffs
      for (let b = 0; b < 4; b++) {
        const ang = (b * Math.PI) / 2 + Math.PI / 4;
        const bx = Math.cos(ang) * (this.tankR - 0.075);
        const bz = Math.sin(ang) * (this.tankR - 0.075);
        const baffle = new THREE.Mesh(new THREE.BoxGeometry(0.085, this.tankH * 0.86, 0.018), steel316);
        baffle.position.set(bx, 0, bz);
        baffle.rotation.y = -ang;
        this.root.add(baffle);
      }

      // 7. Dynamic Broth Cylinder + Meniscus Ring
      this.brothMat = new THREE.MeshPhysicalMaterial({
        color: 0x1e6b63,
        transparent: true,
        opacity: 0.44,
        roughness: 0.18,
        metalness: 0.02,
        clearcoat: 0.6,
        depthWrite: false,
      });
      this.brothCyl = new THREE.Mesh(
        new THREE.CylinderGeometry(this.tankR - 0.022, this.tankR - 0.022, 1.0, 48),
        this.brothMat
      );
      this.root.add(this.brothCyl);

      this.meniscusMat = new THREE.MeshBasicMaterial({
        color: 0x0f766e,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
      });
      this.meniscusDisk = new THREE.Mesh(new THREE.RingGeometry(0.06, this.tankR - 0.022, 48), this.meniscusMat);
      this.meniscusDisk.rotation.x = -Math.PI / 2;
      this.root.add(this.meniscusDisk);

      // 8. Central Shaft & Dual 6-Blade Rushton Turbines
      this.impellerGroup = new THREE.Group();
      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.044, this.tankH + 0.16, 20), steel316);
      shaft.position.y = 0.06;
      this.impellerGroup.add(shaft);

      const bladeGeo = new THREE.BoxGeometry(0.18, 0.14, 0.02);
      for (const yPos of [-0.92, -0.14]) {
        const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.07, 24), steelMachined);
        hub.position.y = yPos;
        this.impellerGroup.add(hub);

        const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.29, 0.29, 0.018, 28), steel316);
        disc.position.y = yPos;
        this.impellerGroup.add(disc);

        for (let k = 0; k < 6; k++) {
          const ang = (k * Math.PI) / 3;
          const blade = new THREE.Mesh(bladeGeo, steel316);
          blade.position.set(Math.cos(ang) * 0.275, yPos, Math.sin(ang) * 0.275);
          blade.rotation.y = -ang;
          this.impellerGroup.add(blade);
        }
      }
      this.root.add(this.impellerGroup);

      // Subtle hydrodynamic radial wake ring inside the broth around the lower impeller
      this.shearMat = new THREE.MeshBasicMaterial({
        color: 0x0f766e,
        transparent: true,
        opacity: 0.26,
        side: THREE.DoubleSide,
      });
      this.shearRing = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.5, 48), this.shearMat);
      this.shearRing.rotation.x = -Math.PI / 2;
      this.shearRing.position.y = -0.92;
      this.root.add(this.shearRing);

      // 9. Ring Sparger & Vertical Gas Supply Downcomer
      this.spargerMat = new THREE.MeshStandardMaterial({
        color: 0xb8c2cc,
        metalness: 0.85,
        roughness: 0.26,
        emissive: 0x0f766e,
        emissiveIntensity: 0.0,
      });
      const spargerRing = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.028, 12, 36), this.spargerMat);
      spargerRing.rotation.x = Math.PI / 2;
      spargerRing.position.y = this.tankBottomY + 0.22;
      this.root.add(spargerRing);

      const spargerPipe = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, this.tankH - 0.18, 12), steel316);
      spargerPipe.position.set(0, 0.08, -0.42);
      this.root.add(spargerPipe);

      // 10. Sterile Feed Port, Elbow, & Nutrient Droplet Stream
      const feedCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.08, 16), steelMachined);
      feedCollar.position.set(0.4, headY + 0.09, 0.16);
      this.root.add(feedCollar);

      this.feedPipeMat = copperFitting.clone();
      const feedPipe = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.56, 12), this.feedPipeMat);
      feedPipe.position.set(0.4, headY + 0.06, 0.16);
      this.root.add(feedPipe);

      this.dropGroup = new THREE.Group();
      const dropGeo = new THREE.SphereGeometry(0.026, 8, 8);
      for (let i = 0; i < 12; i++) {
        const drop = new THREE.Mesh(dropGeo, this.feedPipeMat);
        this.dropGroup.add(drop);
        this._dropSeeds.push({ phase: i / 12 });
      }
      this.root.add(this.dropGroup);

      // 11. In-Situ PAT Sensor Probes (Capacitance Biomass + Optical DO with Sensing Cap)
      const capCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.1, 16), steelMachined);
      capCollar.position.set(-0.5, headY + 0.1, 0.28);
      this.root.add(capCollar);
      this.capProbeMat = steel316.clone();
      const capProbe = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 2.38, 14), this.capProbeMat);
      capProbe.position.set(-0.5, 0.54, 0.28);
      this.root.add(capProbe);
      const capTip = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.024, 0.18, 14), ptfeMat);
      capTip.position.set(-0.5, -0.72, 0.28);
      this.root.add(capTip);

      const doCollar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.1, 16), steelMachined);
      doCollar.position.set(-0.28, headY + 0.1, -0.46);
      this.root.add(doCollar);
      const doProbeBody = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 2.26, 14), steel316);
      doProbeBody.position.set(-0.28, 0.58, -0.46);
      this.root.add(doProbeBody);

      this.doTipMat = new THREE.MeshStandardMaterial({
        color: 0x0f766e,
        emissive: 0x0f766e,
        emissiveIntensity: 0.35,
        roughness: 0.3,
      });
      const doProbeTip = new THREE.Mesh(new THREE.SphereGeometry(0.044, 14, 14), this.doTipMat);
      doProbeTip.position.set(-0.28, -0.58, -0.46);
      this.root.add(doProbeTip);

      // 12. Instanced Rising Sparger Micro-Bubbles
      this.maxBubbles = 220;
      const bubGeo = new THREE.SphereGeometry(0.021, 8, 8);
      this.bubMat = new THREE.MeshBasicMaterial({ color: 0xe2f5f2, transparent: true, opacity: 0.72 });
      this.bubbleMesh = new THREE.InstancedMesh(bubGeo, this.bubMat, this.maxBubbles);
      this.bubbleDummy = new THREE.Object3D();
      for (let i = 0; i < this.maxBubbles; i++) {
        const ang = (i * 2.39996) % (Math.PI * 2);
        const rad = 0.25 + (0.58 * ((i * 17) % 100)) / 100;
        this._bubbleSeeds.push({
          yFrac: Math.random(),
          ang: ang,
          rad: rad,
          speedScale: 0.75 + 0.5 * Math.random(),
          size: 0.65 + 0.7 * Math.random(),
        });
      }
      this.root.add(this.bubbleMesh);

      // 13. Equation-Explorer Physical Zone Highlight Torus
      this.zoneHighlightMat = new THREE.MeshBasicMaterial({
        color: 0xb45309,
        transparent: true,
        opacity: 0.0,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      this.zoneHighlightRing = new THREE.Mesh(
        new THREE.TorusGeometry(this.tankR + 0.09, 0.024, 12, 64),
        this.zoneHighlightMat
      );
      this.zoneHighlightRing.rotation.x = Math.PI / 2;
      this.zoneHighlightRing.visible = false;
      this.root.add(this.zoneHighlightRing);

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
        this.orbitTarget.theta = this.orbit.theta;
        this.orbitTarget.phi = this.orbit.phi;
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
          this.orbit.radius = clamp(this.orbit.radius + e.deltaY * 0.004, 4.8, 10.5);
          this.orbitTarget.radius = this.orbit.radius;
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
      this.setHighlightZone("all");
      this.orbit.theta = 0.54;
      this.orbit.phi = 1.24;
      this.orbit.radius = 8.0;
      this.orbit.targetY = 0.24;
      this._updateCamera();
    }

    setHighlightZone(zoneName) {
      const z = zoneName || "all";
      this.highlightZone = z;
      const presets = {
        all: { theta: 0.54, phi: 1.24, radius: 8.0, targetY: 0.24 },
        growth: { theta: 0.35, phi: 1.30, radius: 6.6, targetY: -0.20 },
        sparger: { theta: 0.72, phi: 1.42, radius: 6.2, targetY: -0.95 },
        impeller: { theta: -0.45, phi: 1.32, radius: 6.0, targetY: -0.55 },
        feed: { theta: 0.65, phi: 1.02, radius: 6.4, targetY: 1.05 },
        patankar: { theta: 0.54, phi: 1.24, radius: 7.4, targetY: 0.0 },
      };
      const target = presets[z] || presets.all;
      this.orbitTarget.theta = target.theta;
      this.orbitTarget.phi = target.phi;
      this.orbitTarget.radius = target.radius;
      this.orbitTarget.targetY = target.targetY;
      if (this.motionPaused) {
        this.orbit.theta = target.theta;
        this.orbit.phi = target.phi;
        this.orbit.radius = target.radius;
        this.orbit.targetY = target.targetY;
        this._updateCamera();
      }
      this._renderStep(0.016);
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
          this.ambientLight.intensity = 0.72;
          this.keyLight.intensity = 1.15;
          if (this.shadowMat) this.shadowMat.opacity = 0.42;
          if (this.tickMat) {
            this.tickMat.color.setHex(0x475569);
            this.tickMat.opacity = 0.55;
          }
        } else {
          this.ambientLight.intensity = 0.52;
          this.keyLight.intensity = 1.05;
          if (this.shadowMat) this.shadowMat.opacity = 0.65;
          if (this.tickMat) {
            this.tickMat.color.setHex(0x94a3b8);
            this.tickMat.opacity = 0.45;
          }
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
        this._updateCamera();
      } else if (this.canvas) {
        const dpr = Math.min(global.devicePixelRatio || 1, 2);
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    }

    _loop(now) {
      if (!this._isVisible) {
        this._rafId = null;
        return;
      }
      const dt = Math.min(0.05, Math.max(0.001, (now - this._lastTime) / 1000.0));
      this._lastTime = now;
      this._renderStep(dt);
      this._rafId = global.requestAnimationFrame((t) => this._loop(t));
    }

    _renderStep(dt) {
      const s = this.state;
      const fillFrac = clamp(s.V / Math.max(0.1, s.V_max), 0.12, 0.96);
      const rpmNorm = clamp(s.rpm / (this.scaleL >= 500 ? 134.8 : 450.0), 0.2, 2.5);
      const vvmNorm = clamp(s.vvm / (this.scaleL >= 500 ? 0.2052 : 1.2), 0.15, 2.5);
      const feedNorm = clamp(s.F / (0.15 * (this.scaleL / 5.0)), 0.0, 2.0);
      const xvNorm = clamp(s.X_v / 35.0, 0.0, 1.0);
      const doLow = s.DO_pct < 20.0;
      const highShear = s.v_tip > 1.65;

      if (this.backend === "webgl") {
        if (!this.orbit.dragging) {
          const ease = 1.0 - Math.exp(-8.0 * dt);
          const dT = Math.abs(this.orbitTarget.theta - this.orbit.theta);
          const dP = Math.abs(this.orbitTarget.phi - this.orbit.phi);
          const dR = Math.abs(this.orbitTarget.radius - this.orbit.radius);
          const dY = Math.abs(this.orbitTarget.targetY - this.orbit.targetY);
          if (dT + dP + dR + dY > 0.0005) {
            this.orbit.theta += (this.orbitTarget.theta - this.orbit.theta) * ease;
            this.orbit.phi += (this.orbitTarget.phi - this.orbit.phi) * ease;
            this.orbit.radius += (this.orbitTarget.radius - this.orbit.radius) * ease;
            this.orbit.targetY += (this.orbitTarget.targetY - this.orbit.targetY) * ease;
            this._updateCamera();
          }
        }

        this._highlightPulse += dt * 4.0;
        const hz = this.highlightZone || "all";

        const brothH = this.tankH * fillFrac;
        const brothCenterY = this.tankBottomY + 0.5 * brothH + 0.04;
        this.brothCyl.scale.set(1, brothH, 1);
        this.brothCyl.position.y = brothCenterY;
        this.meniscusDisk.position.y = this.tankBottomY + brothH + 0.04;

        // Broth turbidity deepens with biomass X_v and shifts warmer under oxygen limitation.
        const rCol = 0.07 + 0.44 * xvNorm + (doLow ? 0.16 : 0.0);
        const gCol = 0.43 - 0.11 * xvNorm + (doLow ? -0.1 : 0.06);
        const bCol = 0.40 - 0.24 * xvNorm;
        this.brothMat.color.setRGB(clamp(rCol, 0, 1), clamp(gCol, 0, 1), clamp(bCol, 0, 1));
        this.brothMat.opacity = (0.34 + 0.28 * xvNorm) * (hz === "growth" ? 1.18 : 1.0);

        if (doLow) {
          this.doTipMat.color.setHex(0xb91c1c);
          this.doTipMat.emissive.setHex(0x991b1b);
          this.doTipMat.emissiveIntensity = hz === "sparger" ? 0.9 : 0.35;
        } else {
          this.doTipMat.color.setHex(0x0f766e);
          this.doTipMat.emissive.setHex(0x0d6861);
          this.doTipMat.emissiveIntensity = hz === "sparger" ? 0.85 : 0.35;
        }

        if (this.spargerMat) {
          this.spargerMat.emissiveIntensity = hz === "sparger" ? 0.65 : 0.0;
        }
        if (this.capProbeMat) {
          this.capProbeMat.emissive.setHex(hz === "growth" ? 0xb45309 : 0x000000);
          this.capProbeMat.emissiveIntensity = hz === "growth" ? 0.45 : 0.0;
        }
        if (this.feedPipeMat) {
          this.feedPipeMat.emissive.setHex(hz === "feed" ? 0xf59e0b : 0x000000);
          this.feedPipeMat.emissiveIntensity = hz === "feed" ? 0.55 : 0.0;
        }

        if (hz === "impeller" || s.v_tip > 2.45) {
          this.shearMat.color.setHex(hz === "impeller" ? 0xb45309 : 0xb91c1c);
          this.shearMat.opacity = hz === "impeller" ? 0.56 + 0.14 * Math.sin(this._highlightPulse) : 0.46;
        } else if (highShear) {
          this.shearMat.color.setHex(0xb45309);
          this.shearMat.opacity = 0.38;
        } else {
          this.shearMat.color.setHex(0x0f766e);
          this.shearMat.opacity = 0.22;
        }

        if (this.zoneHighlightRing && this.zoneHighlightMat) {
          if (hz === "all") {
            this.zoneHighlightRing.visible = false;
          } else {
            this.zoneHighlightRing.visible = true;
            const ringSpecs = {
              growth: { y: -0.35, color: 0x0f766e, scale: 1.0 },
              sparger: { y: -1.22, color: 0x0f766e, scale: 0.72 },
              impeller: { y: -0.92, color: 0xb45309, scale: 0.64 },
              feed: { y: this.tankBottomY + brothH + 0.06, color: 0xb45309, scale: 0.96 },
              patankar: { y: this.tankBottomY + brothH + 0.04, color: 0x0f766e, scale: 0.98 },
            };
            const rs = ringSpecs[hz] || ringSpecs.growth;
            this.zoneHighlightRing.position.y = rs.y;
            this.zoneHighlightRing.scale.set(rs.scale, rs.scale, 1);
            this.zoneHighlightMat.color.setHex(rs.color);
            this.zoneHighlightMat.opacity = 0.68 + 0.18 * Math.sin(this._highlightPulse);
          }
        }

        if (!this.motionPaused) {
          this.impellerGroup.rotation.y += dt * 7.5 * rpmNorm;
        }

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
          drop.position.set(0.4, topNozzleY - ds.phase * dropSpan, 0.16);
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
      const steelCol = isLight ? "#4a5056" : "#a7b2bc";
      const textCol = isLight ? "#181a1b" : "#eceff1";
      const gtCol = isLight ? "#0f766e" : "#2dd4bf";
      const hybCol = isLight ? "#b45309" : "#f59e0b";
      const alertCol = isLight ? "#b91c1c" : "#f87171";

      const tankW = Math.min(w * 0.44, 175);
      const tankH = Math.min(h * 0.66, 250);
      const cx = w * 0.5;
      const topY = h * 0.17;
      const leftX = cx - tankW / 2;
      const botY = topY + tankH;

      if (this.scaleL >= 500) {
        ctx.fillStyle = isLight ? "rgba(74, 80, 86, 0.16)" : "rgba(167, 178, 188, 0.18)";
        ctx.fillRect(leftX - 8, topY + 6, tankW + 16, tankH - 12);
      }

      const liqH = tankH * fillFrac;
      const liqY = botY - liqH;
      ctx.fillStyle = doLow
        ? "rgba(185, 28, 28, 0.34)"
        : "rgba(15, 118, 110, " + (0.24 + 0.32 * xvNorm).toFixed(2) + ")";
      ctx.fillRect(leftX + 2, liqY, tankW - 4, liqH);

      ctx.beginPath();
      ctx.ellipse(cx, liqY, tankW / 2 - 2, 8, 0, 0, Math.PI * 2);
      ctx.fillStyle = doLow ? alertCol : gtCol;
      ctx.globalAlpha = 0.38;
      ctx.fill();
      ctx.globalAlpha = 1.0;

      ctx.strokeStyle = steelCol;
      ctx.lineWidth = 2;
      ctx.strokeRect(leftX, topY, tankW, tankH);

      ctx.fillStyle = steelCol;
      ctx.fillRect(leftX - 10, topY - 12, tankW + 20, 12);
      ctx.fillRect(cx - 18, topY - 40, 36, 28);

      ctx.beginPath();
      ctx.moveTo(cx, topY - 12);
      ctx.lineTo(cx, botY - 16);
      ctx.strokeStyle = steelCol;
      ctx.lineWidth = 4;
      ctx.stroke();

      if (!this.motionPaused) this._impellerAngle += dt * 7.5 * rpmNorm;
      const bladeSpan = tankW * 0.28 * Math.max(0.25, Math.abs(Math.cos(this._impellerAngle)));
      for (const iy of [botY - tankH * 0.25, botY - tankH * 0.56]) {
        ctx.fillStyle = highShear ? hybCol : steelCol;
        ctx.fillRect(cx - bladeSpan, iy - 5, bladeSpan * 2, 10);
      }

      ctx.strokeStyle = gtCol;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - tankW * 0.25, botY - 14);
      ctx.lineTo(cx + tankW * 0.25, botY - 14);
      ctx.stroke();

      const activeBubbles = Math.round(clamp(16 + 32 * vvmNorm, 8, this._bubbleSeeds.length));
      ctx.fillStyle = isLight ? "rgba(15, 118, 110, 0.55)" : "rgba(226, 245, 242, 0.68)";
      for (let i = 0; i < activeBubbles; i++) {
        const bs = this._bubbleSeeds[i];
        if (!this.motionPaused) {
          bs.yFrac += dt * (0.28 + 0.25 * vvmNorm) * bs.speedScale;
          if (bs.yFrac > 1.0) bs.yFrac -= 1.0;
        }
        const bx = cx + bs.xNorm * (tankW * 0.36);
        const by = botY - 16 - bs.yFrac * Math.max(10, liqH - 18);
        ctx.beginPath();
        ctx.arc(bx, by, bs.r, 0, Math.PI * 2);
        ctx.fill();
      }

      if (feedNorm > 0.02) {
        ctx.strokeStyle = hybCol;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx + tankW * 0.28, topY);
        ctx.lineTo(cx + tankW * 0.28, liqY);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.fillStyle = textCol;
      ctx.fillText(this.scaleL + " L Vessel", leftX, topY - 18);

      if (this.highlightZone && this.highlightZone !== "all") {
        const zoneBoxes = {
          growth: { y: liqY + 8, h: Math.max(24, liqH - 18), col: gtCol, label: "ZONE: BIOMASS GROWTH (mu)" },
          sparger: { y: botY - 30, h: 28, col: gtCol, label: "ZONE: SPARGER / O2 TRANSFER (k_La)" },
          impeller: { y: botY - tankH * 0.62, h: tankH * 0.42, col: hybCol, label: "ZONE: IMPELLER SHEAR (Delta k_d)" },
          feed: { y: topY - 4, h: 36, col: hybCol, label: "ZONE: FEED & DILUTION (D = F/V)" },
          patankar: { y: liqY - 10, h: 20, col: gtCol, label: "ZONE: BOUNDARY POSITIVITY phi_1(z)" },
        };
        const zb = zoneBoxes[this.highlightZone];
        if (zb) {
          ctx.save();
          ctx.strokeStyle = zb.col;
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 3]);
          ctx.strokeRect(leftX - 10, zb.y, tankW + 20, zb.h);
          ctx.setLineDash([]);
          ctx.fillStyle = zb.col;
          ctx.font = 'bold 10.5px "IBM Plex Mono", monospace';
          ctx.fillText(zb.label, Math.max(8, leftX - 10), Math.max(14, zb.y - 5));
          ctx.restore();
        }
      }
    }

    destroy() {
      if (this._rafId) global.cancelAnimationFrame(this._rafId);
      if (this._ro) this._ro.disconnect();
      if (this._io) this._io.disconnect();
      if (this._envRT && this._envRT.dispose) this._envRT.dispose();
      if (this.shadowTex && this.shadowTex.dispose) this.shadowTex.dispose();
      if (this.renderer && this.renderer.dispose) this.renderer.dispose();
    }
  }

  global.Bioreactor3DView = Bioreactor3DView;
})(window);
