# DESIGN.md: Controlling the Living Reactor (`hybrid-fedbatch-twin` Interactive Scientific Monograph)

## Design Read
Reading this as: Interactive Scientific Monograph & Bioprocess Digital Twin for computational biologists and process engineers, in a Distill.pub / Bartosz Ciechanowski explorable explanation language, dial ENERGY 2 / RHYTHM 3 / MOTION 2.

## Product Character & Identity
- **Name**: `hybrid-fedbatch-twin`
- **Role**: Distill.pub-style interactive scientific monograph and zero-backend bioprocess digital twin combining Monod/Luedeking-Piret mass balances with an 86-parameter bootstrap Neural ODE (`GAM-86`, `B=15`), Patankar positivity-preserving integration, Continuous-Discrete Padé(6,6) EKF state estimation, and `5 L` to `1000 L` geometric scale transfer.
- **Visual Language**: Academic publication editorial reading spine paired with interactive Canvas/WebGL apparatus instruments. High-contrast warm paper (`Laboratory Alabaster`) and deep slate (`Control Room Slate`) palettes, progressive disclosure from physical intuition to formal math, and direct inline triggers choreographed to live physics.
- **Identity Motif**: Three-trace validation signature comparing Solid Teal Ground Truth vs Amber Copper Hybrid Twin Estimate vs Dashed Slate Uncorrected Mechanistic Baseline.

## Color Palette (R-29, R-25 WCAG AA Compliant)
- **Light Mode (Laboratory Alabaster, Default)**:
  - Canvas background: `#F7F6F2`
  - Elevated panel surface: `#EFECE6`
  - Border / divider: `#D5D0C5`
  - Primary ink: `#181A1B` (contrast > 13:1 on `#F7F6F2` and `#EFECE6`)
  - Secondary ink: `#4A5056` (contrast > 6.5:1 on `#F7F6F2`)
- **Dark Mode (Control Room Slate, Toggleable)**:
  - Canvas background: `#111518`
  - Elevated panel surface: `#1A2026`
  - Border / divider: `#2E3842`
  - Primary ink: `#ECEFF1` (contrast > 14:1 on `#111518`)
  - Secondary ink: `#A7B2BC` (contrast > 7:1 on `#1A2026`)
- **Semantic Trace & State Accents (Core 2-3 + 1 deliberate accent)**:
  - Ground Truth Plant: Deep Bioreactor Teal (`#0F766E` Light / `#2DD4BF` Dark)
  - Hybrid Mechanistic-ML Estimate (Primary Accent): Amber Copper (`#B45309` / `#92400E` ink Light / `#F59E0B` Dark)
  - Nominal Uncorrected Model: Slate Grey (`#475569` dashed Light / `#94A3B8` Dark)
  - Constraint / Hypoxia / Toxicity Alert: Crimson (`#B91C1C` Light / `#F87171` Dark)

## Typography & Reading Spine (R-06)
- **Long-Form Monograph Prose & Display Title**: `"IBM Plex Serif", "Georgia", Cambria, serif` inside a centered `max-width: 680px` reading spine with comfortable `line-height: 1.75` and `16.5px` body size.
- **UI Controls, Breakout Figure Chrome, & Margin Sidenotes**: `"IBM Plex Sans", "Segoe UI", system-ui, sans-serif`.
- **Telemetry, Hashes, & Numeric Tables**: `"IBM Plex Mono", "Consolas", monospace` with `font-variant-numeric: tabular-nums`.
- **Mathematical Typesetting**: KaTeX (`0.16.11`) rendering crisp equations cleanly without layout shifts.

## Layout & Choreography (RHYTHM 3, MOTION 2)
- **Three-Column Editorial Grid (`>= 1180px`)**: Sticky Left Table of Contents (`210px`) + Academic Prose Spine (`680px`) + Right Margin Sidenotes (`250px`), collapsing gracefully into inline technical notes below `1180px`.
- **Breakout Interactive Chapters**:
  1. **Section 1**: The Bioprocess Dilemma with inline `[Trigger Overflow Spike]` / `[Optimal Controlled Feed]` toggles and a 60 FPS dual-pane metabolic flux and bubble coalescence simulator.
  2. **Section 2**: Serial Hybrid Physics with an interactive KaTeX equation explorer coupled to live 3D bioreactor zone highlighting and an interactive Patankar vs. Explicit Euler positivity demonstrator.
  3. **Section 3**: The Adversarial PI Audit (`moritz_von_stosch_auditor` literature-grounded specification) with an interactive Iteration 0 -> 1 -> 2 diff stepper and `[Compare Naive vs. Audited Model]` chart toggle.
  4. **Section 4**: Geometric Scale-Up (`5 L` -> `1000 L`) with proportional hydrodynamic vessel animation and a live `B=15` bootstrap uncertainty fan chart showing Domain-of-Validity relaxation.
  5. **Section 5**: Full Unlocked Digital Twin Workbench combining the 3D WebGL vessel, actuator controls, 8-cell telemetry ledger, and 2x2 EKF charts, followed by ALCOA+ provenance and `8/8` pytest verification tables.
- **Motion (MOTION 2)**: Smooth 60 FPS offscreen-paused canvas choreography, eased 3D camera transitions when inspecting bioreactor zones, scroll-spy active TOC tracking, and explicit support for `prefers-reduced-motion: reduce`.
