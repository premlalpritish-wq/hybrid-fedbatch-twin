# DESIGN.md: Hybrid Mechanistic-ML Bioreactor State-Estimator (`hybrid-fedbatch-twin`)

## Design Read
Reading this as: Scientific & Bioprocess Control Console for bioprocess engineers, in a precision instrumentation language, dial ENERGY 1 / RHYTHM 2 / MOTION 1.

## Product Character & Identity
- **Name**: `hybrid-fedbatch-twin`
- **Role**: Industrial bioprocess digital twin combining Monod/Luedeking-Piret mechanistic kinetics with a Physics-Informed Neural ODE residual corrector, Continuous-Discrete EKF state estimator, and closed-loop growth-rate controller.
- **Visual Language**: High-contrast laboratory bench and fermentation control console. Grounded flat surfaces, crisp 1px borders, tabular numeric telemetry, and a purposeful split between the live vessel schematic and synchronized diagnostic trajectories.
- **Identity Motif**: Dual-trace validation signature comparing Solid Ground Truth vs Amber Hybrid Twin Estimate vs Dashed Uncorrected Mechanistic Baseline.

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
  - Hybrid Mechanistic-ML Estimate (Primary Accent): Amber Copper (`#B45309` Light / `#F59E0B` Dark)
  - Nominal Uncorrected Model: Slate Grey (`#64748B` dashed)
  - Constraint / Hypoxia / Toxicity Alert: Crimson (`#B91C1C` Light / `#F87171` Dark)

## Typography (R-06)
- **UI & Headings**: `"IBM Plex Sans", "Segoe UI", system-ui, sans-serif` chosen for clean industrial legibility across labels, controls, and narrative diagnostics.
- **Telemetry & Equations**: `"IBM Plex Mono", "Consolas", monospace` with `font-variant-numeric: tabular-nums` reserved strictly for numeric telemetry, ODE equations, and parameter tables.

## Layout & Rhythm (RHYTHM 2, MOTION 1)
- **Asymmetric Top Instrumentation Split**: 40% Interactive Bioreactor Vessel Twin Canvas + 60% Primary Process Outcomes & Operating Regime Diagnostics.
- **Synchronized Trajectory Workspace**: Tabbed multi-channel Altair charts with interactive hover tooltips, zoom/pan, and uncertainty bands.
- **Motion (MOTION 1)**: Functional hover/focus states plus an explicit Pause/Resume toggle on the bioreactor impeller and sparger animation, honoring `prefers-reduced-motion: reduce`.
- **Copywriting & Accessibility**: Zero em dashes, zero decorative emoji, zero fabricated numbers, keyboard-accessible controls with visible focus rings, and explicit Empty, Loading, and Error states.
