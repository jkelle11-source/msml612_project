# Milestones Overview — Missingness-Aware Diffusion Imputation (MSML612)

Ten weekly GitHub milestones (`M1`–`M10`), one Markdown file each. This index is the map: the deliverable calendar, the per-person load balance, the soft ownership tracks, and the critical path. Each milestone file holds the full issue detail (Assignee / Labels / Blocked by / Blocks / Context / Deliverables / Definition of Done).

**Team:** Jake `@jkelle11-source` · Josh `@JoshOlu` · Sankha `@SankhaS`
**Duration:** ~10 weeks (Sep 30 – Dec 10, 2026) · **Dataset:** PhysioNet/CinC Challenge 2012

---

## Milestone calendar

| # | Milestone | Week | Dates | Deliverable |
|---|---|---|---|---|
| [M1](Milestone_01_Setup_Data_Proposal.md) | Setup, Data & Proposal | 1 | Sep 30 – Oct 7 | ▶ **Proposal (Wed Oct 8)** |
| [M2](Milestone_02_Diffusion_Baselines.md) | Diffusion Process & Baselines | 2 | Oct 8 – Oct 14 | — |
| [M3](Milestone_03_Base_Reproduction.md) | Base CSDI Reproduction | 3 | Oct 15 – Oct 21 | — |
| [M4](Milestone_04_Gate_Ext1_Interim.md) | Gate, Extension 1 & Interim | 4 | Oct 22 – Oct 28 | ▶ **Interim Report (Wed Oct 29)** |
| [M5](Milestone_05_Ext1_Complete_Ext2_Start.md) | Ext 1 Complete, Ext 2 Start | 5 | Oct 29 – Nov 4 | — |
| [M6](Milestone_06_Combined_Model_Harness.md) | Combined Model & Harness | 6 | Nov 5 – Nov 11 | — |
| [M7](Milestone_07_Paired_Seed_Sweep.md) | Full Paired 5-Seed Training | 7 | Nov 12 – Nov 18 | — |
| [M8](Milestone_08_Ablation_Runtime.md) | Ablation & Runtime | 8 | Nov 19 – Nov 25 | *(conformal gate check)* |
| [M9](Milestone_09_FinalEval_Stretch_Presentation.md) | Final Eval, Stretch & Presentation | 9 | Nov 26 – Dec 1 | ▶ **Presentation (Tue Dec 2)** |
| [M10](Milestone_10_Final_Report.md) | Final Report & Code Cleanup | 10 | Dec 2 – Dec 9 | ▶ **Final Report (Wed Dec 10)** |

US Thanksgiving falls Thu Nov 26; Week 9 is treated as a light/buffer week (§6). The reproduction **gate** (§4.4: ≤ ~0.24 MAE @10%, ≥3 seeds) is the hard prerequisite in M4 before any extension delta is trusted.

---

## Per-person load

Issues owned (including co-owned). The five shared kickoff/writing issues (`M1-4`, `M1-5`, `M4-3`, `M8-3`, `M9-4`) are jointly owned by all three.

| Person | Issues | Heavy weeks | Primary surface |
|---|---|---|---|
| **Jake** `@jkelle11-source` | 11 | W1, W2, W5, W7 | Data/diffusion engine, TSLO + Ext 2, seed sweep, latency, conformal, reproducibility |
| **Josh** `@JoshOlu` | 10 | W2, W3, W4 | Fork, denoiser, Ext 1, combined model, qualitative + per-variable |
| **Sankha** `@SankhaS` | 10 | W3, W6, W7, W8 | Maskers, baselines, harness, headline table, ablation, final eval |

**Watch item:** Josh's W2–W4 is the tightest stretch (denoiser → lead reproduction → gate + Ext 1). It's mitigated by moving Ext 2 to Jake and pairing Jake on reproduction/gate — but if Josh has lighter availability, swap his track with Sankha's to even it further.

---

## Critical path

`M1-1`/`M1-2` (data + fork) → `M2-1`/`M2-2` (loop + denoiser) → **`M3-1` (reproduction, paired)** → **`M4-1` (gate, paired)** → `M5-2` (Ext 2) + `M4-2` (Ext 1) → `M6-1` (combined) + `M6-2` (harness) → **`M7-1` (20-run sweep)** → `M7-2` (headline table) → `M8-1`/`M8-2` (ablation + runtime) → `M9-1` (final table) → `M10-2` (final report).

Everything downstream of `M4-1` assumes a **passed gate**. If the gate fails by Oct 29, the interim (`M4-3`) says so honestly and Week 5 becomes baseline debugging (§4.4); Extension 3 (conformal) is an explicit stretch gated at the end of Week 8 (`M8-3`) and can be cut cleanly without weakening the core project.

---