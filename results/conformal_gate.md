# The conformal gate (M8-3)

**Decision: go.** Extension 3, the studentized patient-level conformal intervals (PROJECT_PLAN §4.3), is
attempted in Week 9 as `M9-3`. Recorded Nov 26, 2026; Jake, Josh and Sankha agree.

## The rule

PROJECT_PLAN §4.3: attempt the stretch only if the combined model (Ext 1+2) is stable with test MAE
≤ ~0.24 @10% by the end of Week 8. Otherwise cut it cleanly and redirect the time to per-variable
analysis and uncertainty fan plots.

## The number it rests on

Ext 1+2's test MAE @10% in the sweep (`results/sweep.json`, tag `sweep`): random training regime, each
run's final weights after 200 epochs, scored on `EvalMasker`'s frozen 10% holdout of the 800 test
patients with the frozen 100-draw sampler, every run on the paired NVIDIA A100-SXM4-80GB with TF32
(PROJECT_PLAN §3.1, §3.3).

| Seed | Test MAE @10% (standardized) |
| --- | --- |
| 42 | 0.2178 |
| 43 | 0.2175 |
| 44 | 0.2193 |
| 45 | 0.2200 |
| 46 | 0.2183 |
| **Mean ± SE** | **0.2186 ± 0.0005** (SD 0.0010) |

Every seed is under 0.24: the largest, 0.2200, is 0.020 below the bar.

**Stable:** all five runs trained the full 200 epochs with every epoch's loss finite, and each run's
500-step health report is healthy (finite, learning, no collapse, no explosion; `summary.json` in each
run folder). The seeds span 0.0025 of MAE.

The runs' final validation MAEs (0.242–0.246) sit above 0.24, but they are not the gate's measure:
validation passes are tracking only, on the 400 validation patients with 5 draws, and a 5-draw median
is a noisier guess than the 100-draw median the test scoring takes. §4.3's gate is the test MAE above.

## Week-9 plan (`M9-3`, owner Jake)

Studentized, patient-level split-conformal intervals around Ext 1+2's posterior at 1 − α = 0.90, with
no retraining (PROJECT_PLAN §4.3; M9-3's deliverables):

- **Model:** Ext 1+2, random regime, the sweep's final weights: the best variant by §3.5's rule ("lowest
  mean test MAE, not separated from base") and the model the gate passed on.
- **Draws:** K = 50 posterior draws per patient, giving each position's mean μ̂ and spread σ̂. The
  sweep's scoring kept only each run's median, so this needs a new sampling pass that keeps μ̂ and σ̂:
  half a 100-draw pass, about 20 minutes for the 800 test patients and 10 for the 400 validation
  patients per run on the G4 (a 100-draw test pass took 39 minutes in the sweep).
- **Score and split:** s = |x − μ̂| / (σ̂ + ε), aggregated per patient; calibration and test patients
  disjoint; q̂ the finite-sample (1 − α) quantile of the calibration patients' scores; intervals
  μ̂ ± q̂σ̂.
- **Report:** coverage on held-out patients (target ≥ 90%, the patient-level guarantee), mean interval
  width, and coverage by patient missingness rate; position-level coverage only as a secondary check.
- **Settled in `M9-3` itself:** which patients calibrate (the 400 validation patients, which no run
  used to choose anything since every run keeps its final weights, or a split of the test patients);
  how a patient's scores are aggregated, and ε; which holdout's targets are covered; one seed or all five.
- **Order and timing:** after `M8-2`'s MAE-vs-inference-time plot, which is finished before the stretch
  starts (M8's coordination notes). Aim for the presentation (Dec 2) if it is ready by Dec 1, and the
  final report's uncertainty section (Dec 10) in any case; `M10-1`'s fan plots add the 90% band once it
  exists.
- **If it cannot finish:** cut it cleanly (§4.3). `M10-1`'s fan plots go ahead without the band.
