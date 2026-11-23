# Reconstructed trajectories: a qualitative read (M7-3)

![Five test patients by HR, serum sodium and GCS](trajectories.png)

**What is shown.** Five test patients (rows) by three variables (columns), in clinical units, over
the 48 hours, on the frozen 10% random holdout. The model is the sweep's Ext 1+2, the variant with
the lowest mean test MAE at 10% (0.2186 over seeds 42-46; M7-2 has the paired verdict). Its 100 draws
are from seed 42's weights. The patients were picked by a fixed rule, not by eye: among the 137 test
patients with at least one held-out value and at least two visible values in all three variables,
five evenly spaced in the test split's order. Filled dots are what the model saw; open rings are the
held-out values it is scored on. Data and per-panel numbers: `trajectories.npz`,
`trajectories.json` (`diffusion.trajectories`; figure by `diffusion.trajectories_plot`).

**The numbers behind it** (test split, all 800 patients, mean absolute error in clinical units on
the held-out values; diffusion is each seed's protocol median of 100 draws):

| Variable | Held-out values | Diffusion, seeds 42-46 | Forward fill | Mean imputation |
| --- | --- | --- | --- | --- |
| HR (bpm) | 3,490 | 4.14-4.17 | 5.29 | 13.57 |
| Serum sodium (mmol/L) | 269 | 2.14-2.42 | 2.25 | 3.22 |
| GCS | 1,247 | 0.66-0.69 | 0.75 | 3.51 |

## Where diffusion tracks the dynamics

- **Densely charted vitals (HR).** With a value nearly every hour, the median runs through the visible
  points and, across a short gap, joins its two neighbours. Forward fill holds the last value for
  the whole gap. It is one hour late at every change, which shows as the staircase lagging the
  dots. This is where the diffusion model's margin is largest: 4.1-4.2 bpm against forward fill's
  5.3, the same on every seed.
- **Both sides of a gap, not only the past.** Forward fill reads only earlier values; the model reads
  the whole stay. Patient 137542's sodium falls from 142 at admission to 135 at hour 15. Forward fill
  holds 142 for those 15 hours and is 9 mmol/L off at the held-out value at hour 12 (133). The model's
  median comes down through the gap towards the next visible value and is 2.2 off.
- **Plateaus and transitions in GCS.** GCS sits on plateaus (15 for the alert patients, 3-4 for
  patient 137542) and the median sits on them exactly, as forward fill does. Where it changes,
  as in patient 139841's rise from 3 to 15 over hours 12-30, the median ramps between the visible
  values instead of waiting for the next one, and it nears the new level sooner.
- **Uncertainty where there is little to go on.** The band widens before the first visible value and
  in long gaps (patient 139841's HR over hours 0-8, every sodium panel), and narrows to almost
  nothing beside dense visible values. Of the 43 held-out values in the figure, 35 fall inside the
  5-95% band of the draws (HR 18 of 22, sodium 4 of 5, GCS 13 of 16). That is a little under 90%, on
  too few points to call the band miscalibrated.

## Where it does not

- **Single draws are not coherent trajectories.** Each draw moves 4-8 times as much from one hour to
  the next as the median does in the gaps: 6.5 bpm against 1.3 for HR, 2.2 mmol/L against 0.26 for
  sodium, 0.70 against 0.19 for GCS. A draw of sodium, which changes over days, jitters by 2 mmol/L
  every hour, and a draw of GCS takes values between the integers. The draws are plausible values
  hour by hour, not plausible patients. The smooth, coherent curve is the median's, an average over
  draws, so the per-hour spread should be read as the model's uncertainty, not as a picture of how
  the patient moved.
- **Isolated departures are not predicted.** A held-out value that breaks from both neighbours is
  missed by the model as by forward fill. Examples: patient 137542's HR of 78 at hour 14 between
  visible values of 64-65 (the median gives 65), and patient 140743's GCS of 10 at hours 12, 32, 36
  and 40 between visible 7-9s (the median gives 7.7-9.5, closer than forward fill's 7-8, but still
  low). Neither method could know, but the band also fails to reach them: these four and the
  turning point below are 5 of the 8 held-out values outside it.
- **Turning points get smoothed.** Next to a sharp peak, the median carries the trend into the gap.
  Patient 135031's HR of 86 at hour 22, held out between visible values climbing to 120 at hour 23, is
  imputed as 105 (forward fill: 95).
- **Sparse labs gain little.** With two to nine visible sodium values in 48 hours, the median stays
  close to the patient's running level. On the whole test split the model and forward fill are level
  on sodium (2.14-2.42 against 2.25; seed 46 is worse than forward fill). There is little temporal
  structure for the model to use between labs drawn once or twice a day.

## For Week 9 (M9-2)

The per-variable gains follow how densely a variable is charted: largest on HR (hourly), smaller on
GCS (every few hours), none on sodium (once or twice a day). That is the first thing to test across
all 35 variables. The draws' hour-to-hour jitter makes a fan plot of single draws misleading; plot the
median with its band, as here.
