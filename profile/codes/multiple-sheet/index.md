---
title: Multiple-sheet model
description: Short-pulse electron-beam trajectories, current-limit comparisons, and reproducible Python code.
category: Scientific computing
date: 2026-10-25
period: Updated October 2026
---

A one-dimensional electrostatic model that represents an electron pulse as a sequence of charged sheets. The code accompanies my work on [short-pulse beam dynamics in a drift space](https://doi.org/10.1109/TED.2026.3670804).

[Download code and results (ZIP)](/downloads/multiple-sheet.zip) · [Browse on GitHub](https://github.com/yvesheri4/starter-hugo-academic/tree/main/code/multiple-sheet)

## How the pulse evolves

Each curve below follows one sheet as it moves through the gap. The separation of arrival times shows how space charge spreads the pulse during transport. The three panels compare square-top, trapezoidal, and Gaussian injection profiles.

{{< figure src="/media/multiple-sheet/trajectories.png" link="/media/multiple-sheet/trajectories.png" alt="Electron-sheet trajectories through a drift space for square-top, trapezoidal, and Gaussian pulses" caption="New demonstration using the archived numerical core: 80 sheets per profile, 1,200 time samples, and normalized pulse-duration parameter k = 0.1. Click to enlarge." >}}

This example uses half the archived critical sheet charge for each profile. All 80 sheets reach the downstream boundary in each case, and the active charged sheets retain their order. It is a compact demonstration, rather than an exact reproduction of the publication figures or a convergence study.

## Current-limit comparison

Shorter pulses can carry a larger normalized limiting current density in the saved comparison. The plot brings together the Multiple-sheet and XOOPIC values for the three pulse shapes.

{{< figure src="/media/multiple-sheet/current-limits.png" link="/media/multiple-sheet/current-limits.png" alt="Archived normalized limiting current versus pulse duration comparing Multiple-sheet and PIC results" caption="Archived research values from the February 2026 notebook, replotted for this page. PIC error bars retain the notebook’s recorded values. Click to enlarge." >}}

These values are included as a CSV table. They are saved research results, not a new parameter scan performed by the demonstration script.

## Run it yourself

Use Python 3.10 or newer, from the extracted `multiple-sheet` folder:

```sh
python -m pip install -r requirements.txt
python run_example.py --check
python run_example.py --output results
```

The package includes the original numerical core, the example driver, both plots, sampled trajectories, arrival data, and the current-limit comparison table. The README describes normalization, reproduction, and limitations; the provenance file identifies the source notebook and extracted cells.

The archived solver assumes ordered sheets and does not re-sort them after crossing. The example checks this condition and stops if it fails. Its ballistic and profile-normalization checks are useful consistency checks, not a substitute for numerical convergence testing.

- [Browse the source and documentation](https://github.com/yvesheri4/starter-hugo-academic/tree/main/code/multiple-sheet)
- [Current-limit data](https://github.com/yvesheri4/starter-hugo-academic/blob/main/code/multiple-sheet/data/current_limits.csv)
- [Demonstration results](https://github.com/yvesheri4/starter-hugo-academic/tree/main/code/multiple-sheet/results)
- [Back to Codes](/codes/)
