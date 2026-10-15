# Optimal Stopping Model for Preemptive TEVAR After Type B Aortic Dissection

This repository contains the computational implementation, model inputs, numerical results, figures, and manuscript associated with the study:

**Optimal Stopping Formulation Coupling Entry Tear Diameter, False Lumen Expansion, and Serial Enlargement for Preemptive TEVAR After Type B Dissection**

**Author:** Prihantini

---

## Overview

This project develops a literature-informed mathematical proof-of-concept for representing continued surveillance versus preemptive thoracic endovascular aortic repair (TEVAR) in uncomplicated type B aortic dissection (TBAD) as a state-dependent optimal-stopping problem.

The model combines potentially observable aortic morphology with model-defined latent dynamics, a continuous-time Markov regime representation, theoretical waiting and procedural intensities, and a finite-horizon Bellman decision rule.

The model is intended for mathematical and computational investigation. It is not a clinically calibrated prediction model and does not provide patient-specific treatment recommendations.

---

## Mathematical Model

The continuous state vector is

X(t) = [F(t), D(t), G(t), B(t), R(t)]

where:

- `F(t)` = false-lumen diameter
- `D(t)` = total aortic diameter
- `G(t)` = model-defined growth momentum
- `B(t)` = model-defined morphologic burden
- `R(t)` = model-defined stability reserve

`F(t)` and `D(t)` represent potentially observable CTA-derived quantities.

`G(t)`, `B(t)`, and `R(t)` are dimensionless latent mathematical states and should not be interpreted as validated biomarkers, physiological measurements, or patient-specific risk probabilities.

The model includes:

1. Literature-informed morphological normalization
2. Coupled pre-intervention ODE dynamics
3. A post-TEVAR dynamical subsystem
4. A four-state continuous-time Markov representation
5. Theoretical waiting and procedural intensities
6. Finite-horizon Bellman optimal stopping
7. Free-boundary continuation
8. Equilibrium and Jacobian stability analysis
9. Global and local sensitivity analyses
10. Stochastic input perturbation and numerical verification
11. Internal synthetic model-recovery analysis

---

## Evidence Status

No individual-level longitudinal clinical outcome dataset was used to fit, calibrate, or validate the present model.

Literature-derived morphological and temporal quantities are used as normalization or contextual anchors. Structural ODE coefficients, latent-state dynamics, Markov transition parameters, theoretical intensities, and Bellman decision parameters are provisional model components.

These quantities were not estimated from patient-level event rates, hazard ratios, treatment effects, or observed longitudinal trajectories.

The model therefore evaluates the internal behavior of an explicitly specified mathematical system rather than estimating clinical treatment effects or optimal patient-level TEVAR timing.

---

## Reference Simulation

The prespecified reference configuration included:

- Entry-tear diameter: 12 mm
- Initial false-lumen diameter: 22 mm
- Initial total aortic diameter: 41 mm
- Initial growth momentum: 0.20
- Initial morphologic burden: 0.15
- Initial stability reserve: 0.78

Under this parameterization, the model produced a first stopping boundary at **day 33**.

Across the explored entry-tear range of 6–16 mm and false-lumen range of 16–30 mm, model-derived stopping boundaries ranged from **25 to 47 days**.

These values are simulation outputs and must not be interpreted as recommended clinical treatment times.

---

## Computational Analyses

The repository includes outputs from:

- A 120-day reference simulation
- A 609-point entry-tear/false-lumen stopping surface
- Eight prespecified scenario stress tests
- 500 Latin-hypercube parameter sets
- Partial rank correlation coefficient analysis
- Local parameter perturbation analysis
- 500 stochastic input perturbations
- RK45 and DOP853 numerical verification
- Post-TEVAR equilibrium and Jacobian analysis
- An 800-state internal synthetic model-recovery experiment

The synthetic model-recovery experiment evaluates internal recoverability only and does not represent clinical predictive validation.

---

## Key Model Outputs

Under the prespecified analyses:

- Reference stopping boundary: 33 days
- Morphology-dependent stopping surface: 25–47 days
- Median stopping day across Latin-hypercube parameter sets: 31 days
- 2.5th–97.5th percentile range across Latin-hypercube sets: 11–50.5 days
- Median stopping day under stochastic input perturbation: 33 days
- 2.5th–97.5th percentile range under stochastic perturbation: 27–41 days

The strongest PRCCs with the simulated stopping day were observed for baseline waiting intensity, acute procedural penalty, and baseline procedure intensity.

All reported quantities characterize the specified mathematical model and are not patient-level effect estimates, confidence intervals, or clinical risk estimates.

---

## Repository Structure

```text
tbad-tevar-optimal-stopping-model/
├── code/
│   └── Model and figure-generation scripts
├── data/
│   └── Model inputs and parameter data
├── docs/
│   └── Reproducibility and parameter-provenance documentation
├── figures/
│   └── Manuscript figures and model visualizations
├── manuscript/
│   └── Full manuscript PDF
├── results/
│   └── Numerical model outputs
├── .gitignore
├── LICENSE
└── README.md
