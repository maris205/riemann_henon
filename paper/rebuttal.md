# Response to Reviewers — Manuscript 1833442

**Title:** An Area-Preserving Hénon-Map Model for the Riemann Zeros: A Deterministic-Dynamics Approach with Quantum and Dissipative Solvers
*(previously: "The Physical Topology of Riemann Zeros: Dual Evidence from Quantum Coherence and Macroscopic Dissipation")*

**Author:** Liang Wang
**Journal:** Frontiers in Physics — Statistical and Computational Physics

---

## Cover note to the Editor

Dear Editor,

Thank you for the opportunity to submit a revised version. I am grateful to all three reviewers; their criticism was sharp and, on the central points, correct. The previous manuscript repeatedly elevated numerical observations to the level of proof and physical confirmation, mischaracterized the standard position of random matrix theory (RMT), and was written in a promotional register inappropriate for a paper that touches the Riemann Hypothesis. I have rewritten the manuscript to fix these problems.

The single most important change is a **reframing of the contribution**, made possible by the fact that the predecessor study — previously cited only as a Zenodo preprint — is now formally published:

> Wang, L. (2026). *The emergence of prime distribution from low-dimensional deterministic chaos.* **Research in Mathematics, 13(1).** https://doi.org/10.1080/27684830.2026.2684334

That published paper establishes an isomorphism between the symbolic dynamics of the Logistic map at its band-merging point and the prime sieve (including the emergence of the Hardy–Littlewood twin-prime constant as a dynamical fixed point). The present paper now follows from it logically: *given* a deterministic map that encodes the arithmetic of the sieve, the Hilbert–Pólya viewpoint asks for its **spectrum**. Because the 1D Logistic map is dissipative ($\det J \to 0$) and cannot host a unitary, time-reversible spectrum, I lift it to the 2D **area-preserving** Hénon map ($\det J = 1$) and analyze that spectrum. This gives the paper a concrete, published anchor to the primes — directly addressing Reviewers 1 and 5, who (rightly, for the old draft) found no clear connection to the actual zeros/primes.

Alongside the reframing I have: replaced all proof-level and promotional language with cautious, quantitative wording; corrected the RMT/GUE discussion to reflect the role of unfolding and of the Hardy–Littlewood corrections (citing Keating & Smith 2019); renamed the former "mathematical proof" section to a numerical determination; clarified that $\hbar_{\mathrm{eff}}$ and the $a(t)$ schedules are fitted/modeling parameters; downgraded the quartic-term/Weyl-law and the hardware-comparison claims to empirical/qualitative observations; added a table classifying every claim as Established / Numerical / Heuristic / Conjectural; added a reproducibility section; and added the references suggested by the reviewers (Keating & Smith 2019; Bishop, Aiken & Singleton 2019).

I have also been candid about scope. I **computed the standard local spectral diagnostics** the reviewers requested (NNSD, pair correlation, number variance, spectral rigidity; new §3.5, Figure 8), including a grid-refinement convergence check. One request I did **not** complete is a fully statistically controlled comparison with hardware data; I have not manufactured it, instead stating explicitly in the manuscript that it is the natural next step, with Table 1 marking that claim as qualitative only. I hope this honest scoping is acceptable as a basis for further consideration.

A point-by-point response follows. Reviewers' comments are in *italics*; my responses and the corresponding manuscript locations are in plain text.

---

## Reviewer 1

> *"…this idea remains rather vague, and at no point in the article is a clear or direct connection with the actual Riemann zeros established."*

I accept this for the previous version. The revised paper builds the connection in two explicit stages. (1) The link to the **primes** is no longer asserted but rests on a now-published result, Wang (2026, *Research in Mathematics*), which proves a symbolic-dynamics isomorphism between the Logistic band-merging map and the prime sieve. (2) The link to the **zeros** is made operational, not rhetorical: I define two concrete numerical operators (a unitary Fourier propagator and a Markovian operator) and compare their spectra directly to the first 100 nontrivial ordinates $\gamma_n$, reporting quantitative errors (MAPE $\approx 2.3\%$ and $6.5\%$ under single-point anchoring). I have also added an explicit "Scope and claims" subsection (§1.4) and Table 1, which state that I do **not** claim the operator spectrum equals the zeros — that remains conjectural. *(See revised §1.3, §1.4, §3.2–§3.6, Table 1.)*

> *"The author also claims the existence of a fundamental flaw in comparing RMT predictions with the density of the Riemann zeros. This claim is not justified. First, the comparison… requires the standard unfolding procedure, which renders the local density of zeros constant. Second, … subleading corrections … are related to the arithmetic structure of the primes, as captured by the Hardy–Littlewood conjectures. … see Keating and Smith (2019)."*

This was the most important technical error in the old draft, and it is fully corrected. I have rewritten the RMT discussion (§1.2 and §4.1) to state accurately that:
- the GUE correspondence concerns **unfolded local** statistics, obtained after removing the smooth mean density $\bar N(E)\sim (E/2\pi)\log(E/2\pi)$, so RMT never claims a finite Wigner semicircle reproduces the global counting function;
- the deviations from pure GUE are **not** failures but carry arithmetic content, equivalent to the Hardy–Littlewood twin-prime conjecture, per **Keating & Smith (2019)**, which is now cited.

I have removed every statement that RMT "fails." The contribution is reframed as a **constructive, complementary** question: whether one deterministic low-dimensional system can reproduce, within a single model and without a separate unfolding step, both the mean density and the local repulsion. The GUE surrogate comparison in Figure 6 is now explicitly described as asymmetric (globally fitted GUE vs. single-point-anchored solvers) and is stated *not* to be a test of RMT's unfolded local statistics. *(See revised §1.2, §3.3, §4.1; new reference Keating & Smith 2019.)*

> *Checklist: reference list not unbiased; statistical methods not valid; methods not sufficient for replication.*

Addressed respectively by: adding the RMT/arithmetic references (Keating & Smith 2019) and the Berry–Keating-tradition reference (Bishop et al. 2019), and updating the predecessor citation to the published version; reframing so that no statistical-significance claim is made beyond what the data support (and flagging the missing diagnostics in §4.4); and adding a dedicated **Reproducibility** subsection (§3.6) listing grid sizes, boundary conditions, propagation steps, optimizer settings, and extracted parameters, with all scripts/logs in the public repository.

---

## Reviewer 5

> *"Most of the phrases are completely unsupported or undemonstrated … and poorly argued. The contents are mostly unconnected …"*

The rewrite directly targets this. The paper now has a single logical spine (published prime–chaos isomorphism → Hilbert–Pólya → dissipation obstruction in 1D → 2D area-preserving lift → spectral comparison), each section feeds the next, and every quantitative statement is tied to a specific figure, a numerical setting, and a row in Table 1 indicating its status. Unsupported assertions have been removed or downgraded to clearly labeled heuristics.

> *"section 2.3 pretended to introduce a mathematical proof of the so-called Microscopic singularity. However, such a section is merely based on unconnected comments and not on a mathematically rigorous proof."*

Agreed. The section is renamed **"Numerical determination of the homoclinic tangency"** (§2.3). The text now states plainly that the critical parameter $a_c \approx 1.00561$ is obtained by **numerical root-finding** on the iterated unstable manifold (residual $O(10^{-5})$), and explicitly that this is *not* an analytic theorem. The figure caption says the same. *(See revised §2.3, Figure 3, Table 1 row "Numerical determination".)*

> *"a lot of physical terminology … poorly argued or just speculative … e.g. 'fatal Capacity Collapse crisis', 'ground-state cosmic constants', 'global ergodic catastrophe'."*

All such phrases have been deleted. "Capacity collapse" is now stated plainly as the cubic potential being unbounded below; "ground-state cosmic constants" is gone (the parameters are described as fitted resolution/optimization parameters); "ergodic catastrophe" is replaced by "global KAM breakup." A search of the revised manuscript confirms none of the flagged promotional terms remain.

> *"section 2.6 … a merely hypothetical behaviour of the parameter a(t) is introduced, [with] no intention to discuss the manner in which the different values of a(t) are obtained."*

§2.6 now states, for each of the three forms, exactly how it is chosen and what it models, and explicitly labels the constants ($k_1, k_2, c$) as fitted/optimization parameters rather than derived laws. I also state that the functional forms are smooth modeling choices, not unique or first-principles-derived, and that other schedules with the same leading behavior should give similar results. *(See revised §2.6.)*

---

## Reviewer 6 (Major Revision)

> **1.** *"The central claim that the Riemann zeros are the native eigenstates … is not established. … distinguish between numerical agreement, conjecture, and theorem. Statements such as 'unambiguously confirms', 'proves', 'irrefutably' … should be replaced …"*

Done throughout. The abstract, introduction, results, and conclusion now consistently say "numerical comparison," "is consistent with," "tracks," and "we do not claim a proof." The new §1.4 ("Scope and claims") and **Table 1** make the proof/observation/heuristic/conjecture distinction explicit for every statement, including the headline one ("Operator spectrum equals the Riemann zeros — Not claimed (conjectural)"). All instances of "proves / unambiguously confirms / irrefutably / stunningly" have been removed.

> **2.** *"The discussion of RMT and GUE is not sufficiently accurate. … the traditional connection … concerns unfolded local spectral statistics, not a claim that a finite-dimensional Wigner semicircle reproduces the full Riemann counting function. … avoid attacking a formulation that is not generally advocated."*

Fully addressed (see also Reviewer 1 above). §1.2 and §4.1 now present the standard RMT position correctly — unfolding, local statistics, Hardy–Littlewood subleading corrections (Keating & Smith 2019) — and I no longer attack a straw-man "global semicircle" claim. The comparison in Figure 6 is reframed as a counting-function comparison under an asymmetric fitting protocol, explicitly *not* a test of RMT's local statistics.

> **3.** *"The claim that the quartic regularization term generates the Weyl logarithmic density law is unsupported. … A detailed analytical argument is required. Alternatively, the claim should be substantially weakened and presented only as an empirical observation."*

I have taken the second option and weakened the claim. §2.5 now states that the quartic term is a **confining regularization** that makes the potential bounded below and the operator well posed, and explicitly says I do **not** have an analytic derivation that it generates the Weyl mean density. The only retained statement is empirical: with confinement, the computed mean counting function is consistent with logarithmic growth over the tested range. Table 1 lists "Quartic term *derives* the Weyl log-density — Not claimed (open)." An analytic treatment is named as future work. *(See §2.5, Table 1, §4.4.)*

> **4.** *"Several quantities are introduced as though they have fundamental physical significance, including the extracted value of ħ. … these parameters appear to function primarily as fitting or optimization parameters. … clarify the units, physical meaning, and sensitivity. Error estimates … should also be provided."*

I now call the quantity $\hbar_{\mathrm{eff}}$ and describe it as a **dimensionless effective semiclassical resolution** of the rescaled map (the phase-space cell size of the propagator), explicitly **not** the physical Planck constant, and **operationally an optimization parameter** obtained by differential evolution (§3.1, §3.6). I have added a genuine sensitivity/robustness study (new §4.2, Figure 9). Re-optimizing $\hbar_{\mathrm{eff}}$ at each setting, a fit at the $\approx 10$--$20\%$ MAPE level persists across regularization strengths ($\lambda=0.03$--$0.08$) and grids ($N=200$--$320$), with the optimal $\hbar_{\mathrm{eff}}$ shifting in $\approx 0.056$--$0.084$ — so the agreement is not specific to one resolution. I also report, candidly, that the deep sub-$3\%$ optimum is a *sharp* feature (narrower than the $0.001$ step of the local scan), i.e.\ finely tuned; and the complementary ablation (removing the quartic term inflates $\hbar_{\mathrm{eff}}$ to $\approx 0.277$ with the error stalling at MSE $\approx 17.1$). Both the robust band and the sharpness are now stated in the abstract, §3.2, §4.2, and Table 1, so the reader is not given only the best number.

> **5.** *"The section 'Mathematical Proof of the Microscopic Singularity' does not contain a mathematical proof … It appears to provide a numerical … determination of a critical parameter … The title and discussion should be revised accordingly."*

Done — see Reviewer 5 above. Renamed to "Numerical determination of the homoclinic tangency"; the text and caption now describe numerical root-finding, not a proof. *(§2.3.)*

> **6.** *"The comparison with superconducting quantum-hardware data is … overinterpreted. Similarities between error profiles do not by themselves establish a common physical mechanism. Statistical significance tests, comparisons with alternative models, and a more careful discussion of uncertainties are required …"*

Downgraded to a **qualitative** observation (§3.4). The text now states that a similarity of error profiles does **not** establish a shared mechanism, and that the two downturns have different origins (a finite-grid/finite-$\hbar_{\mathrm{eff}}$ aliasing artifact in the model vs. physical decoherence in the hardware). I explicitly note that a quantitative claim would require significance testing, alternative-model comparison, and an uncertainty budget, which I do not provide here and flag as future work. The figure caption was rewritten in the same spirit. Table 1 lists this as "Qualitative observation."

> **Additional:** *"…clearer separation between proven results, numerical observations, heuristic arguments, and conjectural interpretations. A table … would be helpful. … more information regarding numerical reproducibility …"*

Both added: **Table 1** provides exactly this classification, and the new **Reproducibility** subsection (§3.6) records grid sizes ($N=250$, $L=3.5$, $T_{\text{steps}}=300$ for the quantum solver; $200\times200$ grid for the Markovian solver), boundary conditions (periodic FFT, with a note on the resulting wrap-around artifacts), optimizer settings (differential evolution; bounds, population, tolerance), and the extracted parameters. All scripts and logs are public.

> **Additional:** *"…standard spectral diagnostics … nearest-neighbor spacing distributions, pair-correlation functions, number variance, and spectral rigidity …"*

I agree this is the right way to connect to the existing literature, and I have **computed all four diagnostics** (new §3.5, Figure 8; script `8-spectral_diagnostics.py`). Two findings are worth stating plainly:

1. **Validation on a known answer.** Applied to the first 100 true Riemann ordinates, the pipeline recovers GUE statistics (NNSD CDF-distance $0.068$ to GUE vs $0.349$ to Poisson), as it must.
2. **The model.** The correct spectral object is the set of **Floquet eigenphases** of the one-period propagator for the time-reversal-breaking drive $a(t)$ — a periodically driven chaotic system is the discrete-time analogue of a GUE Hamiltonian. These eigenphases show clear GUE-type level repulsion: NNSD distance $0.038$ to GUE vs $0.279$ to Poisson, $\langle s^2\rangle = 1.18$ (GUE $1.27$, Poisson $2.0$), the GUE correlation hole in $R_2(r)$, and logarithmic $\Sigma^2(L)$ and $\Delta_3(L)$. A grid-refinement convergence check (169 → 519 levels, forward passes only, no re-optimization) shows the GUE agreement is stable and sharpens with level count (NNSD distance to GUE down to $0.017$).

I have been careful not to oversell this. GUE statistics are **generic** to chaotic time-reversal-breaking systems, so reproducing them is a necessary consistency check, not the distinctive content of the model; the constrained, model-specific result remains the agreement with the actual zero **values** (the mean density). I also report openly that the GUE signature is cleanest in the Floquet eigenphases, while the reconstructed-energy spectrum used for the density comparison looks closer to Poisson — which I attribute to the phase-to-energy unwrapping scrambling the fine spacings, not to a loss of repulsion in the dynamics. Table 1 and §4.4 reflect exactly this status.

> **Additional:** *"The coefficient of the quartic regularization term should be justified. … whether 0.05 is theoretically derived or empirically chosen. A parameter scan … would improve credibility."*

I have **performed the parameter scan** (new §4.2, Figure 9). With $\hbar_{\mathrm{eff}}$ re-optimized at each coefficient, the quartic value is not unique: coefficients from $0.03$ to $0.08$ all yield a fit at the $\approx 8$--$21\%$ MAPE level, so $0.05$ is a convenient (empirically chosen) choice rather than a derived or privileged constant, and the result is robust to it. I state this explicitly in §2.5 and §4.2; I have not claimed the coefficient is theoretically derived.

> **Additional:** *"There is a related work M. Bishop, et al., Phys. Rev. D 99, 026012 (2019) …"*

Added and discussed. Bishop, Aiken & Singleton (2019) is now cited in §1.1 (Berry–Keating / Bender–Brody–Müller lineage and modified commutators) and in §2.5/§4.4 as a related setting in which confining/minimal-length modifications arise.

> **Presentation and Style:** *"highly promotional language … 'nature's destiny', 'ultimate topological limit', 'absolute sovereignty', 'stunningly', 'irrefutably' … should be removed or replaced with standard scientific language."*

The manuscript has been rewritten in a neutral, quantitative register. All listed phrases and their relatives have been removed. The title itself has been changed to a descriptive, non-promotional form.

---

## Summary of principal changes

1. **Reframed** around the now-published Wang (2026, *Research in Mathematics*) prime–chaos isomorphism; clear logical chain to the 2D area-preserving map.
2. **Corrected** the RMT/GUE discussion (unfolding, local statistics, Hardy–Littlewood / Keating & Smith 2019); removed all "RMT fails" language.
3. **Renamed** §2.3 to a numerical determination; removed all proof-level claims.
4. **Downgraded** the Weyl-law-from-quartic claim and the hardware comparison to empirical/qualitative observations.
5. **Clarified** $\hbar_{\mathrm{eff}}$ and the $a(t)$ schedules as fitted/modeling parameters; added a sensitivity analysis.
6. **Added** Table 1 (claim classification), a Reproducibility subsection, and the requested references (Keating & Smith 2019; Bishop et al. 2019).
7. **Computed** the standard local spectral diagnostics (NNSD, pair correlation, number variance, spectral rigidity; §3.5, Fig. 8) with a convergence check; **removed** all promotional language; changed the title; honestly scoped the remaining deferred work (statistical hardware comparison, quartic-coefficient scan).

I thank the reviewers again for criticism that materially improved the paper.

Liang Wang
