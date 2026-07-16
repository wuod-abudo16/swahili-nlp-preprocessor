Classifying Women's Economic Narratives: A Small-Scale, Explainable NLP Prototype

Submitted by: Samson Odhiambo
For: FemDigiNomics — NLP and AI Intern application
Files in this sample: `annotated_narratives.csv` (data), `classify_narratives.py` (code), `run_output.txt` (a real run of the pipeline against this data), this document (methodology and discussion)

---

## 1. Objective

FemDigiNomics needs language tools that reflect how women *actually* describe their economic realities — not how a standard, English-first NLP toolkit assumes they will. This sample builds a small, end-to-end prototype of the kind of workflow described in the internship's responsibilities: collecting and annotating code-mixed Swahili/English/Sheng narratives, defining category taxonomy around lived financial experience (savings, debt, chama obligations,
business risk, care work, health shocks, lending readiness), and testing a first-pass classifier — while being explicit about what a 25-narrative prototype can and cannot tell us.

## 2. The data and annotation approach

`annotated_narratives.csv` contains 25 short, first-person narratives written in the register women commonly use in these conversations: Swahili with embedded English financial terms, Sheng-adjacent phrasing, and code-switching mid-sentence. Each narrative carries:

- `categories` — one or more of the seven taxonomy labels, or `none` for narratives that are financially stable and shouldn't be forced into a risk category (N022 exists specifically to guard against this failure mode).
- `annotator_notes` — a short justification for the label choice. This is the single most important column in the file. In real annotation work, disagreements between annotators almost always trace back to an unstated assumption; writing the reasoning down at annotation time, not after the fact, is what makes the label set auditable later.

Two design choices are deliberate and worth flagging to a reviewer:

1. Multi-label, not multi-class. Narratives like N003 (a child's malaria episode draining school-fee savings) are simultaneously a health shock, a care-work event, and a savings event. Collapsing that into one label would discard exactly the co-occurrence pattern a support-pathway system needs to detect — e.g., that health shocks are a leading cause of savings depletion, not just a standalone category.
2. A "none" / neutral class is included on purpose. A classifier trained only on distress narratives will learn to see distress everywhere. N022 and N023 (stable business, successful savings habit) exist to keep the model — and any human reviewer — honest about base rates.

3. Methodology

The pipeline (`classify_narratives.py`) uses TF-IDF features (unigrams and bigrams) with a One-vs-Rest logistic regression classifier — a deliberately simple, fully inspectable model rather than a large pretrained language model, for two reasons specific to this stage of the project:

- With a handful of annotated examples per category, a large model will either refuse to differentiate categories meaningfully or will latch onto spurious correlations that look fine in a demo and fail in the field. A linear model's weights can be read directly, which matters for the next point.
- Explainability is a stated project requirement, not a bonus. The pipeline prints the top-weighted terms per label after training (see `run_output.txt`), so a non-technical reviewer on the research team can check the model's reasoning against their own domain knowledge — e.g., confirming that "deni" (debt) and "mkopo" (loan) are in fact driving the `debt` label, rather than an artifact like sentence length.

A small hand-built lexicon (`DOMAIN_LEXICON` in the code) seeds each category with known Swahili/Sheng/English terms. It is not used to hard-code predictions; it exists so that generic preprocessing doesn't strip out the exact words — "chama," "akiba," "mzigo" — that carry the most signal, and to give the research team a concrete, extensible artifact to build on rather than an opaque model.

4. Results — and an honest reading of them

Run against this 25-row sample with a 70/30 train/test split, the held-out F1 scores are effectively zero (full output in `run_output.txt`). This is the expected and correct outcome at this data size, and I want to be transparent about it rather than present a misleadingly clean metric. With 7–10 examples per category split across train and test, a held-out set has only 1–4 examples per label — nowhere near enough for a statistical model to generalize. What the run output _does_ show usefully:

- The class-balance check confirms the taxonomy is reasonably represented even in this tiny sample (16–40% coverage per category), which is a sensible target to design toward as real annotation volume grows.
- The explainability report shows the model is, even with almost no data, already attaching sensible weight to the right terms per category (e.g. `debt → deni, sasa deni`; `lending_readiness → mkopo wa benki, dhamana`), which is a good sign that the feature representation (TF-IDF over code-mixed text, without aggressive stemming) is sound — the bottleneck is data volume, not approach.

The honest conclusion: this prototype validates the _pipeline and taxonomy_, not yet a deployable classifier. The realistic next step is scaling annotation to several hundred narratives per category (ideally via the community and research team collection process described in the role) before re-evaluating.

## 5. Responsible AI considerations

Four things I'd want built in from day one, not retrofitted later:

- Bias. The class-balance report above is a first-pass check. In production, label frequency should be cross-tabulated against respondent demographics (region, age, business type) so the model isn't just well-calibrated on average but well-calibrated for the women least likely to already be visible in formal financial data* — which is arguably the whole point of the project.
- Explainability. Every prediction should be traceable to the terms that drove it (as demonstrated above), so that a support worker or researcher can challenge a label, not just accept it.
- Privacy. The script includes a minimal PII scrubber for phone numbers as a placeholder; a real deployment needs a written consent and anonymization protocol — agreed with the community team, not just the engineering team — covering names, exact locations, and identifying business details before any narrative is stored or annotated.
- Ethical use of women's data. A classification like `lending_readiness` carries real consequences if it ever informs an actual lending decision. The taxonomy and model should be treated as a research and support-routing tool, with clear documentation (like this README) of what it was validated on and what it wasn't, so it is never quietly repurposed beyond that scope.
