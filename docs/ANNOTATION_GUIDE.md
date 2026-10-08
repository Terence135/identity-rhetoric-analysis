# Rhetorical annotation guide (draft)
Author: Somto Terence Ndu. Project DAIM_A_105.

These are candidate operational definitions, requiring pilot review and supervisor agreement.

| Column | Positive label criteria |
|---|---|
| dehumanisation | Represents group members as less than human, objects, animals or contamination. |
| threat_framing | Represents an identity group as a danger to safety, culture or institutions. |
| moral_condemnation | Attributes moral corruption to an identity group. |
| essentialisation | Presents a negative characteristic as inherent or universal to a group. |
| exclusion_hierarchy | Advocates identity-based exclusion, subordination or unequal treatment. |

Use 0/1 per strategy and permit multiple positive labels. Do not label identity mentions alone.
Record target_identities as semicolon-separated racism/sexism/homophobia/transphobia categories;
these describe the discriminatory target dimension, not inferred author demographics.
Record stance as endorsed_hostility, quotation, counter_speech, neutral or uncertain.
For this baseline, strategies mark endorsed identity-based discriminatory rhetoric only.
Quoted or rejected strategies are 0 unless also endorsed; retain stance for separate analysis.
Keep uncertain or insufficient-context rows outside training until adjudicated.
Use group_id to keep the same author/conversation in one split; connected author/thread components
should share one ID. Audit near duplicates manually before training. A random split cannot establish
cross-platform generalisation. Never infer target identity from a slur alone without context.

Pilot 100-150 posts, revise definitions, double-annotate a subset, record agreement per label,
and reserve evaluation data before prompt/model tuning. Retain intersectional labels.
