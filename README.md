![Digital Payments and Default Choice](assets/header.svg)

# Digital Payments and Default Choice

Synthetic demonstration of payment defaults, linear probability models, logistic regression and average discrete changes.

**Research question:** How does a preselected payment option relate to payment choice after accounting for prior use, switching effort and reliance on the platform?

**Public edition:** Runnable synthetic demonstration plus [three original scripts and eight historical aggregate result tables](research_archive/). Participant-level records remain excluded.

**3 functional checks passed locally**, plus a complete demonstration run.

## Recovered original analysis

The [original archive](research_archive/) contains three original scripts and eight aggregate result tables. It preserves the group study's source and historical summaries alongside the synthetic example. All three scripts parse, but the original empirical analysis was not rerun because its participant-level input is excluded. The archived results retain their original provenance and interpretation limits.

## What you can inspect

- Generate reproducible fictional responses with explicit analysis fields.
- Compare unadjusted payment-choice shares across groups.
- Estimate an HC3 linear probability model and HC3 logistic regression.
- Calculate the mean discrete predicted-probability change when Default switches from zero to one.
- Export aggregate summaries, coefficients and a labelled example chart.

## Results preview

All values below are synthetic or illustrative workflow results.

![Digital Payments and Default Choice results preview](demo_output/choice.svg)

| Synthetic comparison | Probability-point change |
| --- | --- |
| Unadjusted group difference | 23.3093 |
| LPM Default coefficient | 22.358 |
| Logit average discrete change | 22.4228 |

These estimates describe 800 fictional records. Their agreement illustrates related summaries of the generated data, rather than reproducing the original academic study.

## Run the example

Use Python 3.12. From this project's directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python demo.py --demo --output demo_output
```

For an isolated environment, run `python -m venv .venv` first. Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\activate` on Windows. The standard example runs offline after dependencies are installed.

The [notebook walkthrough](notebooks/01_results_walkthrough.ipynb) explains the saved outputs. Exact command-line options appear in [CLI.txt](docs/CLI.txt).

## Documentation

| Guide | What it covers |
| --- | --- |
| [Methods](docs/METHODS.md) | Formulas, assumptions and interpretation |
| [Data dictionary](docs/DATA_DICTIONARY.md) | Variables, units and validation rules |
| [Results](docs/RESULTS.md) | Actual bundled output and how to read it |
| [Reproducibility](REPRODUCIBILITY.md) | Commands, checks and the recorded environment |
| [Provenance and attribution](docs/PROVENANCE.md) | Historical context, reconstruction and contribution |
| [Roadmap](docs/ROADMAP.md) | Specific extensions, with unfinished work labelled |
| [References](docs/REFERENCES.md) | Method documentation and source links |

## Limits of interpretation

- Bundled estimates are simulated and are not original study findings.
- Causal interpretation of actual survey results depends on allocation and study design.
- The public code does not certify a fully randomised factorial experiment.
- Small samples, separation and model misspecification require additional assessment with real data.

## Contribution and research context

GIPE Group 8 academic project comparing active choice with UPI preselection.

Arinjai Das contributed to quantitative analysis and interpretation within the group. The academic paper lists Aayushika Bhardwaj, Shreya Bhardwaj, Adhya Pande, Kunal Sachdeva, Bipasha and Arinjai Das. The public synthetic implementation was prepared with ChatGPT assistance.

**Skills demonstrated:** Python, behavioural economics, survey analysis, linear probability models, logistic regression, marginal effects.

[Portfolio](https://github.com/arinjaidas/My-Projects) · [GitHub profile](https://github.com/arinjaidas)
