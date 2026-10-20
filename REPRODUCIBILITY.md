# Reproducibility

New synthetic analytical demonstration; original scripts, participant records and study estimates excluded.

## Execute from this project directory

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python demo.py --demo --output demo_output
```

**3 functional checks passed locally**, plus a complete demonstration run.

Public verification used Python **3.12.14**. Package versions are recorded in [environment.json](docs/environment.json). Requirements specify compatible minimums rather than a fully locked cross-platform environment.

Randomised examples use explicit seeds. The recorded numerical output is inspectable; small library/platform differences can affect optimisation and floating-point results. SVG creation timestamps and internal element identifiers can vary between renders.

The standalone package includes a GitHub Actions workflow. It has been parsed locally; a successful hosted run is not claimed before that workflow actually executes. The CI uses Python 3.12 and read-only repository contents permission.

MATLAB companion scripts in the ARMA project remain unexecuted in this release. The optional gold-project downloader was not exercised. The review reproduction retains archived statistical vintages rather than downloading fresh observations.
