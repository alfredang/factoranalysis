# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for interactive exploratory factor analysis with varimax rotation. Users upload CSV/Excel files, select variables via checkboxes, and get auto-labeled factors with rich visualizations.

## Commands

```bash
# Install dependencies
uv sync

# Run the app (opens at http://localhost:8501)
uv run streamlit run app.py
```

There are no tests or linting configured.

## Architecture

**app.py** is the entire application — a single-file Streamlit app with these sections:

1. **varimax_rotation()** — Custom orthogonal rotation implementation (not from a library). Rotates raw factor loadings for interpretability.
2. **Theme system** — Dark/light toggle using CSS injection (`st.markdown` with `unsafe_allow_html`) plus matplotlib `rcParams` overrides. State stored in `st.session_state["theme"]`.
3. **Data loading** — Sidebar with file upload (CSV/Excel via pandas + openpyxl) or built-in demo dataset (`creditcarddata.csv`).
4. **Column selection** — Checkboxes in a 3-column grid. Column names are prettified from `snake_case` to `Title Case` via `prettify()`.
5. **run_factor_analysis()** — Core pipeline: `StandardScaler` → `sklearn.FactorAnalysis` → varimax rotation → auto-label factors by top-2 loading variables. Returns loadings, scores, variance explained, communalities, and correlation matrix.
6. **Results display** — Uses `st.session_state["results"]` to persist across reruns. Loadings ≥0.4 are bolded. Charts use matplotlib rendered via `st.pyplot()`.

**factoranalysis.py** is the original standalone script (not imported by app.py). It uses `input()` for column selection and `plt.show()` for display. Kept as reference only.

## Key Patterns

- All analysis state is stored in `st.session_state` to survive Streamlit reruns
- Factor labels are auto-generated as `"Factor N: VarA / VarB"` from the two highest-loading variables
- The slider for number of factors is hidden when only 3 columns are selected (max_factors would equal min_value=2)
- `prettify()` is used consistently for display names in checkboxes, loadings table, and chart labels
