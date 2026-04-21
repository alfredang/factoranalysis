import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import bartlett


def varimax_rotation(loadings_matrix, max_iter=100, tol=1e-6):
    """Apply varimax rotation to factor loadings for better interpretability."""
    n, k = loadings_matrix.shape
    rotated = loadings_matrix.copy()
    rotation_matrix = np.eye(k)
    for _ in range(max_iter):
        for i in range(k):
            for j in range(i + 1, k):
                u = rotated[:, i]
                v = rotated[:, j]
                a = 2 * np.sum(u * v)
                b = np.sum(u**2 - v**2)
                angle = 0.25 * np.arctan2(a, b)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotated[:, i] = u * cos_a + v * sin_a
                rotated[:, j] = -u * sin_a + v * cos_a
                rot = np.eye(k)
                rot[i, i] = cos_a
                rot[j, j] = cos_a
                rot[i, j] = sin_a
                rot[j, i] = -sin_a
                rotation_matrix = rotation_matrix @ rot
        old = loadings_matrix @ rotation_matrix
        if np.max(np.abs(rotated - old)) < tol:
            break
    return rotated


st.set_page_config(page_title="Factor Analysis", layout="wide", page_icon="\U0001F4CA")

# --- Color Palette ---
ACCENT = "#2dd4bf"       # teal-400
ACCENT_DIM = "#0d9488"   # teal-600
BG_DARK = "#0f172a"      # slate-900
BG_CARD = "#1e293b"      # slate-800
BG_SURFACE = "#334155"   # slate-700
TEXT_PRIMARY = "#f1f5f9"  # slate-100
TEXT_MUTED = "#94a3b8"    # slate-400
DANGER = "#f43f5e"       # rose-500
CHART_COLORS = ["#2dd4bf", "#818cf8", "#fb923c", "#f472b6", "#a3e635", "#38bdf8", "#e879f9", "#fbbf24"]


# --- Theme (dark only) ---
plt.rcParams.update({
    "figure.facecolor": BG_CARD,
    "axes.facecolor": BG_CARD,
    "axes.edgecolor": BG_SURFACE,
    "axes.labelcolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": BG_SURFACE,
    "grid.alpha": 0.4,
    "legend.facecolor": BG_CARD,
    "legend.edgecolor": BG_SURFACE,
    "font.family": "sans-serif",
})


def inject_css():
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

            :root {{
                --accent: {ACCENT};
                --bg-dark: {BG_DARK};
                --bg-card: {BG_CARD};
                --bg-surface: {BG_SURFACE};
                --text-primary: {TEXT_PRIMARY};
                --text-muted: {TEXT_MUTED};
            }}

            .stApp {{
                background: linear-gradient(135deg, {BG_DARK} 0%, #1a1a2e 50%, {BG_DARK} 100%);
                font-family: 'DM Sans', sans-serif;
            }}

            [data-testid="stSidebar"] {{
                background: {BG_CARD};
                border-right: 1px solid {BG_SURFACE};
                min-width: 320px !important;
                width: 320px !important;
            }}

            [data-testid="stSidebar"] > div:first-child {{
                width: 320px !important;
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            /* Hide sidebar collapse/expand icon text */
            [data-testid="stBaseButton-headerNoPadding"],
            [data-testid="collapsedControl"] {{
                font-size: 0 !important;
                color: transparent !important;
                overflow: hidden !important;
                width: 2rem !important;
                height: 2rem !important;
            }}

            [data-testid="stBaseButton-headerNoPadding"] span,
            [data-testid="collapsedControl"] span {{
                font-size: 0 !important;
                color: transparent !important;
            }}

            [data-testid="stBaseButton-headerNoPadding"] svg,
            [data-testid="collapsedControl"] svg {{
                font-size: 1.2rem !important;
                color: {TEXT_PRIMARY} !important;
            }}

            h1, h2, h3 {{
                font-family: 'DM Sans', sans-serif !important;
                color: {TEXT_PRIMARY} !important;
                font-weight: 700 !important;
                letter-spacing: -0.02em;
            }}

            h4, h5, h6, p, label, .stMarkdown {{
                color: {TEXT_PRIMARY} !important;
                font-family: 'DM Sans', sans-serif !important;
            }}

            span {{
                color: {TEXT_PRIMARY} !important;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0;
                background: {BG_CARD};
                border-radius: 12px;
                padding: 4px;
                border: 1px solid {BG_SURFACE};
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 8px;
                padding: 10px 20px;
                color: {TEXT_MUTED};
                font-family: 'DM Sans', sans-serif;
                font-weight: 500;
                font-size: 0.9rem;
                background: transparent;
                border: none;
            }}

            .stTabs [aria-selected="true"] {{
                background: {ACCENT} !important;
                color: {BG_DARK} !important;
                font-weight: 700;
            }}

            .stTabs [data-baseweb="tab-highlight"] {{
                display: none;
            }}

            .stTabs [data-baseweb="tab-border"] {{
                display: none;
            }}

            .stDataFrame {{
                border-radius: 12px;
                overflow: hidden;
            }}

            div[data-testid="stDataFrame"] > div {{
                border-radius: 12px;
                border: 1px solid {BG_SURFACE};
            }}

            .stButton > button {{
                background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DIM} 100%);
                color: {BG_DARK};
                border: none;
                border-radius: 10px;
                font-weight: 700;
                font-family: 'DM Sans', sans-serif;
                padding: 0.6rem 2rem;
                font-size: 0.95rem;
                letter-spacing: 0.02em;
                transition: all 0.2s ease;
                box-shadow: 0 4px 14px rgba(45, 212, 191, 0.25);
            }}

            .stButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 6px 20px rgba(45, 212, 191, 0.35);
            }}

            .stSlider [data-baseweb="slider"] {{
                margin-top: 0.5rem;
            }}

            div[data-testid="stMetric"] {{
                background: {BG_CARD};
                border: 1px solid {BG_SURFACE};
                border-radius: 12px;
                padding: 1rem 1.2rem;
            }}

            div[data-testid="stMetric"] label {{
                color: {TEXT_MUTED} !important;
                font-size: 0.8rem !important;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}

            div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
                color: {ACCENT} !important;
                font-family: 'JetBrains Mono', monospace !important;
                font-weight: 700 !important;
            }}

            .metric-card {{
                background: {BG_CARD};
                border: 1px solid {BG_SURFACE};
                border-radius: 14px;
                padding: 1.4rem;
                text-align: center;
            }}

            .metric-card .metric-value {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 2rem;
                font-weight: 700;
                color: {ACCENT};
                line-height: 1.2;
            }}

            .metric-card .metric-label {{
                font-size: 0.75rem;
                color: {TEXT_MUTED};
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-top: 0.3rem;
            }}

            .section-header {{
                display: flex;
                align-items: center;
                gap: 0.6rem;
                margin-bottom: 0.5rem;
            }}

            .section-header .icon {{
                font-size: 1.4rem;
            }}

            .results-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                background: rgba(45, 212, 191, 0.1);
                border: 1px solid rgba(45, 212, 191, 0.3);
                color: {ACCENT};
                border-radius: 8px;
                padding: 0.4rem 1rem;
                font-size: 0.85rem;
                font-weight: 600;
                font-family: 'DM Sans', sans-serif;
                margin-bottom: 1rem;
            }}

            /* --- Sidebar text --- */
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label {{
                color: {TEXT_PRIMARY} !important;
            }}

            /* --- Dark Mode toggle --- */
            [data-testid="stSidebar"] .stCheckbox > label {{
                background: transparent !important;
                border: none !important;
                padding: 0.3rem 0 !important;
            }}

            /* --- File Uploader --- */
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
                background: rgba(255, 255, 255, 0.03) !important;
                border: 1px dashed {BG_SURFACE} !important;
                border-radius: 10px !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span,
            [data-testid="stFileUploader"] small {{
                color: {TEXT_MUTED} !important;
            }}

            [data-testid="stFileUploaderDropzone"] button {{
                background: {TEXT_PRIMARY} !important;
                color: {BG_DARK} !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                padding: 0.45rem 1.2rem !important;
                transition: all 0.2s ease !important;
            }}

            [data-testid="stFileUploaderDropzone"] button:hover {{
                background: {ACCENT} !important;
            }}

            [data-testid="stFileUploaderDropzone"] button span,
            [data-testid="stFileUploaderDropzone"] button p {{
                color: {BG_DARK} !important;
            }}

            /* --- Alerts --- */
            [data-testid="stAlert"] {{
                background: {BG_CARD} !important;
                border: 1px solid {BG_SURFACE} !important;
                border-radius: 10px !important;
            }}

            /* --- Slider --- */
            .stSlider [data-baseweb="slider"] div[role="slider"] {{
                background: {ACCENT} !important;
                border-color: {ACCENT} !important;
            }}

            .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {{
                background: {BG_SURFACE} !important;
            }}

            /* --- Radio buttons --- */
            [data-testid="stSidebar"] .stRadio label span {{
                color: {TEXT_PRIMARY} !important;
            }}

            /* --- Scrollbar --- */
            ::-webkit-scrollbar {{
                width: 6px;
                height: 6px;
            }}
            ::-webkit-scrollbar-track {{
                background: {BG_DARK};
            }}
            ::-webkit-scrollbar-thumb {{
                background: {BG_SURFACE};
                border-radius: 3px;
            }}
            ::-webkit-scrollbar-thumb:hover {{
                background: {TEXT_MUTED};
            }}

            /* --- Separator --- */
            hr {{
                border-color: {BG_SURFACE} !important;
                opacity: 0.5;
            }}

            /* --- Footer --- */
            .footer-link {{
                color: {ACCENT} !important;
                text-decoration: none;
                font-weight: 600;
                transition: opacity 0.2s;
            }}

            .footer-link:hover {{
                opacity: 0.8;
            }}

            /* --- Factor checkboxes (main area) --- */
            [data-testid="stMain"] div.stCheckbox > label {{
                background: {BG_CARD};
                border: 1px solid {BG_SURFACE};
                border-radius: 8px;
                padding: 0.5rem 0.8rem;
                transition: border-color 0.2s;
            }}

            [data-testid="stMain"] div.stCheckbox > label:hover {{
                border-color: {ACCENT};
            }}
        </style>
        """, unsafe_allow_html=True)


def prettify(col_name):
    """Convert column_name to nice English: credit_history_years -> Credit History Years."""
    return col_name.replace("_", " ").title()


def prettify_map(columns):
    """Return {pretty_label: original_col} mapping, handling duplicates."""
    mapping = {}
    for col in columns:
        label = prettify(col)
        mapping[label] = col
    return mapping


CHART_STYLE = {
    "line": ACCENT,
    "fill": ACCENT,
    "fill_alpha": 0.15,
    "marker": ACCENT,
    "grid": BG_SURFACE,
    "text": TEXT_PRIMARY,
    "muted": TEXT_MUTED,
    "bar_colors": CHART_COLORS,
    "scatter": ACCENT,
    "scatter_alpha": 0.5,
    "heatmap_cmap": LinearSegmentedColormap.from_list("custom", [BG_CARD, ACCENT_DIM, DANGER]),
}


# --- Apply theme ---
inject_css()

# --- Sidebar ---
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Upload a file", "Use demo dataset"])

df = None

if data_source == "Upload a file":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
else:
    demo_path = Path(__file__).parent / "creditcarddata.csv"
    if demo_path.exists():
        df = pd.read_csv(demo_path)
        st.sidebar.success("Demo dataset loaded.")
    else:
        st.sidebar.error("Demo file creditcarddata.csv not found.")

st.title("Factor Analysis Tool")
st.markdown("Upload a CSV or Excel file, select factors, and run factor analysis.")

if df is None:
    st.info("Please upload a file or select the demo dataset to get started.")
    st.stop()

# --- Data Preview ---
st.subheader("Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# --- Column Selection with Checkboxes ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 3:
    st.error("The dataset must have at least 3 numeric columns for factor analysis.")
    st.stop()

label_to_col = prettify_map(numeric_cols)

st.subheader("Select Factors")
st.caption("Choose the factors you want to include in the analysis.")

selected_cols = []
cols_per_row = 3
rows = [numeric_cols[i:i + cols_per_row] for i in range(0, len(numeric_cols), cols_per_row)]

for row in rows:
    checkbox_cols = st.columns(len(row))
    for i, col in enumerate(row):
        label = prettify(col)
        with checkbox_cols[i]:
            if st.checkbox(label, value=False, key=f"cb_{col}"):
                selected_cols.append(col)

st.markdown("---")

if len(selected_cols) < 3:
    st.warning("Please select at least 3 factors to run the analysis.")
    st.stop()

max_factors = min(10, len(selected_cols) - 1)
if max_factors < 2:
    max_factors = 2

if max_factors == 2:
    n_components = 2
else:
    n_components = st.slider("Number of factors to extract", min_value=2, max_value=max_factors, value=2)


# --- Factor Analysis ---
def run_factor_analysis(df, columns, n_components=2):
    data = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    n_dropped = len(df) - len(data)

    if len(data) < n_components:
        raise ValueError("Not enough valid rows after cleaning.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    fa = FactorAnalysis(n_components=n_components, random_state=42)
    fa.fit(X_scaled)

    # Raw loadings
    raw_loadings = fa.components_.T

    # Varimax rotation for interpretability
    rotated_loadings = varimax_rotation(raw_loadings)

    pretty_names = [prettify(c) for c in data.columns]

    # Auto-label factors by top-loading variables
    factor_labels = []
    for i in range(n_components):
        abs_loadings = np.abs(rotated_loadings[:, i])
        top_indices = abs_loadings.argsort()[-2:][::-1]
        top_vars = [pretty_names[idx] for idx in top_indices]
        factor_labels.append(f"Factor {i+1}: {top_vars[0]} / {top_vars[1]}")

    loadings_df = pd.DataFrame(
        rotated_loadings,
        index=pretty_names,
        columns=factor_labels,
    )

    # Eigenvalues and variance explained
    eigenvalues = np.sum(rotated_loadings**2, axis=0)
    total_var = X_scaled.shape[1]
    var_explained = eigenvalues / total_var * 100
    cumulative_var = np.cumsum(var_explained)

    variance_df = pd.DataFrame({
        "Factor": factor_labels,
        "Eigenvalue": eigenvalues.round(4),
        "Variance (%)": var_explained.round(2),
        "Cumulative Variance (%)": cumulative_var.round(2),
    })

    # Communalities (proportion of each variable's variance explained by the factors)
    communalities = np.sum(rotated_loadings**2, axis=1)
    communalities_df = pd.DataFrame({
        "Variable": pretty_names,
        "Communality": communalities.round(3),
    })

    # Factor scores
    scores = fa.transform(X_scaled)
    scores_df = pd.DataFrame(scores, columns=factor_labels)

    corr_matrix = np.corrcoef(X_scaled, rowvar=False)

    return {
        "loadings": loadings_df,
        "scores": scores_df,
        "variance": variance_df,
        "communalities": communalities_df,
        "factor_labels": factor_labels,
        "n_dropped": n_dropped,
        "corr_matrix": pd.DataFrame(corr_matrix, index=pretty_names, columns=pretty_names),
        "var_explained": var_explained,
        "cumulative_var": cumulative_var,
        "eigenvalues": eigenvalues,
    }


if st.button("Run Factor Analysis", type="primary"):
    try:
        results = run_factor_analysis(df, selected_cols, n_components)
        st.session_state["results"] = results
    except Exception as e:
        st.error(f"Analysis failed: {e}")

if "results" in st.session_state:
    results = st.session_state["results"]
    loadings = results["loadings"]
    scores_df = results["scores"]
    variance_df = results["variance"]
    communalities_df = results["communalities"]
    factor_labels = results["factor_labels"]
    n_dropped = results["n_dropped"]
    var_explained = results["var_explained"]
    cumulative_var = results["cumulative_var"]
    eigenvalues = results["eigenvalues"]
    colors = CHART_STYLE

    st.markdown(
        '<div class="results-badge">\u2705 Factor Analysis completed!</div>',
        unsafe_allow_html=True,
    )

    if n_dropped > 0:
        st.warning(f"{n_dropped} rows were dropped due to missing or non-numeric values.")

    # --- Summary Metrics ---
    st.markdown("### Analysis Results")

    metric_cols = st.columns(len(factor_labels) + 1)
    for i, fl in enumerate(factor_labels):
        short_label = f"Factor {i+1}"
        with metric_cols[i]:
            st.metric(label=short_label, value=f"{var_explained[i]:.1f}%")
    with metric_cols[-1]:
        st.metric(label="Total Explained", value=f"{cumulative_var[-1]:.1f}%")

    # --- Variance Explained Table ---
    st.markdown("#### Variance Explained")
    st.dataframe(variance_df.set_index("Factor"), use_container_width=True)

    # --- Factor Loadings Table ---
    st.markdown("#### Factor Loadings (Varimax Rotated)")
    st.caption("Values closer to +1 or -1 indicate a strong relationship between the variable and the factor.")

    def highlight_loadings(val):
        if abs(val) >= 0.4:
            return f"font-weight: bold; color: {ACCENT}"
        return ""

    st.dataframe(
        loadings.round(3).style.map(highlight_loadings),
        use_container_width=True,
    )

    # --- Communalities ---
    st.markdown("#### Communalities")
    st.caption("How much of each variable's variance is explained by the extracted factors.")
    st.dataframe(communalities_df.set_index("Variable"), use_container_width=True)

    # --- Visualizations in Tabs ---
    st.markdown("### Visualizations")

    tab_corr, tab_loadings, tab_scree, tab_scores = st.tabs([
        "\U0001F4CA Correlation Matrix",
        "\U0001F4CA Factor Loadings",
        "\U0001F4C9 Scree Plot",
        "\U0001F3AF Factor Scores",
    ])

    with tab_corr:
        st.markdown("#### Correlation Matrix")
        corr_data = results["corr_matrix"]
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        im = ax_corr.imshow(corr_data.values, cmap=colors["heatmap_cmap"], aspect="auto", vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(corr_data.columns)))
        ax_corr.set_yticks(range(len(corr_data.index)))
        ax_corr.set_xticklabels(corr_data.columns, rotation=45, ha="right", fontsize=9)
        ax_corr.set_yticklabels(corr_data.index, fontsize=9)
        for i_row in range(len(corr_data)):
            for j_col in range(len(corr_data)):
                val = corr_data.values[i_row, j_col]
                text_color = TEXT_PRIMARY
                if abs(val) > 0.6:
                    text_color = "#ffffff"
                ax_corr.text(j_col, i_row, f"{val:.2f}", ha="center", va="center",
                             fontsize=8, color=text_color, fontweight="bold" if abs(val) >= 0.4 else "normal")
        ax_corr.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=16)
        cbar = fig_corr.colorbar(im, ax=ax_corr, shrink=0.8, aspect=30)
        cbar.ax.tick_params(labelsize=8)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)

    with tab_loadings:
        st.markdown("#### Factor Loadings")
        fig_load, ax_load = plt.subplots(figsize=(10, 6))
        x = np.arange(len(loadings.index))
        n_factors = len(loadings.columns)
        bar_width = 0.7 / n_factors
        for i_f, col in enumerate(loadings.columns):
            offset = (i_f - n_factors / 2 + 0.5) * bar_width
            bars = ax_load.bar(x + offset, loadings[col], bar_width, label=f"Factor {i_f+1}",
                               color=colors["bar_colors"][i_f % len(colors["bar_colors"])], alpha=0.85,
                               edgecolor="none", zorder=3)
        ax_load.axhline(y=0.4, color=TEXT_MUTED, linestyle="--", alpha=0.5, linewidth=0.8)
        ax_load.axhline(y=-0.4, color=TEXT_MUTED, linestyle="--", alpha=0.5, linewidth=0.8)
        ax_load.axhline(y=0, color=colors["grid"], linewidth=0.5)
        ax_load.set_xticks(x)
        ax_load.set_xticklabels(loadings.index, rotation=45, ha="right", fontsize=9)
        ax_load.set_ylabel("Loading Value", fontsize=11)
        ax_load.set_title("Factor Loadings (Varimax Rotated)", fontsize=14, fontweight="bold", pad=16)
        ax_load.legend(fontsize=9, loc="upper right", framealpha=0.8)
        ax_load.grid(axis="y", alpha=0.2, zorder=0)
        ax_load.spines["top"].set_visible(False)
        ax_load.spines["right"].set_visible(False)
        fig_load.tight_layout()
        st.pyplot(fig_load)

    with tab_scree:
        st.markdown("#### Scree Plot")
        fig_scree, ax_scree = plt.subplots(figsize=(10, 6))
        x_factors = np.arange(1, len(var_explained) + 1)
        ax_scree.fill_between(x_factors, var_explained, alpha=colors["fill_alpha"],
                              color=colors["fill"], zorder=2)
        ax_scree.plot(x_factors, var_explained, color=colors["line"], linewidth=2.5,
                      marker="o", markersize=10, markerfacecolor=colors["marker"],
                      markeredgecolor="white",
                      markeredgewidth=2, zorder=3)
        for i_pt, (xv, yv) in enumerate(zip(x_factors, var_explained)):
            ax_scree.annotate(f"{yv:.1f}%", (xv, yv), textcoords="offset points",
                              xytext=(0, 14), ha="center", fontsize=10, fontweight="bold",
                              color=colors["text"])
        ax_scree.set_xlabel("Factor", fontsize=12)
        ax_scree.set_ylabel("Variance (%)", fontsize=12)
        ax_scree.set_title("Scree Plot \u2014 Explained Variance by Factor", fontsize=14,
                           fontweight="bold", pad=16)
        ax_scree.set_xticks(x_factors)
        ax_scree.set_xticklabels([f"Factor {i}" for i in x_factors], fontsize=10)
        ax_scree.grid(axis="y", alpha=0.2)
        ax_scree.spines["top"].set_visible(False)
        ax_scree.spines["right"].set_visible(False)
        fig_scree.tight_layout()
        st.pyplot(fig_scree)

    with tab_scores:
        st.markdown("#### Factor Scores")
        fig_scores, ax_scores = plt.subplots(figsize=(10, 6))
        scatter = ax_scores.scatter(
            scores_df.iloc[:, 0], scores_df.iloc[:, 1],
            c=colors["scatter"], alpha=colors["scatter_alpha"],
            s=30, edgecolors="none", zorder=3,
        )
        ax_scores.set_xlabel(factor_labels[0], fontsize=11)
        ax_scores.set_ylabel(factor_labels[1], fontsize=11)
        ax_scores.set_title("Factor Scores", fontsize=14, fontweight="bold", pad=16)
        ax_scores.axhline(y=0, color=colors["grid"], linewidth=0.5, alpha=0.5)
        ax_scores.axvline(x=0, color=colors["grid"], linewidth=0.5, alpha=0.5)
        ax_scores.grid(alpha=0.15, zorder=0)
        ax_scores.spines["top"].set_visible(False)
        ax_scores.spines["right"].set_visible(False)
        fig_scores.tight_layout()
        st.pyplot(fig_scores)

# --- Footer ---
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #64748b; padding: 1rem 0; font-family: DM Sans, sans-serif;">'
    'Powered by <a href="https://www.tertiarycourses.com.sg/" target="_blank" '
    f'class="footer-link" style="color: {ACCENT}; text-decoration: none; font-weight: 600;">'
    'Tertiary Infotech Academy Pte Ltd</a>'
    '</div>',
    unsafe_allow_html=True,
)
