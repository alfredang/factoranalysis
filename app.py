import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

st.set_page_config(page_title="Factor Analysis", layout="wide")


# --- Theme ---
def get_theme():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"
    return st.session_state["theme"]


def apply_theme(theme):
    if theme == "Dark":
        plt.rcParams.update({
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#0e1117",
            "axes.edgecolor": "#555",
            "axes.labelcolor": "#fafafa",
            "text.color": "#fafafa",
            "xtick.color": "#fafafa",
            "ytick.color": "#fafafa",
            "grid.color": "#333",
            "legend.facecolor": "#262730",
            "legend.edgecolor": "#555",
        })
        st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: #fafafa; }
            [data-testid="stSidebar"] { background-color: #262730; }
            [data-testid="stHeader"] { background-color: #0e1117; }
            h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {color: #fafafa !important;}
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label { color: #fafafa !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        plt.rcParams.update(plt.rcParamsDefault)


theme = get_theme()
apply_theme(theme)


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


# --- Sidebar ---
st.sidebar.header("Settings")

if st.sidebar.toggle("Dark Mode", value=(theme == "Dark")):
    st.session_state["theme"] = "Dark"
else:
    st.session_state["theme"] = "Light"

st.sidebar.markdown("---")
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
        "Eigenvalue": eigenvalues.round(3),
        "Variance Explained (%)": var_explained.round(2),
        "Cumulative (%)": cumulative_var.round(2),
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

    # KMO-like measure: check adequacy
    corr_matrix = np.corrcoef(X_scaled, rowvar=False)

    return {
        "loadings": loadings_df,
        "scores": scores_df,
        "variance": variance_df,
        "communalities": communalities_df,
        "factor_labels": factor_labels,
        "n_dropped": n_dropped,
        "corr_matrix": pd.DataFrame(corr_matrix, index=pretty_names, columns=pretty_names),
    }


if st.button("Run Factor Analysis"):
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

    if n_dropped > 0:
        st.warning(f"{n_dropped} rows were dropped due to missing or non-numeric values.")

    # --- Variance Explained ---
    st.subheader("Variance Explained")
    st.dataframe(variance_df.set_index("Factor"), use_container_width=True)

    # --- Factor Loadings (rotated) ---
    st.subheader("Factor Loadings (Varimax Rotated)")
    st.caption("Values closer to +1 or -1 indicate a strong relationship between the variable and the factor.")

    # Highlight strong loadings with styling
    def highlight_loadings(val):
        if abs(val) >= 0.4:
            return "font-weight: bold"
        return ""

    st.dataframe(
        loadings.round(3).style.map(highlight_loadings),
        use_container_width=True,
    )

    # --- Communalities ---
    st.subheader("Communalities")
    st.caption("How much of each variable's variance is explained by the extracted factors.")
    st.dataframe(communalities_df.set_index("Variable"), use_container_width=True)

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Factor Loadings Bar Chart**")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        loadings.plot(kind="bar", ax=ax1)
        ax1.set_title("Factor Loadings (Varimax Rotated)")
        ax1.set_ylabel("Loading Value")
        ax1.axhline(y=0.4, color="gray", linestyle="--", alpha=0.5)
        ax1.axhline(y=-0.4, color="gray", linestyle="--", alpha=0.5)
        ax1.tick_params(axis="x", rotation=45)
        ax1.legend(fontsize=8)
        fig1.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.markdown("**Factor Scores Scatter Plot**")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(scores_df.iloc[:, 0], scores_df.iloc[:, 1], alpha=0.6)
        ax2.set_xlabel(factor_labels[0])
        ax2.set_ylabel(factor_labels[1])
        ax2.set_title("Factor Scores")
        ax2.grid(True)
        fig2.tight_layout()
        st.pyplot(fig2)

    # --- Correlation Matrix ---
    with st.expander("Correlation Matrix"):
        st.dataframe(results["corr_matrix"].round(3), use_container_width=True)
