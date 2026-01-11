import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_input, load_artifacts
from utils.helpers import month_to_season, risk_label

# =================================================
# APP CONFIG
# =================================================
st.set_page_config(
    page_title="Tree Survival Planner",
    page_icon="🌱",
    layout="wide"
)

aft, encoder, feature_names = load_artifacts()

if "history" not in st.session_state:
    st.session_state.history = []

# =================================================
# SIDEBAR (NAVIGATION + INPUTS)
# =================================================
st.sidebar.title("🌱 Tree Survival Planner")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "New Prediction", "Explainability", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌿 Plantation Inputs")

species = st.sidebar.selectbox(
    "Tree Species",
    ["Prunus serotina", "Quercus alba", "Quercus rubra"]
)

soil = st.sidebar.selectbox(
    "Soil Type",
    ["Forest", "Sterile"]
)

light_val = st.sidebar.slider(
    "Light Availability",
    0, 100, 60
)

month = st.sidebar.slider(
    "Planting Month",
    1, 12, 7
)

myco_type = st.sidebar.selectbox(
    "Mycorrhizal Type",
    ["EMF", "AMF"]
)

census = st.sidebar.slider(
    "Observation Intensity (Census)",
    1, 20, 6
)

emf = st.sidebar.slider(
    "EMF Level",
    0, 100, 50
)

# =================================================
# DERIVED FEATURES
# =================================================
season = month_to_season(month)

light_cat = (
    "Low" if light_val < 40 else
    "Med" if light_val < 70 else
    "High"
)

soil_map = {
    "Forest": "Quercus alba",
    "Sterile": "Sterile"
}

# =================================================
# MODEL PREDICTION (SHARED)
# =================================================
X = preprocess_input(
    species=species,
    soil=soil_map[soil],
    light=light_cat,
    season=season,
    census=census,
    emf=emf,
    myco_type=myco_type,
    encoder=encoder,
    feature_names=feature_names
)

surv_fn = aft.predict_survival_function(X)

TARGET_TIME = 36
closest_time = surv_fn.index[
    np.abs(surv_fn.index - TARGET_TIME).argmin()
]

survival_prob = float(surv_fn.loc[closest_time].values[0])
risk = risk_label(survival_prob)

# Save history (only when visiting prediction)
if page == "New Prediction":
    st.session_state.history.append({
        "prob": survival_prob,
        "risk": risk,
        "season": season
    })

# =================================================
# DASHBOARD
# =================================================
if page == "Dashboard":

    st.title("📊 Plantation Risk Dashboard")

    if not st.session_state.history:
        st.info("Run predictions to populate the dashboard.")
    else:
        df = st.session_state.history

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_prob = np.mean([x["prob"] for x in df])
            st.metric(
                "Avg Survival Probability",
                f"{avg_prob*100:.1f}%"
            )

        with col2:
            low_count = sum(x["risk"] == "Low" for x in df)
            st.metric("Low Risk Predictions", low_count)

        with col3:
            common_season = max(
                set(x["season"] for x in df),
                key=[x["season"] for x in df].count
            )
            st.metric("Best Planting Window", common_season)

        st.markdown("### Risk Distribution")
        risk_counts = {
            "Low": sum(x["risk"] == "Low" for x in df),
            "Medium": sum(x["risk"] == "Medium" for x in df),
            "High": sum(x["risk"] == "High" for x in df),
        }
        st.bar_chart(risk_counts)

# =================================================
# NEW PREDICTION (MAIN SCREEN)
# =================================================
elif page == "New Prediction":

    st.title("🌿 Survival Prediction Result")

    st.progress(min(survival_prob, 1.0))

    st.metric(
        "Estimated Survival Probability (36 months)",
        f"{survival_prob*100:.1f}%"
    )

    risk_color = {
        "Low": "🟢 Low Risk",
        "Medium": "🟡 Medium Risk",
        "High": "🔴 High Risk"
    }
    st.subheader(risk_color[risk])

    # Insight line (MOST IMPORTANT)
    insights = []

    if light_cat == "Low":
        insights.append("low light availability")
    if soil == "Sterile":
        insights.append("poor soil quality")
    if myco_type == "AMF":
        insights.append("weaker mycorrhizal support")
    if season == "Summer":
        insights.append("summer planting stress")
    if season == "Monsoon":
        insights.append("favorable monsoon conditions")

    if insights:
        st.info(
            f"⚠️ High impact factors: {', '.join(insights[:2])}."
        )
    else:
        st.success(
            "✅ Environmental conditions are favorable for survival."
        )

    st.markdown(
        f"**Recommended Planting Window:** 🌧️ **{season}**"
    )

    # Dynamic survival curve
    st.markdown("### ⏱️ Survival Curve")

    fig, ax = plt.subplots()
    ax.step(surv_fn.index, surv_fn.iloc[:, 0], where="post")
    ax.axvline(closest_time, linestyle="--", color="red")
    ax.scatter(closest_time, survival_prob, color="red")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)

# =================================================
# EXPLAINABILITY
# =================================================
elif page == "Explainability":

    st.title("🔍 Why this prediction?")

    st.markdown(
        """
        This prediction is driven by:
        - **Observation duration (time & census)**
        - **Environmental stress factors (light & soil)**
        - **Seasonal planting effects**
        - **Mycorrhizal association strength**

        Survival analysis estimates *how long* a tree is likely to survive,
        not just whether it survives.
        """
    )

# =================================================
# ABOUT
# =================================================
else:

    st.title("ℹ️ About the Application")

    st.markdown(
        """
        **Tree Survival Planner** is a decision-support tool designed to help
        planners reduce plantation failure by selecting the right species and
        planting window.

        **Key characteristics:**
        - Survival Analysis (Weibull AFT)
        - Interactive decision UI
        - Explainable predictions
        - Mobile-friendly design

        **Limitations:**
        - Dataset is ecological, not city-specific
        - Not an automation system
        - Can be extended using weather APIs and soil sensors
        """
    )