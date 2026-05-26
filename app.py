
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="BHP Dynamic Ore Recovery Optimisation Prototype", layout="wide")

st.title("BHP Dynamic Ore Recovery Optimisation Prototype")
st.caption("Functional AI demo: input mining conditions → ML prediction → digital twin scenario comparison → recommended business action")

# ----------------------------
# 1. Create demo training data
# ----------------------------
@st.cache_data
def create_demo_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    ore_grade = rng.uniform(48, 68, n)
    drilling_accuracy = rng.uniform(70, 99, n)
    blast_quality = rng.uniform(55, 98, n)
    equipment_availability = rng.uniform(60, 99, n)
    haulage_delay = rng.uniform(0, 55, n)
    plant_throughput = rng.uniform(65, 105, n)
    weather_risk = rng.uniform(0, 100, n)
    waste_ratio = rng.uniform(10, 45, n)

    recovery = (
        28
        + 0.55 * ore_grade
        + 0.14 * drilling_accuracy
        + 0.10 * blast_quality
        + 0.12 * equipment_availability
        + 0.08 * plant_throughput
        - 0.08 * haulage_delay
        - 0.045 * weather_risk
        - 0.11 * waste_ratio
        + rng.normal(0, 1.6, n)
    )
    recovery = np.clip(recovery, 55, 96)

    return pd.DataFrame({
        "ore_grade_pct": ore_grade.round(2),
        "drilling_accuracy_pct": drilling_accuracy.round(2),
        "blast_quality_score": blast_quality.round(2),
        "equipment_availability_pct": equipment_availability.round(2),
        "haulage_delay_min": haulage_delay.round(2),
        "plant_throughput_pct": plant_throughput.round(2),
        "weather_risk_score": weather_risk.round(2),
        "waste_ratio_pct": waste_ratio.round(2),
        "ore_recovery_pct": recovery.round(2)
    })

demo_data = create_demo_data()

# ----------------------------
# 2. Train ML model
# ----------------------------
features = [
    "ore_grade_pct",
    "drilling_accuracy_pct",
    "blast_quality_score",
    "equipment_availability_pct",
    "haulage_delay_min",
    "plant_throughput_pct",
    "weather_risk_score",
    "waste_ratio_pct"
]
target = "ore_recovery_pct"

X_train, X_test, y_train, y_test = train_test_split(demo_data[features], demo_data[target], test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=4)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# ----------------------------
# Sidebar: show AI processing
# ----------------------------
st.sidebar.header("AI Processing / Logic")
st.sidebar.write("Model: Random Forest Regression")
st.sidebar.write("Input: operational and geological variables")
st.sidebar.write("Output: predicted ore recovery % + recommended action")
st.sidebar.metric("Model MAE", f"{mae:.2f}%")
st.sidebar.metric("Model R²", f"{r2:.2f}")

uploaded = st.sidebar.file_uploader("Optional: upload a CSV with the same feature columns", type=["csv"])

st.markdown("""
### Live demo workflow
**User input → AI model predicts recovery → digital twin scenario compares options → dashboard recommends action**
""")

if uploaded is not None:
    input_df = pd.read_csv(uploaded)
    missing = [c for c in features if c not in input_df.columns]
    if missing:
        st.error(f"Uploaded file is missing these columns: {missing}")
        st.stop()
    current = input_df[features].iloc[[0]].copy()
else:
    st.subheader("1. User Input: Current Mine Shift Conditions")
    c1, c2, c3, c4 = st.columns(4)
    ore_grade = c1.slider("Ore grade (%)", 48.0, 68.0, 59.5, 0.1)
    drilling_accuracy = c2.slider("Drilling accuracy (%)", 70.0, 99.0, 84.0, 0.1)
    blast_quality = c3.slider("Blast quality score", 55.0, 98.0, 76.0, 0.1)
    equipment_availability = c4.slider("Equipment availability (%)", 60.0, 99.0, 82.0, 0.1)

    c5, c6, c7, c8 = st.columns(4)
    haulage_delay = c5.slider("Haulage delay (minutes)", 0.0, 55.0, 22.0, 0.5)
    plant_throughput = c6.slider("Plant throughput (%)", 65.0, 105.0, 88.0, 0.1)
    weather_risk = c7.slider("Weather risk score", 0.0, 100.0, 35.0, 1.0)
    waste_ratio = c8.slider("Waste ratio (%)", 10.0, 45.0, 28.0, 0.1)

    current = pd.DataFrame([{
        "ore_grade_pct": ore_grade,
        "drilling_accuracy_pct": drilling_accuracy,
        "blast_quality_score": blast_quality,
        "equipment_availability_pct": equipment_availability,
        "haulage_delay_min": haulage_delay,
        "plant_throughput_pct": plant_throughput,
        "weather_risk_score": weather_risk,
        "waste_ratio_pct": waste_ratio
    }])

# ----------------------------
# 3. Prediction and scenario logic
# ----------------------------
base_recovery = float(model.predict(current)[0])

scenario = current.copy()
scenario["drilling_accuracy_pct"] = np.minimum(99, scenario["drilling_accuracy_pct"] + 4)
scenario["haulage_delay_min"] = np.maximum(0, scenario["haulage_delay_min"] - 10)
scenario["waste_ratio_pct"] = np.maximum(8, scenario["waste_ratio_pct"] - 5)
scenario["plant_throughput_pct"] = np.minimum(110, scenario["plant_throughput_pct"] + 3)
scenario_recovery = float(model.predict(scenario)[0])
uplift = scenario_recovery - base_recovery

risk_score = (
    0.35 * (current["weather_risk_score"].iloc[0] / 100)
    + 0.25 * (current["haulage_delay_min"].iloc[0] / 55)
    + 0.25 * (1 - current["equipment_availability_pct"].iloc[0] / 100)
    + 0.15 * (current["waste_ratio_pct"].iloc[0] / 45)
) * 100

if uplift >= 2.0 and risk_score < 60:
    action = "APPROVE recommended adjustment"
    explanation = "Expected recovery uplift is meaningful and operational risk is acceptable."
elif uplift >= 1.0:
    action = "ESCALATE to mining engineer for review"
    explanation = "Potential uplift exists, but risk or confidence requires human validation."
else:
    action = "HOLD current plan"
    explanation = "Predicted benefit is too small to justify changing the current extraction plan."

confidence = max(55, min(96, 100 - abs(risk_score - 30) * 0.6))

st.subheader("2. AI Output: Recovery Prediction and Recommendation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current predicted recovery", f"{base_recovery:.2f}%")
m2.metric("Optimised scenario recovery", f"{scenario_recovery:.2f}%", f"{uplift:+.2f}%")
m3.metric("Operational risk score", f"{risk_score:.1f}/100")
m4.metric("AI confidence", f"{confidence:.1f}%")

st.subheader("3. Digital Twin Scenario Comparison")
comparison = pd.DataFrame({
    "Variable": features,
    "Current plan": [current[c].iloc[0] for c in features],
    "Optimised scenario": [scenario[c].iloc[0] for c in features]
})
st.dataframe(comparison, use_container_width=True)

st.subheader("4. Business Action / Workflow Outcome")
if action.startswith("APPROVE"):
    st.success(action)
elif action.startswith("ESCALATE"):
    st.warning(action)
else:
    st.info(action)

st.write(explanation)

st.markdown("**Recommended operational changes:**")
st.write("- Improve drill alignment in the target zone.")
st.write("- Reduce haulage delay by prioritising high-grade material routing.")
st.write("- Reduce waste movement by updating the short-term extraction sequence.")
st.write("- Increase plant throughput coordination for the next shift.")

st.markdown("**Audit note generated by system:**")
st.code(
    f"AI recommendation generated for current shift. Current recovery={base_recovery:.2f}%, "
    f"optimised recovery={scenario_recovery:.2f}%, expected uplift={uplift:.2f}%, "
    f"risk={risk_score:.1f}/100, confidence={confidence:.1f}%. Action: {action}."
)

with st.expander("Show sample training data used for prototype"):
    st.dataframe(demo_data.head(30), use_container_width=True)

with st.expander("Explain how the AI works"):
    st.write("""
    This prototype trains a Random Forest Regression model on sample mining data.
    The model learns relationships between ore grade, drilling accuracy, equipment availability,
    haulage delay, plant throughput, weather risk, waste ratio and ore recovery.
    It then predicts the expected ore recovery for the current operating conditions.
    A simplified digital twin scenario tests whether practical changes could improve the predicted recovery.
    The final dashboard converts the model output into a human-reviewable business action.
    """)
