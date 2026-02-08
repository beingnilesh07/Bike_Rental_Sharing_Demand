import streamlit as st
import joblib
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bike Rental Demand Prediction",
    page_icon="🏍️",
    layout="centered"
)

# ---------------- LOAD MODEL & ARTIFACTS ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(
        "C:\\Users\\bilad\\ExcelR\\Project\\Models\\Gradient_Boosting_Model.pkl"
    )
    preprocessed_data = joblib.load(
        "C:\\Users\\bilad\\ExcelR\\Project\\Models\\preprocessed_data.pkl"
    )
    X_test = joblib.load(
        "C:\\Users\\bilad\\ExcelR\\Project\\Models\\X_test.pkl"
    )
    y_test = joblib.load(
        "C:\\Users\\bilad\\ExcelR\\Project\\Models\\y_test.pkl"
    )
    return model, preprocessed_data, X_test, y_test


model, PREPROCESSED_DATA, X_test, y_test = load_artifacts()

# ---------------- TITLE ----------------
st.title("🏍️ Bike Rental Demand Prediction")
st.caption("Gradient Boosting Regression Model")
st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🔧 Input Features")

year = st.sidebar.selectbox("Year", [2011, 2012], index=1)

month = st.sidebar.selectbox(
    "Month",
    list(range(1, 13)),
    format_func=lambda x: datetime.date(1900, x, 1).strftime("%B")
)

hr = st.sidebar.slider("Hour", 0, 23, 9)

temp_raw = st.sidebar.slider("Temperature (°C)", 0.0, 40.0, 20.0)
hum_raw = st.sidebar.slider("Humidity (%)", 0, 100, 60)
windspeed_raw = st.sidebar.slider("Windspeed (km/h)", 0.0, 67.0, 10.0)

holiday = st.sidebar.radio("Holiday", ["No", "Yes"])
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
weather = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Mist", "Light Rain", "Heavy Rain"]
)

# ---------------- RAW → MODEL VALUES ----------------
temp = temp_raw / 40          # normalize
humidity = hum_raw
windspeed = windspeed_raw

holiday_val = 1 if holiday == "Yes" else 0
holiday_str = "Holiday" if holiday == "Yes" else "No Holiday"

today = datetime.datetime.today()
day = today.day
weekday = today.weekday()
weekday_str = today.strftime("%A")

is_weekend = 1 if weekday in [5, 6] else 0
workingday = 0 if holiday_val == 1 or is_weekend == 1 else 1
workingday_str = "Working Day" if workingday == 1 else "Non-Working Day"

month_str = datetime.date(1900, month, 1).strftime("%B")

# ---------------- USER INPUT SUMMARY ----------------
display_df = pd.DataFrame({
    "Hour": [hr],
    "Day": [day],
    "Month": [month_str],
    "Year": [year],
    "Weekday": [weekday_str],
    "Holiday": [holiday_str],
    "Day Type": [workingday_str],
    "Season": [season.capitalize()],
    "Weather": [weather],
    "Temperature (°C)": [temp_raw],
    "Humidity (%)": [hum_raw],
    "Windspeed (km/h)": [windspeed_raw]
})

st.subheader("🧾 User Input Summary")
st.dataframe(display_df, use_container_width=True)

# ---------------- FEATURE ENGINEERING ----------------
input_df = pd.DataFrame([{
    "hr": hr,
    "holiday": holiday_val,
    "workingday": workingday,
    "temp": temp,
    "hum": humidity,
    "windspeed": windspeed,
    "day": day,
    "month": month,
    "year": year,
    "weekday": weekday,
    "is_weekend": is_weekend,
    "season": season,
    "weathersit": weather
}])

# ---------------- ENCODING ----------------
input_df = pd.get_dummies(
    input_df,
    columns=["season", "weathersit"]
)

input_df = input_df.reindex(
    columns=PREPROCESSED_DATA.columns,
    fill_value=0
)

# Safety check
if "count" in input_df.columns:
    input_df = input_df.drop(columns=["count"])

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Bike Demand"):
    prediction = model.predict(input_df)[0]

    st.success("Prediction Successful")
    st.metric("Estimated Rentals", int(prediction))

    # ---------------- LINE PLOT ----------------
    st.subheader("📈 Actual vs Predicted Demand")

    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y_test.values[:100], label="Actual", linewidth=2)
    ax.plot(y_pred[:100], label="Predicted", linestyle="-")

    ax.set_title("Actual vs Predicted Bike Rentals")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Bike Rentals")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.divider()
st.caption("Bike Rental Sharing System | ExcelR Project")
