import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Diabetes Health Portal", page_icon="💙", layout="wide")

# Load model
model = joblib.load("model.joblib")


# ------------------ HEART ANIMATION ------------------
heart_css = """
<style>
.heart {
  width: 80px;
  height: 80px;
  background: red;
  position: relative;
  transform: rotate(-45deg);
  animation: beat 1s infinite;
  margin: auto;
  margin-top: 40px;
}

.heart:before, .heart:after {
  content: "";
  width: 80px;
  height: 80px;
  background: red;
  border-radius: 50%;
  position: absolute;
}

.heart:before {
  top: -40px;
  left: 0;
}

.heart:after {
  left: 40px;
  top: 0;
}

@keyframes beat {
  0% { transform: rotate(-45deg) scale(1); }
  50% { transform: rotate(-45deg) scale(1.2); }
  100% { transform: rotate(-45deg) scale(1); }
}
</style>
"""
# ------------------------------------------------------


# ------------------ NAVIGATION ------------------
menu = ["Home", "Diabetes Predictor", "Medicine Recommender", "Home Remedies", "Feedback"]
choice = st.sidebar.selectbox("Navigation", menu)
# ------------------------------------------------------



# ------------------ HOME PAGE ------------------
if choice == "Home":
    st.title("💙 Diabetes Health Portal")
    st.markdown(heart_css, unsafe_allow_html=True)
    st.markdown("<div class='heart'></div>", unsafe_allow_html=True)

    st.write("Your complete portal for diabetes prediction, remedies and health support.")




# ------------------ DIABETES PREDICTOR ------------------
if choice == "Diabetes Predictor":
    st.title("🔍 Diabetes Prediction System")

    name = st.text_input("Your Name")
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.subheader("Enter Medical Values")
    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 300)
    bp = st.number_input("Blood Pressure", 0, 200)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)

    if st.button("Predict"):
        features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error(f"⚠️ {name}, you are likely Diabetic")
        else:
            st.success(f"🎉 {name}, you are NOT Diabetic")




# ------------------ MEDICINE RECOMMENDER ------------------
if choice == "Medicine Recommender":
    st.title("💊 Diabetes Medicine Guide")

    st.write("""
    ### Common Medicines for Diabetes:
    **Type-2 Diabetes:**
    - Metformin  
    - Glimepiride  
    - Sitagliptin  
    - Dapagliflozin  
    - Insulin (advanced cases)

    **Type-1 Diabetes:**
    - Insulin Regular  
    - Insulin Lispro  
    """)




# ------------------ HOME REMEDIES ------------------
if choice == "Home Remedies":
    st.title("🌿 Home Remedies to Control Diabetes")
    
    st.write("""
    - Drink **methi (fenugreek)** water every morning  
    - **Cinnamon** helps control blood sugar  
    - Walk **30 minutes daily**  
    - Eat **brown rice** instead of white  
    - Avoid sweet snacks  
    - Stay hydrated  
    """)




# ------------------ FEEDBACK PAGE ------------------
if choice == "Feedback":
    st.title("📝 Share Your Feedback or Remedies")

    user = st.text_input("Your Name")
    message = st.text_area("Write Here")

    if st.button("Submit"):
        st.success("Thank you for your feedback!")
