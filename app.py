# app.py
"""
Diabetes Assistant - Streamlit frontend
Features:
- Home (animated heart + profile)
- Predictor (lazy load model.joblib or upload CSV/model)
- Home Remedies (Hindi + English romanized)
- Medicines (educational)
- Feedback (persist to feedback.json)
- Pastel styling
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, json, time
from pathlib import Path
from sklearn.pipeline import Pipeline

# --------- Config ----------
MODEL_PATH = "model.joblib"
FEEDBACK_PATH = "feedback.json"
st.set_page_config(page_title="Diabetes Sahayak / डायबिटीज़ सहायता", layout="wide")

# --------- Pastel CSS & small styling ----------
PASTEL_CSS = """
<style>
:root{
  --bg:#f7f6ff;
  --card:#ffffff;
  --muted:#6b6b6b;
  --accent:#f6d6f5;
  --primary:#b7d7e8;
  --soft:#f7f0ff;
}
body { background: var(--bg); }
.appview-container .main .block-container{ padding:1.25rem 2rem; }
.card {
  background: var(--card);
  border-radius:14px;
  padding:18px;
  box-shadow: 0 6px 18px rgba(20,20,50,0.06);
}
.header-small { color: #333; font-weight:700; }
.small-muted { color: var(--muted); font-size:0.9rem; }
.pastel-btn { background: linear-gradient(90deg,#ffd6e0,#e6f0ff); border: none; padding:8px 14px; border-radius:10px; }
.footer-note { font-size:0.8rem; color:#666; margin-top:10px; }
.lang-p { font-size:0.95rem; color:#333; }
</style>
"""

HEART_HTML = """
<div style="position:relative; border-radius:12px; overflow:hidden; margin-bottom:12px;">
  <svg viewBox="0 0 1200 240" preserveAspectRatio="none" style="width:100%; height:180px;">
    <defs>
      <linearGradient id="g" x1="0" x2="1">
        <stop offset="0%" stop-color="#f8d7ff"/>
        <stop offset="100%" stop-color="#d7f0ff"/>
      </linearGradient>
    </defs>
    <path d="M0 120 C 300 20 900 220 1200 120 L1200 0 L0 0 Z" fill="url(#g)">
      <animate attributeName="d" dur="6s" repeatCount="indefinite"
       values="M0 120 C 300 20 900 220 1200 120 L1200 0 L0 0 Z; M0 120 C 300 220 900 20 1200 120 L1200 0 L0 0 Z; M0 120 C 300 20 900 220 1200 120 L1200 0 L0 0 Z"/>
    </path>
  </svg>
  <div style="position:absolute; left:18px; top:14px; color:#222;">
    <h2 style="margin:0;">Diabetes Sahayak</h2>
    <div style="opacity:0.85;">Swasth jeevan • Quick checks • Ghar ke upaay</div>
  </div>
  <div style="position:absolute; right:18px; top:18px; text-align:right; color:#222;">
    <div id="hr" style="font-size:22px; font-weight:700;">-- bpm</div>
    <div style="font-size:12px; opacity:0.85;">Simulated heartbeat</div>
  </div>
</div>

<script>
let hrElInterval = null;
function startHR() {
  const el = document.getElementById('hr');
  function tick() {
    const base = 68 + Math.round(Math.sin(Date.now()/4500)*10);
    const noise = Math.round((Math.random()-0.5)*6);
    el.innerText = (base + noise) + ' bpm';
  }
  tick();
  if (!hrElInterval) hrElInterval = setInterval(tick, 1000);
}
startHR();
</script>
"""

st.markdown(PASTEL_CSS, unsafe_allow_html=True)

# --------- Lazy model loader ----------
_model = None
_model_err = None
def get_model():
    global _model, _model_err
    if _model is None and _model_err is None:
        if os.path.exists(MODEL_PATH):
            try:
                _model = joblib.load(MODEL_PATH)
            except Exception as e:
                _model_err = str(e)
        else:
            _model_err = "no-file"
    return _model, _model_err

# --------- Utility functions ----------
def save_feedback(entry):
    try:
        data = []
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH,'r',encoding='utf-8') as f:
                data = json.load(f)
        data.append(entry)
        with open(FEEDBACK_PATH,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        return True, None
    except Exception as e:
        return False, str(e)

def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH,'r',encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def predict_with_model(model, Xdf):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xdf)[:,1]
    else:
        proba = model.predict(Xdf)
        proba = np.array(proba, dtype=float)
    pred = model.predict(Xdf)
    return pred, proba

# --------- Sidebar navigation & profile ----------
st.sidebar.markdown("## ☑️ Menu / मेन्यू")
page = st.sidebar.radio("", ["Home / होम", "Predictor / भविष्यवक्ता", "Home Remedies / घरेलू उपाय", "Medicines / दवाइयाँ", "Feedback / सुझाव"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Profile (नाम / उम्र / Gender)")
name = st.sidebar.text_input("Name / Naam", value=st.session_state.get("name",""))
age = st.sidebar.number_input("Age / Umar", min_value=1, max_value=120, value=st.session_state.get("age",25))
gender = st.sidebar.selectbox("Gender / लिंग", ["Prefer not to say / बताना नहीं चाहते", "Male / पुरूष", "Female / महिला", "Other / अन्य"])
if st.sidebar.button("Save Profile / प्रोफ़ाइल सेव करें"):
    st.session_state["name"]=name
    st.session_state["age"]=age
    st.session_state["gender"]=gender
    st.sidebar.success("Saved in session — सेशन् में सेव हो गया")

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='small-muted'>Disclaimer: This app is educational only — चिकित्सकीय सलाह नहीं.</div>", unsafe_allow_html=True)

# --------- PAGES ----------
if page.startswith("Home"):
    st.markdown(HEART_HTML, unsafe_allow_html=True)
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown("### Welcome / स्वागत")
    st.markdown('<div class="lang-p">Hello! Enter your details in the sidebar (Naam, Umar, Gender).<br>नमस्ते! साइडबार में अपना नाम, उम्र और लिंग भरें।</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("#### Quick actions / त्वरित क्रियाएँ")
        if st.button("Go to Predictor / भविष्यवक्ता पर जाएँ"):
            st.experimental_rerun()
        st.markdown("#### About / बारे में")
        st.markdown("This is an educational demo to check diabetes risk and share safe home remedies. / यह एक शैक्षिक डेमो है।")
    with col2:
        st.metric("Simulated Heart Rate / दिल की धड़कन", "72 bpm", delta="+1")
        st.markdown("**Profile**")
        st.write(f"Name: **{st.session_state.get('name','-')}**")
        st.write(f"Age: **{st.session_state.get('age','-')}**")
        st.write(f"Gender: **{st.session_state.get('gender','-')}**")
    st.markdown('</div>',unsafe_allow_html=True)

elif page.startswith("Predictor"):
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.header("Diabetes Predictor / डायबिटीज़ भविष्यवक्ता")
    st.info("Educational only — चिकित्सा निर्णय के लिए डॉक्टर से मिले।")
    # model status
    model, model_err = get_model()
    if model is None:
        if model_err == "no-file":
            st.warning("No model.joblib found in app folder. Place your trained model as model.joblib, or use Upload options below. / model.joblib फ़ोल्डर में नहीं मिला।")
        else:
            st.error(f"Model load error: {model_err}")
    else:
        st.success("model.joblib loaded (lazy) — model ready. / मॉडल लोड हो गया।")

    st.subheader("Options / विकल्प")
    colu1, colu2 = st.columns(2)
    with colu1:
        uploaded_model = st.file_uploader("Upload model.joblib (optional) / मॉडल अपलोड करें", type=["joblib","pkl"])
        uploaded_csv = st.file_uploader("Or upload labeled CSV to train (optional) / CSV अपलोड करें", type=["csv"])
    with colu2:
        st.markdown("**If you upload a CSV**: it should contain a binary target column named 'Outcome' or 'target'.")
        st.markdown("यदि आप CSV अपलोड करते हैं तो उसमें 'Outcome' या 'target' नाम का टारगेट कॉलम होना चाहिए।")

    # handle uploaded model
    if uploaded_model is not None:
        try:
            tmp_path = os.path.join(".", "uploaded_model.joblib")
            with open(tmp_path,"wb") as f:
                f.write(uploaded_model.getbuffer())
            _m = joblib.load(tmp_path)
            _model = _m
            st.success("Uploaded model loaded for this session. / अपलोड किया गया मॉडल लोड हो गया।")
        except Exception as e:
            st.error(f"Uploaded model failed: {e}")

    # optionally train from CSV (session)
    trained_pipeline = None
    trained_cols = None
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.write("Preview:", df.head())
            target_col = None
            for g in ["Outcome","outcome","target","Target","diabetes"]:
                if g in df.columns:
                    target_col = g; break
            if target_col is None:
                st.error("No target column found. Rename your binary label to 'Outcome' or 'target'.")
            else:
                X = df.drop(columns=[target_col])
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.error("No numeric features to train.")
                else:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.pipeline import Pipeline
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.model_selection import train_test_split
                    Xn = X[numeric_cols]
                    y = df[target_col]
                    if len(y.unique())<2 or Xn.shape[0]<10:
                        st.error("Not enough data or labels.")
                    else:
                        Xtr, Xtst, ytr, ytst = train_test_split(Xn,y,test_size=0.2,random_state=42,stratify=y)
                        pipe = Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(max_iter=2000))])
                        pipe.fit(Xtr,ytr)
                        acc = pipe.score(Xtst,ytst)
                        trained_pipeline = pipe
                        trained_cols = numeric_cols
                        st.success(f"Trained pipeline (accuracy {acc:.2f}) — used for this session.")
        except Exception as e:
            st.error(f"CSV read/train failed: {e}")

    st.markdown("---")
    st.subheader("Enter patient values / मान दर्ज करें")
    c1,c2,c3 = st.columns(3)
    with c1:
        pregnancies = st.number_input("Pregnancies / गर्भधारण (count)", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose / ग्लूकोज़ (mg/dL)", min_value=0, max_value=500, value=120)
        bp = st.number_input("Blood Pressure / रक्तचाप (mm Hg)", min_value=0, max_value=200, value=70)
    with c2:
        skin = st.number_input("Skin Thickness / त्वचा मोटाई (mm)", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin / इंसुलिन (mu U/ml)", min_value=0, max_value=1000, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    with c3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
        age_val = st.number_input("Age / उम्र", min_value=1, max_value=120, value=st.session_state.get("age",30))
        run = st.button("Predict / भविष्यवाणी करें")

    input_map = {"Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":bp,"SkinThickness":skin,"Insulin":insulin,"BMI":bmi,"DiabetesPedigreeFunction":dpf,"Age":age_val}

    if run:
        # priority: trained_pipeline -> uploaded model / file -> saved model -> fallback
        used = None
        if trained_pipeline is not None:
            used="trained_csv"
            try:
                Xdf = pd.DataFrame([[input_map[c] for c in trained_cols]], columns=trained_cols)
                pred, proba = predict_with_model(trained_pipeline, Xdf)
                st.success(f"Prediction (trained from CSV) — Class: {pred[0]}, Prob: {proba[0]:.2f}")
            except Exception as e:
                st.error(f"Trained pipeline failed: {e}")
        else:
            model, err = get_model()
            if model is None:
                if err == "no-file":
                    st.warning("No saved model found. Upload model.joblib or train from CSV, or use rule-based check below.")
                else:
                    st.error(f"Model load error: {err}")
                # fallback rule:
                score = 0
                if glucose>125: score+=2
                elif glucose>100: score+=1
                if bmi>=30: score+=1
                if age_val>=45: score+=0.5
                prob = min(0.95,0.2*score)
                label = "High risk / उच्च जोखिम" if prob>=0.4 else ("Medium / मध्यम" if prob>=0.2 else "Low / कम")
                st.info(f"Rule-based result: {label} — approximate prob {prob:.2f}")
            else:
                try:
                    # try to map to model feature names if available
                    fn = None
                    fn = getattr(model, "feature_names_in_", None)
                    if fn is None and isinstance(model, Pipeline):
                        for step in model.steps:
                            fn = getattr(step[1], "feature_names_in_", None)
                            if fn is not None:
                                break
                    if fn is not None:
                        fnames = list(fn)
                        missing = [f for f in fnames if f not in input_map]
                        if missing:
                            st.info(f"Model expects features {missing} — filling missing with 0.")
                        Xrow = {f: float(input_map.get(f,0)) for f in fnames}
                        Xdf = pd.DataFrame([Xrow], columns=fnames)
                    else:
                        default_order = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
                        Xdf = pd.DataFrame([[input_map[c] for c in default_order]], columns=default_order)
                    pred, proba = predict_with_model(model, Xdf)
                    st.success(f"Model prediction — Class: {pred[0]}")
                    try:
                        st.write(f"Probability (positive): {float(proba[0]):.2f}")
                    except:
                        pass
                except Exception as e:
                    st.error(f"Prediction failed with model: {e}")
    st.markdown('</div>',unsafe_allow_html=True)

elif page.startswith("Home Remedies"):
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.header("Home Remedies / घरेलू उपाय")
    st.markdown("नीचे दिए गए सुझाव सामान्य सलाह हैं — चिकित्सकीय सलाह नहीं।")
    remedies = [
        ("Methi water / मेथी का पानी", "Soak fenugreek seeds overnight. Drink in morning. — Helps glucose control. / मेथी बीज रात भर भिगोकर सुबह लें।"),
        ("Cinnamon / दालचीनी", "Add small amount in food or tea — traditional support. / खाना या चाय में थोड़ी दालचीनी मिलाएँ।"),
        ("Walk 30 min / रोज़ 30 मिनट चलें", "Brisk walking improves insulin sensitivity. / तेज़ चलना इन्सुलिन संवेदनशीलता सुधारता है।"),
        ("Fiber rich foods / फाइबर युक्त आहार", "Whole grains, salads, lentils help control spikes. / साबुत अनाज, सलाद, दाल खाएँ।")
    ]
    for title, desc in remedies:
        st.subheader(f"{title}")
        st.write(desc)
    st.markdown('</div>',unsafe_allow_html=True)

elif page.startswith("Medicines"):
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.header("Medicines / दवाइयाँ (सामान्य जानकारी)")
    st.warning("This is educational information only. Consult a doctor before any medicine. / यह जानकारी शैक्षिक है। डॉक्टर से संपर्क करें।")
    meds = {
        "Metformin / मेटफॉर्मिन": "Often first-line for type 2; reduces liver glucose production.",
        "Insulin / इंसुलिन": "Used in type 1 and advanced type 2; dosing by doctor.",
        "SGLT2 inhibitors / एसजीएलटी2": "Help kidneys remove glucose via urine; specialist required.",
        "GLP-1 agonists / जीएलपी-1": "May help lower glucose and reduce weight in some patients."
    }
    for k,v in meds.items():
        st.subheader(k)
        st.write(v)
    st.markdown('</div>',unsafe_allow_html=True)

elif page.startswith("Feedback"):
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.header("Feedback & community tips / सुझाव")
    st.markdown("Share safe home tips. We store suggestions to a local file (session file). / सुरक्षित सुझाव साझा करें।")
    with st.form("feedback_form"):
        fname = st.text_input("Name / नाम")
        tip = st.text_area("Tip / सुझाव (short)")
        submit = st.form_submit_button("Submit / भेजें")
        if submit:
            if not tip.strip():
                st.error("Write a tip before submit. / सुझाव लिखें।")
            else:
                entry = {"name": fname or "Anonymous", "tip": tip.strip(), "time": time.asctime()}
                ok, err = save_feedback(entry)
                if ok:
                    st.success("Thanks! Your tip saved. / धन्यवाद! सुझाव सेव हो गया।")
                else:
                    st.error(f"Save failed: {err}")

    st.markdown("### Community tips (latest)")
    fb = load_feedback()
    if fb:
        for item in reversed(fb[-50:]):
            st.write(f"**{item.get('name','Anonymous')}** — {item.get('tip')}")
    else:
        st.info("No tips yet — be the first! / अभी तक कोई सुझाव नहीं।")
    st.markdown('</div>',unsafe_allow_html=True)

# Footer
st.markdown("<br><hr><div class='footer-note'>Developed as a demo. Educational only. भाषा: English + Hindi (romanized) shown together.</div>", unsafe_allow_html=True)
