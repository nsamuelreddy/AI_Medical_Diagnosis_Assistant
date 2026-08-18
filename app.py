import streamlit as st
import pandas as pd
import joblib
import os
import re
from collections import defaultdict

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
    /* Base */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif;
    }

    /* Card panels */
    .diagnosis-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        color: white;
    }
    .stage-badge {
        display: inline-block;
        background: #00d4aa;
        color: #0f2027;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .symptom-chip {
        display: inline-block;
        background: rgba(0,212,170,0.15);
        border: 1px solid rgba(0,212,170,0.4);
        color: #00d4aa;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .uncertainty-box {
        background: rgba(255,193,7,0.1);
        border-left: 4px solid #ffc107;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL (cached so it only runs once)
# =====================================

@st.cache_resource
def load_model():
    missing = []
    for f in ["disease_model_v2.pkl", "symptoms_v2.pkl"]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        st.error(
            f"Missing file(s): {', '.join(missing)}. "
            "Run train.py first to generate the required model files."
        )
        st.stop()

    model    = joblib.load("disease_model_v2.pkl")
    symptoms = joblib.load("symptoms_v2.pkl")

    # Validate symptoms is a list/array of strings
    if not hasattr(symptoms, '__iter__') or isinstance(symptoms, str):
        st.error("symptoms_v2.pkl must contain a list of symptom strings.")
        st.stop()

    symptoms = list(symptoms)  # ensure plain list
    return model, symptoms


model, symptoms = load_model()

# =====================================
# SYNONYM MAP (auto-generated, no hardcoding)
# Maps common lay terms → canonical symptom names from the model
# Only maps if the canonical name is actually in our symptom list.
# =====================================

@st.cache_data
def build_synonym_map(symptom_list: list) -> dict:
    """
    Build a synonym → canonical_symptom dict.
    All canonical values are guaranteed to be in symptom_list.
    """
    symptom_set_lower = {s.lower(): s for s in symptom_list}

    # Raw synonym pairs: (lay_term, canonical_candidate)
    # The canonical_candidate is looked up in the actual symptom list;
    # if not present it is silently skipped — zero hardcoded assumptions.
    raw_pairs = [
        ("high temperature", "fever"),
        ("high fever", "fever"),
        ("running nose", "nasal congestion"),
        ("runny nose", "nasal congestion"),
        ("stuffy nose", "nasal congestion"),
        ("blocked nose", "nasal congestion"),
        ("sore throat", "throat pain"),
        ("throat ache", "throat pain"),
        ("body ache", "muscle pain"),
        ("body aches", "muscle pain"),
        ("muscular pain", "muscle pain"),
        ("myalgia", "muscle pain"),
        ("tiredness", "fatigue"),
        ("exhaustion", "fatigue"),
        ("feeling tired", "fatigue"),
        ("lethargy", "fatigue"),
        ("breathlessness", "shortness of breath"),
        ("difficulty breathing", "shortness of breath"),
        ("hard to breathe", "shortness of breath"),
        ("dyspnea", "shortness of breath"),
        ("tummy ache", "abdominal pain"),
        ("stomach ache", "abdominal pain"),
        ("stomach pain", "abdominal pain"),
        ("belly pain", "abdominal pain"),
        ("throwing up", "vomiting"),
        ("puking", "vomiting"),
        ("nauseous", "nausea"),
        ("feel sick", "nausea"),
        ("loose stools", "diarrhea"),
        ("loose motion", "diarrhea"),
        ("watery stool", "diarrhea"),
        ("loose bowels", "diarrhea"),
        ("head pain", "headache"),
        ("migraine", "headache"),
        ("skin rash", "rash"),
        ("skin irritation", "rash"),
        ("hives", "rash"),
        ("urticaria", "rash"),
        ("loss of appetite", "decreased appetite"),
        ("no appetite", "decreased appetite"),
        ("not hungry", "decreased appetite"),
        ("anorexia", "decreased appetite"),
        ("sweating", "sweats"),
        ("night sweats", "sweats"),
        ("profuse sweating", "sweats"),
        ("shivering", "chills"),
        ("rigors", "chills"),
        ("feeling cold", "chills"),
        ("back ache", "back pain"),
        ("lower back pain", "back pain"),
        ("joint ache", "joint pain"),
        ("joint aches", "joint pain"),
        ("arthralgia", "joint pain"),
        ("chest tightness", "chest pain"),
        ("chest discomfort", "chest pain"),
        ("dizziness", "dizziness"),
        ("vertigo", "dizziness"),
        ("lightheaded", "dizziness"),
        ("weight loss", "weight loss"),
        ("losing weight", "weight loss"),
        ("blurry vision", "blurred vision"),
        ("blurry eyesight", "blurred vision"),
        ("palpitations", "palpitation"),
        ("heart racing", "palpitation"),
        ("fast heartbeat", "palpitation"),
        ("itching", "itching"),
        ("pruritus", "itching"),
        ("scratching", "itching"),
        ("yellowish skin", "jaundice"),
        ("yellow eyes", "jaundice"),
        ("yellow skin", "jaundice"),
        ("swollen lymph nodes", "lymph node swelling"),
        ("glands swollen", "lymph node swelling"),
        ("painful urination", "burning urination"),
        ("pain while urinating", "burning urination"),
        ("dysuria", "burning urination"),
        ("frequent urination", "increased urination"),
        ("peeing a lot", "increased urination"),
        ("polyuria", "increased urination"),
        ("red eyes", "eye redness"),
        ("pink eye", "eye redness"),
        ("conjunctivitis", "eye redness"),
        ("watery eyes", "watery eyes"),
        ("tearing", "watery eyes"),
        ("hair loss", "hair loss"),
        ("alopecia", "hair loss"),
        ("anxiety", "anxiety"),
        ("panic", "anxiety"),
        ("feeling anxious", "anxiety"),
        ("sad", "depression"),
        ("feeling depressed", "depression"),
        ("low mood", "depression"),
        ("confusion", "confusion"),
        ("disoriented", "confusion"),
        ("forgetful", "memory loss"),
        ("memory problems", "memory loss"),
        ("swelling", "swelling"),
        ("edema", "swelling"),
        ("bloating", "bloating"),
        ("distended abdomen", "bloating"),
    ]

    synonym_map: dict = {}
    for lay, canonical in raw_pairs:
        # Accept if canonical exists verbatim (case-insensitive) in symptom list
        if canonical.lower() in symptom_set_lower:
            synonym_map[lay.lower()] = symptom_set_lower[canonical.lower()]
        # Also map the lay term itself if it happens to be in the symptom list
        if lay.lower() in symptom_set_lower:
            synonym_map[lay.lower()] = symptom_set_lower[lay.lower()]

    return synonym_map


synonym_map = build_synonym_map(symptoms)
symptom_set_lower = {s.lower(): s for s in symptoms}  # lowercase → original-case lookup

# =====================================
# NLP SYMPTOM EXTRACTION
# Works with space-separated symptom names as they exist in symptoms_v2.pkl.
# Strategy:
#   1. Direct phrase match (longest first to avoid partial overlaps)
#   2. Synonym map lookup
# =====================================

def extract_symptoms_from_text(text: str, symptom_list: list, syn_map: dict) -> list:
    """
    Extract known symptoms from free text.
    Returns a list of symptom strings exactly as they appear in symptom_list.
    """
    text_lower = text.lower()
    found = set()

    # Sort longest first so "shortness of breath" beats "breath"
    sorted_symptoms = sorted(symptom_list, key=lambda s: len(s), reverse=True)

    for symptom in sorted_symptoms:
        pattern = re.compile(
            r'(?<![a-z])' + re.escape(symptom.lower()) + r'(?![a-z])',
            re.IGNORECASE
        )
        if pattern.search(text_lower):
            found.add(symptom)

    # Apply synonym map for any remaining lay terms
    for lay_term, canonical in syn_map.items():
        if canonical not in found:
            pattern = re.compile(
                r'(?<![a-z])' + re.escape(lay_term) + r'(?![a-z])',
                re.IGNORECASE
            )
            if pattern.search(text_lower):
                found.add(canonical)

    return list(found)


# =====================================
# BUILD INPUT VECTOR
# Guarantees column order matches training exactly.
# =====================================

def build_input_vector(selected: list, all_symptoms: list) -> pd.DataFrame:
    data = {s: 0 for s in all_symptoms}
    for s in selected:
        if s in data:
            data[s] = 1
    return pd.DataFrame([data])[all_symptoms]


# =====================================
# GET PREDICTIONS
# Returns sorted list of (disease, probability).
# =====================================

def get_predictions(selected: list):
    input_df = build_input_vector(selected, symptoms)
    probs    = model.predict_proba(input_df)[0]
    classes  = model.classes_
    results  = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return results


# =====================================
# DISCRIMINATIVE FOLLOW-UP SYMPTOMS
#
# For the top-N diseases, find symptoms that:
#   - are not already selected
#   - appear in the feature list
#   - best differentiate between top diseases
# Uses a simple information-gain-style heuristic over
# the model's feature log-probabilities (BernoulliNB).
# Falls back to the raw symptom list if model lacks log_prob.
# =====================================

def get_discriminative_symptoms(
    current_symptoms: list,
    top_diseases: list,   # list of disease name strings
    n: int = 5,
) -> list:
    """
    Return up to n symptom names that are most discriminative
    among top_diseases and are not already selected.
    """
    selected_set = set(s.lower() for s in current_symptoms)
    candidate_symptoms = [s for s in symptoms if s.lower() not in selected_set]

    if not candidate_symptoms:
        return []

    # Try to use BernoulliNB log-probabilities for smart ranking
    try:
        classes      = list(model.classes_)
        top_indices  = [classes.index(d) for d in top_diseases if d in classes]

        if len(top_indices) < 2:
            raise ValueError("need at least 2 disease classes")

        # feature_log_prob_: shape (n_classes, n_features)
        log_probs = model.feature_log_prob_   # (n_classes, n_features)

        # For each candidate symptom, compute variance across top disease log-probs
        # High variance → more discriminative
        symptom_index = {s: i for i, s in enumerate(symptoms)}
        scores = []
        for sym in candidate_symptoms:
            if sym not in symptom_index:
                continue
            feat_idx   = symptom_index[sym]
            top_lps    = [log_probs[ci][feat_idx] for ci in top_indices]
            variance   = (max(top_lps) - min(top_lps))
            # Also weight by mean probability (prefer symptoms that are likely)
            mean_lp    = sum(top_lps) / len(top_lps)
            score      = variance + 0.2 * mean_lp
            scores.append((sym, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scores[:n]]

    except Exception:
        # Fallback: return first n unselected symptoms alphabetically
        return sorted(candidate_symptoms)[:n]


# =====================================
# LOAD OPTIONAL DISEASE INFO
# =====================================

@st.cache_data
def load_disease_info() -> dict:
    path = "disease_info.csv"
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        # Expect columns: disease, description (at minimum)
        if "disease" in df.columns and "description" in df.columns:
            return dict(zip(df["disease"].str.lower(), df["description"]))
    except Exception:
        pass
    return {}


disease_info_map = load_disease_info()


# =====================================
# SESSION STATE INIT
# =====================================

def init_state():
    defaults = {
        "stage":               1,           # 1 | 2 | 3
        "selected_symptoms":   [],          # confirmed symptom list
        "followup_answers":    {},          # symptom → bool
        "followup_symptoms":   [],          # symptoms to ask in stage 2
        "intermediate_results": None,       # predictions after stage 1
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

# =====================================
# TITLE
# =====================================

st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
    <span style="font-size:2.5rem;">🏥</span>
    <div>
        <h1 style="margin:0;font-family:'DM Serif Display',serif;font-size:2rem;">
            AI Medical Diagnosis Assistant
        </h1>
        <p style="margin:0;color:#666;font-size:0.9rem;">
            Intelligent multi-stage diagnosis · 754 diseases · BernoulliNB · 246K records
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================
# SIDEBAR — patient info + navigation
# =====================================

st.sidebar.header("👤 Patient Information")
name   = st.sidebar.text_input("Name")
age    = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.sidebar.divider()

# Stage indicator
stage_labels = {1: "🔍 Symptom Entry", 2: "❓ Follow-Up Questions", 3: "🩺 Diagnosis"}
st.sidebar.markdown("**Workflow Stage**")
for s, label in stage_labels.items():
    is_current = (st.session_state.stage == s)
    color = "#00d4aa" if is_current else "#888"
    weight = "600" if is_current else "400"
    st.sidebar.markdown(
        f"<span style='color:{color};font-weight:{weight};'>{label}</span>",
        unsafe_allow_html=True
    )

st.sidebar.divider()

if st.sidebar.button("🔄 Start Over", use_container_width=True):
    for key in ["stage", "selected_symptoms", "followup_answers",
                "followup_symptoms", "intermediate_results"]:
        del st.session_state[key]
    st.rerun()

st.sidebar.caption(
    "Patient info is for display only and does not affect predictions."
)

# =========================================================
# ██████████████  STAGE 1: SYMPTOM ENTRY  ████████████████
# =========================================================

if st.session_state.stage == 1:

    st.markdown('<span class="stage-badge">Stage 1 of 3</span>', unsafe_allow_html=True)
    st.subheader("🔍 Select & Describe Your Symptoms")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("**Choose from known symptoms**")
        selected_symptoms = st.multiselect(
            "Start typing to filter",
            options=sorted(symptoms),
            default=st.session_state.selected_symptoms,
            help="All 377 symptoms from your trained model are listed here.",
            label_visibility="collapsed"
        )

    with col_right:
        st.markdown("**Or describe in your own words**")
        user_text = st.text_area(
            "Free text",
            placeholder="e.g. I have fever, headache and a runny nose for 3 days",
            height=120,
            label_visibility="collapsed"
        )

    # NLP detection
    detected_from_text: list = []
    if user_text.strip():
        detected_from_text = extract_symptoms_from_text(user_text, symptoms, synonym_map)

    # Merge, deduplicate (preserve original case from symptoms list)
    combined_lower_seen = set()
    all_combined: list = []
    for s in (selected_symptoms + detected_from_text):
        if s.lower() not in combined_lower_seen:
            combined_lower_seen.add(s.lower())
            # Always use the canonical form from our symptoms list
            canonical = symptom_set_lower.get(s.lower(), s)
            all_combined.append(canonical)

    # Show auto-detected new symptoms
    if detected_from_text:
        newly = [s for s in detected_from_text if s.lower() not in
                 {x.lower() for x in selected_symptoms}]
        if newly:
            st.info(
                "✨ **Auto-detected from your text:** "
                + "  ·  ".join(s.title() for s in newly)
            )

    # Show current symptom chips
    if all_combined:
        chips_html = "".join(
            f'<span class="symptom-chip">✓ {s.title()}</span>'
            for s in all_combined
        )
        st.markdown(
            f"<div style='margin:0.5rem 0'>{chips_html}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    if st.button("Continue to Follow-Up Questions →", type="primary",
                 disabled=len(all_combined) == 0):

        if not all_combined:
            st.warning("Please select or describe at least one symptom.")
            st.stop()

        # Save symptoms and compute intermediate predictions
        st.session_state.selected_symptoms   = all_combined
        st.session_state.intermediate_results = get_predictions(all_combined)

        # Pick top 5 diseases for follow-up generation
        top5_diseases = [d for d, _ in st.session_state.intermediate_results[:5]]

        # Get discriminative follow-up symptoms
        followup = get_discriminative_symptoms(
            current_symptoms=all_combined,
            top_diseases=top5_diseases,
            n=6,
        )
        st.session_state.followup_symptoms = followup
        st.session_state.followup_answers  = {}
        st.session_state.stage             = 2
        st.rerun()

    if len(all_combined) == 0:
        st.caption("👆 Select or describe at least one symptom to continue.")


# =========================================================
# ████████████  STAGE 2: FOLLOW-UP QUESTIONS  ████████████
# =========================================================

elif st.session_state.stage == 2:

    st.markdown('<span class="stage-badge">Stage 2 of 3</span>', unsafe_allow_html=True)
    st.subheader("❓ Follow-Up Questions")

    st.markdown(
        "Based on your initial symptoms, we have a few targeted questions to "
        "improve the diagnosis accuracy."
    )

    # Show current symptoms
    chips_html = "".join(
        f'<span class="symptom-chip">✓ {s.title()}</span>'
        for s in st.session_state.selected_symptoms
    )
    st.markdown(
        f"<div style='margin:0.5rem 0;'><strong>Your symptoms:</strong> "
        f"{chips_html}</div>",
        unsafe_allow_html=True
    )

    # Preliminary top prediction
    if st.session_state.intermediate_results:
        top_disease, top_prob = st.session_state.intermediate_results[0]
        st.info(
            f"🔬 Preliminary leading diagnosis: **{top_disease.title()}** "
            f"({top_prob * 100:.1f}% confidence) — answer the questions below to refine."
        )

    st.divider()

    followup_symptoms = st.session_state.followup_symptoms

    if not followup_symptoms:
        st.info("No additional follow-up questions for this symptom set. Proceeding to diagnosis.")
        if st.button("View Diagnosis →", type="primary"):
            st.session_state.stage = 3
            st.rerun()
    else:
        answers: dict = {}

        for i, sym in enumerate(followup_symptoms):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i+1}.** Do you also have **{sym.title()}**?")
            with col2:
                val = st.radio(
                    f"fu_{sym}",
                    options=["Not sure", "Yes", "No"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key=f"radio_{sym}",
                )
                answers[sym] = val

        st.divider()

        col_back, col_next = st.columns([1, 3])
        with col_back:
            if st.button("← Back"):
                st.session_state.stage = 1
                st.rerun()
        with col_next:
            if st.button("Get Diagnosis →", type="primary"):
                # Add "Yes" answers to symptom list
                confirmed_extra = [
                    sym for sym, ans in answers.items() if ans == "Yes"
                ]
                # Merge with existing symptoms (deduplicated, canonical)
                existing_lower = {s.lower() for s in st.session_state.selected_symptoms}
                for s in confirmed_extra:
                    canonical = symptom_set_lower.get(s.lower(), s)
                    if canonical.lower() not in existing_lower:
                        st.session_state.selected_symptoms.append(canonical)
                        existing_lower.add(canonical.lower())

                st.session_state.followup_answers = answers
                st.session_state.stage            = 3
                st.rerun()


# =========================================================
# ████████████████  STAGE 3: FINAL DIAGNOSIS  ████████████
# =========================================================

elif st.session_state.stage == 3:

    st.markdown('<span class="stage-badge">Stage 3 of 3</span>', unsafe_allow_html=True)
    st.subheader("🩺 Final Diagnosis Report")

    final_symptoms = st.session_state.selected_symptoms

    if not final_symptoms:
        st.error("No symptoms found. Please start over.")
        st.stop()

    with st.spinner("Running final analysis…"):
        results = get_predictions(final_symptoms)

    predicted_disease = results[0][0]
    confidence        = results[0][1]

    # ── Patient Summary ──────────────────────────────────
    st.subheader("📋 Patient Summary")
    c1, c2, c3 = st.columns(3)
    c1.info(f"👤 {name if name else 'Not provided'}")
    c2.info(f"🎂 {age} years")
    c3.info(f"⚧ {gender}")

    # ── Symptoms Used ────────────────────────────────────
    st.subheader("🩹 Symptoms Used in Diagnosis")
    chips_html = "".join(
        f'<span class="symptom-chip">✓ {s.title()}</span>'
        for s in final_symptoms
    )
    st.markdown(f"<div style='margin:0.5rem 0'>{chips_html}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Top 5 Diseases ───────────────────────────────────
    st.subheader("🔬 Top 5 Possible Diagnoses")

    for rank, (disease, prob) in enumerate(results[:5], 1):
        label    = f"{rank}. {disease.title()}"
        pct_disp = f"{prob * 100:.2f}%"
        col_l, col_r = st.columns([4, 1])
        with col_l:
            st.markdown(f"**{label}**")
            st.progress(float(max(min(prob, 1.0), 0.0)))
        with col_r:
            st.write(pct_disp)

    # ── Uncertainty Warning ──────────────────────────────
    if len(results) >= 2:
        top1_prob = results[0][1]
        top2_prob = results[1][1]
        if abs(top1_prob - top2_prob) < 0.10:
            st.markdown(
                f"""<div class="uncertainty-box">
                ⚠️ <strong>Ambiguous result:</strong> The top two diagnoses
                (<em>{results[0][0].title()}</em> and <em>{results[1][0].title()}</em>)
                have very similar probabilities ({top1_prob*100:.1f}% vs {top2_prob*100:.1f}%).
                Consider adding more specific symptoms or consulting a healthcare professional.
                </div>""",
                unsafe_allow_html=True
            )

    st.divider()

    # ── Primary Result ───────────────────────────────────
    st.subheader("🩺 Most Likely Diagnosis")
    st.success(f"**{predicted_disease.title()}**")

    # Confidence calibration: 60 / 35 thresholds
    st.progress(float(max(min(confidence, 1.0), 0.0)))
    pct_str = f"{confidence * 100:.2f}%"

    if confidence >= 0.60:
        st.success(f"🟢 High confidence — {pct_str}")
    elif confidence >= 0.35:
        st.warning(f"🟡 Moderate confidence — {pct_str}")
    else:
        st.error(
            f"🔴 Low confidence — {pct_str}. "
            "Try adding more specific symptoms or consult a healthcare professional."
        )

    # ── Optional disease description ────────────────────
    if disease_info_map:
        desc = disease_info_map.get(predicted_disease.lower())
        if desc:
            with st.expander("ℹ️ About this condition"):
                st.write(desc)

    st.divider()

    # ── Actions ─────────────────────────────────────────
    col_restart, col_export = st.columns([1, 3])
    with col_restart:
        if st.button("🔄 New Diagnosis", type="primary"):
            for key in ["stage", "selected_symptoms", "followup_answers",
                        "followup_symptoms", "intermediate_results"]:
                del st.session_state[key]
            st.rerun()

    # ── Disclaimer ───────────────────────────────────────
    st.warning(
        "⚠️ **Educational project only.** "
        "This tool is not a substitute for professional medical advice, "
        "diagnosis, or treatment. Always consult a qualified health professional."
    )
    st.caption(
        "Built with Streamlit · BernoulliNB · 246K medical records · 754 diseases · 377 symptoms"
    )