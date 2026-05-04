import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Logistic Regression Dashboard", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#a78bfa;'>
🚀 Logistic Regression Interactive Dashboard
</h1>
<hr>
""", unsafe_allow_html=True)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body { background-color:#0f172a; color:white; }
.block-container { padding:2rem; }

.card {
    background:#1e293b;
    padding:25px;
    border-radius:15px;
    margin-bottom:25px;
    box-shadow:0 8px 25px rgba(0,0,0,0.4);
    transition:0.3s;
}
.card:hover {
    transform:scale(1.01);
}

h1,h2,h3 { color:#a78bfa; }

/* 🔥 BUTTON STYLE */
.stButton > button {
    background:linear-gradient(45deg,#7c3aed,#3b82f6);
    color:white;
    border-radius:10px;
    height:45px;
    font-weight:bold;
}

/* 🔥 SIDEBAR STYLE (ADD HERE) */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] h2 {
    color: #a78bfa;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR NAVBAR ----------------
pages = [
    "Theory","Dataset","Preprocessing","EDA",
    "Training","Prediction","Evaluation"
]

if "page" not in st.session_state:
    st.session_state.page = "Theory"

with st.sidebar:
    st.markdown("## 📊 Navigation")

    for p in pages:
        if st.button(
            p,
            key=p,
            use_container_width=True,
            type="primary" if st.session_state.page == p else "secondary"
        ):
            st.session_state.page = p
            st.rerun()

page = st.session_state.page

# ---------------- SESSION ----------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.model = None
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.scaler = None
    st.session_state.X_test = None
    st.session_state.y_test = None

# ---------------- THEORY ----------------
# ---------------- THEORY ----------------
if page == "Theory":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.title("📘 Logistic Regression - Interactive Theory")

    tabs = st.tabs([
        "📖 Basics",
        "📐 Math",
        "📊 Visualization",
        "⚙️ Working",
        "🧠 Intuition"
    ])

    # ---------------- TAB 1 ----------------
    with tabs[0]:
        st.subheader("What is Logistic Regression?")

        st.write("""
        Logistic Regression is a **supervised machine learning algorithm**  
        used for **classification problems**.
        """)

        st.info("👉 It predicts probability between 0 and 1")

        st.markdown("""
        **Applications:**
        - Spam detection  
        - Disease prediction  
        - Loan approval  
        """)

    # ---------------- TAB 2 ----------------
    with tabs[1]:
        st.subheader("Mathematical Model")

        st.latex(r"z = w_1x_1 + w_2x_2 + ... + b")

        st.write("Then apply sigmoid:")

        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")

        st.subheader("Loss Function")

        st.latex(r"L = -[y \log(p) + (1-y)\log(1-p)]")

    # ---------------- TAB 3 ----------------
    with tabs[2]:
        st.subheader("Sigmoid Function Visualization")

        x = np.linspace(-10, 10, 100)

        # 👉 interactive slider
        scale = st.slider("Adjust Curve Stretch", 0.5, 5.0, 1.0)

        y = 1 / (1 + np.exp(-scale * x))

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Sigmoid Curve")
        ax.set_xlabel("z")
        ax.set_ylabel("Probability")

        st.pyplot(fig)

        st.success("👉 Notice how output stays between 0 and 1")

    # ---------------- TAB 4 ----------------
    with tabs[3]:
        st.subheader("How Model Works")

        st.markdown("""
        **Step-by-step flow:**

        1. Input features (X)  
        2. Multiply with weights  
        3. Add bias  
        4. Apply sigmoid  
        5. Get probability  
        6. Apply threshold (0.5)  
        """)

        st.code("""
z = w1*x1 + w2*x2 + b
p = 1 / (1 + e^-z)

if p > 0.5:
    class = 1
else:
    class = 0
        """)

    # ---------------- TAB 5 ----------------
    with tabs[4]:
        st.subheader("Intuition")

        st.write("""
        Logistic Regression is like drawing a **boundary line**
        between two classes.
        """)

        st.warning("👉 It does NOT predict values, it predicts class probability")

        with st.expander("📌 Binary vs Multi-class"):
            st.write("""
            Binary → 2 classes  
            Multi-class → One-vs-Rest approach  
            """)

        with st.expander("📌 Linear vs Logistic"):
            st.write("""
            Linear → continuous output  
            Logistic → probability output  
            """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DATASET ----------------
# ---------------- DATASET ----------------
elif page == "Dataset":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.title("📂 Dataset Input")

    # 🔹 FILE UPLOAD (FIRST DEFINE THIS)
    file = st.file_uploader("Upload CSV")

    if file is not None:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("Dataset loaded")

    # 🔥 SAMPLE DATASET BUTTON
    if st.button("📊 Load Sample Dataset"):

        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

        st.session_state.df = df

        st.success("✅ Sample dataset loaded")

    # 🔹 DISPLAY DATA
    if st.session_state.df is not None:
        st.subheader("Preview")
        st.dataframe(st.session_state.df.head())

        st.write("📏 Shape:", st.session_state.df.shape)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREPROCESSING ----------------
# ---------------- PREPROCESSING ----------------
# ---------------- PREPROCESSING ----------------
elif page == "Preprocessing":

    if st.session_state.df is None:
        st.warning("⚠️ Upload dataset first")
    else:
        df = st.session_state.df.copy()

        st.subheader("🎯 Select Target Column")

        target_col = st.selectbox("Choose target column", df.columns)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 🔥 HANDLE CONTINUOUS TARGET
        if y.dtype != 'object':
            if y.nunique() > 10:
                st.warning("⚠️ Target seems continuous → converting to binary")
                y = (y > y.median()).astype(int)

        # 🔥 ENCODE CATEGORICAL FEATURES
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # 🔥 SCALE FEATURES
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 🔥 SAVE EVERYTHING
        st.session_state.X = X_scaled
        st.session_state.y = y
        st.session_state.scaler = scaler
        st.session_state.feature_names = X.columns.tolist()

        st.success("✅ Preprocessing completed successfully")

        # 🔹 SHOW PREVIEW
        st.subheader("Preview (Scaled Features)")
        st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

# ---------------- EDA ----------------
elif page == "EDA":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.title("📊 Exploratory Data Analysis")

    df = st.session_state.df

    if df is None:
        st.warning("Upload dataset first")
    else:
        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            annot_kws={"size": 8},
            ax=ax
        )

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TRAINING ----------------
# ---------------- TRAINING ----------------
elif page == "Training":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.title("🧠 Model Training")

    if st.session_state.X is None:
        st.warning("⚠️ Please run preprocessing first")

    else:
        test_size = st.slider("Test Size (%)", 10, 50, 30) / 100

        if st.button("🚀 Train Model"):

            # ---------------- SPLIT ----------------
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X,
                st.session_state.y,
                test_size=test_size,
                random_state=42
            )

            # ---------------- MODEL ----------------
            model = LogisticRegression(max_iter=1000)

            # 🔥 SPINNER
            with st.spinner("⏳ Training model... please wait"):
                model.fit(X_train, y_train)

            # ---------------- SAVE ----------------
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train

            st.success("✅ Model trained successfully!")

            # ---------------- PARAMETERS ----------------
            st.subheader("📊 Learned Parameters")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Coefficients:**")
                st.write(model.coef_)

            with col2:
                st.write("**Intercept:**")
                st.write(model.intercept_)

            # ---------------- CROSS VALIDATION ----------------
            st.subheader("🔁 Cross Validation")

            scores = cross_val_score(
                model,
                st.session_state.X,
                st.session_state.y,
                cv=5
            )

            st.write("Fold Scores:", scores)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Average Score", f"{scores.mean():.3f}")

            with col2:
                st.metric("Std Deviation", f"{scores.std():.3f}")

            # ---------------- MODEL INFO ----------------
            st.subheader("📘 Model Info")

            st.write("""
            Logistic Regression learns weights for each feature.

            z = w1*x1 + w2*x2 + ... + b  
            probability = 1 / (1 + e^(-z))
            """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
    if "feature_names" not in st.session_state:
        st.warning("⚠️ Please run preprocessing first")
        st.stop()
# ---------------- PREDICTION ----------------
    # ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🔮 Prediction")

    # ✅ SAFETY CHECKS
    if st.session_state.model is None:
        st.warning("⚠️ Train model first")
        st.stop()

    if st.session_state.scaler is None:
        st.warning("⚠️ Run preprocessing first")
        st.stop()

    if "feature_names" not in st.session_state:
        st.warning("⚠️ Preprocessing not completed")
        st.stop()

    model = st.session_state.model

    st.subheader("Enter Feature Values")

    inputs = []

    # 🔥 generate inputs
    for name in st.session_state.feature_names:
        val = st.number_input(name, value=0.0)
        inputs.append(val)

    # 🔥 Predict button
    if st.button("🚀 Predict"):

        x = np.array(inputs).reshape(1, -1)

        # scale
        x_scaled = st.session_state.scaler.transform(x)

        pred = model.predict(x_scaled)[0]
        prob = model.predict_proba(x_scaled)[0]

        st.success(f"🎯 Predicted Class: {pred}")

        st.subheader("Probability")

        if len(prob) == 2:
            st.write(f"Class 0: {prob[0]:.3f}")
            st.write(f"Class 1: {prob[1]:.3f}")
            st.progress(float(prob[1]))
        else:
            for i, p in enumerate(prob):
                st.write(f"Class {i}: {p:.3f}")

    st.markdown('</div>', unsafe_allow_html=True)
    
# ---------------- EVALUATION ----------------
# ---------------- EVALUATION ----------------
elif page == "Evaluation":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.title("📊 Model Evaluation")

    if "model" not in st.session_state or st.session_state.model is None:
        st.warning("⚠️ Train the model first")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # 👉 IMPORTANT (fixes your errors)
        y_pred = model.predict(X_test)

        # ---------------- ROC CURVE ----------------
        st.subheader("📉 ROC Curve")

        unique_classes = len(set(y_test))

        if unique_classes == 2:
            y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], '--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()

            st.pyplot(fig_roc)
            st.success(f"AUC Score: {roc_auc:.3f}")

        else:
            st.warning("ROC Curve only works for binary classification")

        # ---------------- CONFUSION MATRIX ----------------
        st.subheader("🔲 Confusion Matrix")

        fig_cm, ax_cm = plt.subplots(figsize=(6,5))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap="Blues",
            cbar=False,
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # ---------------- PROBABILITY DISTRIBUTION ----------------
        if unique_classes == 2:
            st.subheader("📊 Prediction Probability Distribution")

            fig_prob, ax_prob = plt.subplots()
            ax_prob.hist(y_prob, bins=20, color='skyblue')
            ax_prob.set_xlabel("Predicted Probability")
            ax_prob.set_ylabel("Frequency")

            st.pyplot(fig_prob)

        # ---------------- CROSS VALIDATION ----------------
        st.subheader("🔁 Cross Validation")

        scores = cross_val_score(model, st.session_state.X, st.session_state.y, cv=5)

        st.write("Scores:", scores)
        st.success(f"Average CV Score: {scores.mean():.3f}")

        # ---------------- CLASSIFICATION REPORT ----------------
        st.subheader("📄 Classification Report")

        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # ---------------- EXPLANATION ----------------
        st.subheader("🧠 Interpretation")

        st.write("""
        - **Accuracy** → Overall correctness of model  
        - **Precision** → Correct positive predictions  
        - **Recall** → Ability to detect positives  
        - **F1 Score** → Balance of precision & recall  
        - **ROC Curve** → Model discrimination ability  
        - **AUC** → Higher value means better model  
        """)

    st.markdown('</div>', unsafe_allow_html=True)