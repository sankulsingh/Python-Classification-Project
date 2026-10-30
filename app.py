
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Research Abstract Classifier",
    page_icon="📚",
    layout="wide"
)

MODEL_DIR = Path(__file__).parent / "final_model"

@st.cache_resource
def load_model():
    classifier = joblib.load(
        MODEL_DIR / "svm_classifier.joblib"
    )
    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )
    return classifier, vectorizer

st.title("📚 Research Abstract Classification")
st.write(
    "Classify research-paper abstracts into Science, "
    "Business, or Social Science."
)

try:
    classifier, vectorizer = load_model()
except Exception as exc:
    st.error(
        "Could not load the trained model. Check that both "
        "model files exist in the final_model folder."
    )
    st.stop()

tab_single, tab_csv = st.tabs(
    ["Classify one abstract", "Upload a CSV"]
)

with tab_single:
    abstract = st.text_area(
        "Enter a research-paper abstract",
        height=200,
        placeholder="Paste the abstract here..."
    )

    if st.button("Predict category", key="single_predict"):
        if not abstract.strip():
            st.warning("Please enter an abstract first.")
        else:
            features = vectorizer.transform([abstract])
            prediction = classifier.predict(features)[0]
            st.success(f"Predicted category: **{prediction}**")

with tab_csv:
    uploaded_file = st.file_uploader(
        "Upload a CSV containing research abstracts",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data")
            st.dataframe(data.head())

            if data.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                column = st.selectbox(
                    "Select the column containing abstracts",
                    options=list(data.columns)
                )

                if st.button(
                    "Classify CSV",
                    key="csv_predict"
                ):
                    abstracts = data[column].fillna("").astype(str)
                    valid = abstracts.str.strip().ne("")

                    if not valid.any():
                        st.warning(
                            "No non-empty abstracts were found."
                        )
                    else:
                        predictions = [None] * len(data)
                        features = vectorizer.transform(
                            abstracts[valid].tolist()
                        )
                        predicted_labels = classifier.predict(features)

                        for index, label in zip(
                            data.index[valid], predicted_labels
                        ):
                            predictions[data.index.get_loc(index)] = label

                        result = data.copy()
                        result["predicted_label"] = predictions

                        st.subheader("Predictions")
                        st.dataframe(result)

                        csv_bytes = result.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download predictions as CSV",
                            data=csv_bytes,
                            file_name="research_abstract_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as exc:
            st.error(f"Could not read or classify this CSV: {exc}")
