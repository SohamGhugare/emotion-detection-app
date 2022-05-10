# Core pkgs
import streamlit as st
import altair

# EDA pkgs
import pandas as pd
import numpy as np

# Utils
import joblib

# Assets
from PIL import Image
prediction_img = Image.open("assets/prediction.png")
prediction_proba_img = Image.open("assets/prediction_proba.png")
visuals_img = Image.open("assets/visuals.png")

# Loading Model
with open("models/emotion_classifier_pipeline.pkl", "rb") as f:
    pipeline = joblib.load(f)

# Predict function
def predict_emotion(raw):
    return pipeline.predict([raw])[0]

def get_prediction_prob(raw):
    return pipeline.predict_proba([raw])

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():

    st.set_page_config(
        page_title="Emotion Detection App",
        page_icon="🤗",
        layout="wide",
        initial_sidebar_state="expanded",
        )


    st.title("Emotion Classifier App")

    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("This app detects a particular emotion from the input text.")

        with st.form(key="emotion_clf_form"):
            raw_text = st.text_area("Input text here")
            submit_text = st.form_submit_button(label="Submit")
        
        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotion(raw_text)
            probability = get_prediction_prob(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction} : {emoji_icon}")
                st.write(f"Confidence: {round(np.max(probability)*100)}%")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipeline.classes_).T.reset_index()
                proba_df.columns = ["Emotions", "Probability"]
                st.write(proba_df)

                figure = altair.Chart(proba_df).mark_bar().encode(x="Emotions", y="Probability", color="Emotions")
                st.altair_chart(figure, use_container_width=True)

    elif choice == "Monitor":
        st.subheader("Monitor App")
        st.write("Work in progress")

    else:
        st.subheader("About")
        st.write("End-to-end NLP app which detects emotions from provided texts. Also shows the confidence and probabilities of all the emotions predicted by the model.")

        # Images
        st.image(prediction_img, caption="Prediction with confidence")
        st.image(prediction_proba_img, caption="Prediction probabilities for all emotions")
        st.image(visuals_img, caption="Authentic visualizations")

        st.write("Made with ❤️ by Soham Ghugare")

if __name__ == '__main__':
    main()