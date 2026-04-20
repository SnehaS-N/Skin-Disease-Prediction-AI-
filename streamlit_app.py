import streamlit as st
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io

st.set_page_config(page_title="Skin Disease Detection Dashboard", page_icon="🩺", layout="wide")

API_URL = "http://127.0.0.1:5000/predict"

def create_dummy_heatmap(image: Image.Image):
    img = image.resize((400, 300))
    img_np = np.array(img)

    h, w = img_np.shape[:2]
    y, x = np.mgrid[0:h, 0:w]

    center_x = int(w * 0.58)
    center_y = int(h * 0.42)

    heat = np.exp(-(((x - center_x) ** 2) / (2 * (w * 0.10) ** 2) +
                    ((y - center_y) ** 2) / (2 * (h * 0.14) ** 2)))

    heat += 0.30 * np.exp(-(((x - int(w * 0.35)) ** 2) / (2 * (w * 0.08) ** 2) +
                             ((y - int(h * 0.68)) ** 2) / (2 * (h * 0.10) ** 2)))

    heat = heat / heat.max()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img_np)
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.axis("off")
    ax.set_title("Simulated Grad-CAM Heatmap")
    return fig

def generate_report(patient_name, age, gender, symptoms, result):
    precautions_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(result.get("precautions", []))])

    report = f"""
AI Dermatology Report
---------------------
Patient Name: {patient_name}
Age: {age}
Gender: {gender}
Date: {datetime.now().strftime("%d-%m-%Y %H:%M")}

Prediction: {result.get('prediction', 'N/A')}
Confidence: {result.get('confidence', 'N/A')}%

Description:
{result.get('description', 'N/A')}

Symptoms / Notes:
{symptoms}

Precautions:
{precautions_text}

Warning:
{result.get('warning', 'No warning')}

Disclaimer:
{result.get('note', 'This AI tool is for preliminary assistance only and not a final medical diagnosis.')}
"""
    return report.strip()

st.title("🩺 Skin Disease Detection Dashboard")
st.caption("Upload image and get real prediction from your trained model.")

col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Patient Details")
    patient_name = st.text_input("Patient Name", "Sneha Patil")
    age = st.text_input("Age", "24")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    symptoms = st.text_area("Symptoms / Clinical Notes", "Itching, redness, dry inflamed patch on forearm")
    uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])
    run_detection = st.button("Run Detection")

with col2:
    st.subheader("Image Preview")
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Skin Image", use_container_width=True)
    else:
        st.info("Upload an image to preview.")

if run_detection:
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        try:
            uploaded_file.seek(0)
            files = {
                "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            response = requests.post(API_URL, files=files)
            result = response.json()

            if response.status_code == 200 and result.get("success"):
                st.markdown("---")
                st.subheader("Detection Result")

                r1, r2 = st.columns([1, 1])

                with r1:
                    st.success(f"Prediction: {result['prediction']}")
                    st.metric("Confidence", f"{result['confidence']}%")
                    st.progress(int(float(result["confidence"])))

                    st.write("**Description**")
                    st.write(result["description"])

                    st.write("**Precautions**")
                    for item in result["precautions"]:
                        st.write(f"- {item}")

                    if result.get("warning"):
                        st.warning(result["warning"])

                    st.info(result["note"])

                with r2:
                    st.write("**Clinical Summary**")
                    st.write(f"**Patient:** {patient_name}")
                    st.write(f"**Age:** {age}")
                    st.write(f"**Gender:** {gender}")
                    st.write(f"**Date:** {datetime.now().strftime('%d %B %Y')}")
                    st.write("**Symptoms / Notes:**")
                    st.write(symptoms)

                st.markdown("---")
                st.subheader("Explainable AI Heatmap")

                fig = create_dummy_heatmap(image)
                st.pyplot(fig)

                st.markdown("---")
                st.subheader("Download Report")

                report_text = generate_report(
                    patient_name=patient_name,
                    age=age,
                    gender=gender,
                    symptoms=symptoms,
                    result=result
                )

                st.download_button(
                    label="Download Report as TXT",
                    data=report_text,
                    file_name="skin_disease_report.txt",
                    mime="text/plain"
                )
            else:
                st.error(f"Prediction failed: {result}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.markdown("---")
    st.info("Fill details, upload image, and click Run Detection.")