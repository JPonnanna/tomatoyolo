import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from pathlib import Path
from PIL import Image
import pandas as pd
st.set_page_config(page_title="🍅 Tomato Classifier", layout="wide")

# Load the model once during app startup
def load_model():
    model_path = 'best_tomato_model.pt'  # Update with your model path
    model = YOLO(model_path)
    return model

model = load_model()

# Blacken "nottomato" bounding boxes in the image
def blacken_nottomato_bboxes(results, img_path):
    img = cv2.imread(img_path)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        coords = box.xyxy[0].tolist()
        
        if class_name == 'nottomato':
            x_min, y_min, x_max, y_max = map(int, coords)
            img[y_min:y_max, x_min:x_max] = [0, 0, 0]
    return img

# Predict and process the image
def predict_image(model, image_path, conf=0.3):
    results = model.predict(source=image_path, conf=conf, save=False)
    img = cv2.imread(image_path)
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf_score = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        x_min, y_min, x_max, y_max = map(int, coords)

        # Draw bounding box
        color = (0, 255, 255) if class_name == 'objects' else (255, 0, 0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Put label and confidence
        label = f"Tomato {conf_score:.2f}" if class_name == 'objects' else f"{class_name} {conf_score:.2f}"
        cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Convert color format for Streamlit display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    black = blacken_nottomato_bboxes(results, image_path)
    black= cv2.cvtColor(black, cv2.COLOR_BGR2RGB)
    return results, img_rgb, black

# Streamlit app interface
st.title("🍅 Tomato vs. Nottomato Classifier")

# Apply CSS for better design
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .st-expander {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# State management
if 'show_details' not in st.session_state:
    st.session_state.show_details = False
if 'show_more' not in st.session_state:
    st.session_state.show_more = False
if 'show_about' not in st.session_state:
    st.session_state.show_about = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'blackened_img' not in st.session_state:
    st.session_state.blackened_img = None
if 'image_path' not in st.session_state:
    st.session_state.image_path = None

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded image temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file.write(uploaded_file.getbuffer())
    st.session_state.image_path = temp_file.name

    # Predict and display results
    st.write("🔍 Classifying the image...")
    results, pred, blackened_img = predict_image(model, st.session_state.image_path)
    st.session_state.results = results
    st.session_state.blackened_img = blackened_img

    # Display Original and Predicted images
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.image_path, caption="🖼️ Original Image", use_container_width=True)
    with col2:
        st.image(pred, caption="📌 Predicted Image", use_container_width=True)

    st.success("✅ Classification Complete!")

    # Action buttons (one below another)
    # Toggle for Details
    with st.expander("🔍 Prediction Details"):
        st.write("### Prediction Details:")
        objects_count = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            st.write(f"**Class:** {class_name}, **Confidence:** {conf:.2f}, **Box:** {coords}")
            if class_name == "objects":
                objects_count += 1
        st.write(f"### Total tomatoes detected (Conf>30%): **{objects_count}**")
    
    # Toggle for More (Blackened image display)
    with st.expander("🖤 More..."):
        st.image(blackened_img, caption="🖤 Blackened Nottomato Image", use_container_width=True)
    
    # About Model and Training Details
import matplotlib.pyplot as plt

# Model Summary
with st.expander("ℹ️ About Model"):
    st.write("### Model Summary:")
    st.text('A YOLO-based Classification Model that can identify Tomato and NotTomato classes')
    st.code(str(model))

# Training Details
with st.expander("📊 Training Details"):
    st.write("### Training Data:")
    csv_path = "training.csv" 
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)

    #import streamlit as st

    st.header("Test Set Evaluation Metrics")
    st.metric("Precision", "74.3%")
    st.metric("Recall", "67.4%")
    st.metric("mAP@0.5", "73.1%")
    


# Visualizations
with st.expander("📈 Visualizations"):
    csv_path = "training.csv" 
    data = pd.read_csv(csv_path)
    loss_columns = ['cls_loss', 'box_loss', 'dfl_loss']

    # Plot loss over epochs
    st.subheader("Loss Over Epochs")
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in loss_columns:
        ax.plot(data['epoch'], data[col], label=col)
        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Over Epochs')
    ax.legend()
    st.pyplot(fig)
    
    # Precision, Recall, mAP Visualizations
    st.subheader("Precision, Recall Visualizations")
    
    # Plot Precision and Recall over epochs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['epoch'], data['Precision'], label='Precision', color='blue')
    ax.plot(data['epoch'], data['Recall'], label='Recall', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Precision & Recall Over Epochs')
    ax.legend()
    st.pyplot(fig)
    
    # Plot mAP50 and mAP50-95 over epochs
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(data['epoch'], data['mAP50'], label='mAP50', color='orange')
    # ax.plot(data['epoch'], data['mAP50-95'], label='mAP50-95', color='red')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Value')
    # ax.set_title('mAP50 & mAP50-95 Over Epochs')
    # ax.legend()
    # st.pyplot(fig)
