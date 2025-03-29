import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def get_selected_features():
    selector = pickle.load(open("model/selector.pkl", "rb"))
    data = get_clean_data()
    feature_names = data.drop(['diagnosis'], axis=1).columns
    selected_features = feature_names[selector.get_support()]
    return selected_features.tolist()

def add_sidebar(selected_features):
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    input_dict = {}

    for key in selected_features:
        label = key.replace("_", " ").capitalize()
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = list(input_data.keys())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(input_data.values()),
        theta=categories,
        fill='toself',
        name='Input Features'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig
def add_info_box():
    st.sidebar.markdown("### ℹ️ About the Measurements")
    st.sidebar.info(
        "These measurements are obtained from breast tissue samples collected via **Fine Needle Aspiration (FNA)**. "
    )


def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)  

    selected_features = get_selected_features()
    add_info_box()
    input_data = add_sidebar(selected_features)

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This Breast Cancer Predictor uses a machine learning model trained on cell nuclei measurements from breast tissue samples to classify whether a mass is benign (non-cancerous) or malignant (cancerous). By adjusting the measurements using the sliders, you can simulate different patient data and observe the prediction results. This tool is designed to assist medical professionals in supporting diagnosis.")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()
