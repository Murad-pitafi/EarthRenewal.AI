import streamlit as st
import joblib
import os
import requests

st.set_page_config(layout="wide")
# --- Function for Home Page ---
def home_page():
    st.markdown(
        """
        <style>
        .home-page {
            background-color: #edecec;
            height: 100vh; /* Full viewport height */
            width: 100wh;  /* Full viewport width */
            padding: 20px;
            padding-right: 0rem;
            padding-left: 0rem;
            left: 0;
            display: flex;
            flex-direction: column; /*
            justify-content: center; /* Center the content vertically */
            align-items: center; /* Center the content horizontally */
            box-sizing: border-box;
        }
        .home-page h1 {
            color: #2d6a4f;
        }
        .home-page p {
            color: #2d6a4f;
            font-size: 18px;
        }
        </style>
        <div class="home-page">
            <h1>Our Goal</h1>
            <p>Adopting innovative techniques and leveraging advanced technologies, sustainable agriculture aims to enhance productivity, conserve natural resources, and promote long-term environmental stability.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Function for Chatbot Page ---
def chatbot_page():
    st.title("Chatbot Section")
    st.write("Chat with our AI-powered chatbot here.")
    
    user_input = st.text_input("Ask me anything:")
    
    if user_input:
        st.write(f"You asked: {user_input}")
        st.write("Chatbot is processing your request...")
        
        # Call the Flask API to get the chatbot's response
        try:
            response = requests.post("http://127.0.0.1:5000/api/ask", json={"question": user_input})
            if response.status_code == 200:
                result = response.json().get('answer')
                st.write(f"Chatbot response: {result}")
            else:
                st.write("Error: Unable to get response from the chatbot API.")
        except Exception as e:
            st.write(f"An error occurred while contacting the chatbot API: {e}")

# --- Function for Soil Prediction Page ---
def soil_prediction_page():
    st.title("Soil Monitoring and Prediction")
    st.write("This section helps predict soil conditions using AI models.")
    
    # User input for soil monitoring features
    air_temperature = st.number_input("Air Temperature (°C)", min_value=-50.0, max_value=50.0, value=28.9)
    soil_temperature = st.number_input("Soil Temperature (°C)", min_value=-50.0, max_value=50.0, value=26.81)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=42.0)
    moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=60.74)
    nitrogen = st.number_input("Nitrogen (ppm)", min_value=0.0, value=4.97)
    phosphorous = st.number_input("Phosphorous (ppm)", min_value=0.0, value=32.24)
    potassium = st.number_input("Potassium (ppm)", min_value=0.0, value=24.44)
    
    # Bundle input data into a list for prediction
    input_data = [air_temperature, soil_temperature, humidity, moisture, nitrogen, phosphorous, potassium]

    # Load the model
    model = load_model()

    # Make prediction when the user clicks the button
    if st.button("Predict Soil Health"):
        prediction = make_prediction(model, input_data)
        
        # Display prediction result
        if prediction == 0:
            st.write("Predicted Soil Health: **Poor Soil Health**")
        elif prediction == 1:
            st.write("Predicted Soil Health: **Moderate Soil Health**")
        else:
            st.write("Predicted Soil Health: **Good Soil Health**")

# --- Load the pre-trained model ---
def load_model():
    # Use a relative path to ensure portability in deployment
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(model_path)
    return model

# --- Function to make prediction ---
def make_prediction(model, input_data):
    prediction = model.predict([input_data])
    return prediction

# --- Function to Create a Custom Header and Navigation Bar ---
hide_st_style = """
            <style>
            #MainMenu {visibilty: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
def navbar():
    st.markdown("""
    <style>
        body {
            background-color: #fff;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #2d6a4f;
            padding: 10px;
            color: #edecec;
            position: fixed;
            top: 0;
            left:0;
            width: 100%;
            z-index: 1000;
            padding-right: 50px;
            margin=0;
            box-sizing: border-box;
        }
        .header img {
            height: 50px;
        }
        .header .project-name {
            font-size: 24px;
            font-weight: bold;
            margin-left: 10px;
        }
        .header .nav-links {
            display: flex;
            gap: 15px;
        }
        .header .nav-links a {
            color: white;
            text-decoration: none;
            font-size: 16px;
        }
        .header .nav-links a:hover {
            text-decoration: underline;
        }
    .navbar-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .navbar-buttons button {
            padding: 8px 16px;
            font-size: 16px;
            background-color: #2d6a4f;
            color: white;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .navbar-buttons button:hover {
            background-color: #d6e865;
        }
        .navbar-buttons button:active {
            background-color: #9edd6c;
        }
        .navbar-buttons button.selected {
            background-color: #9edd6c;
        }
    </style>
    <div class="header">
        <div style="display: flex; align-items: center;">
            <img src="Frontend/logo.png" alt="Logo" style="height:50px;">
            <span class="project-name">EarthRenewal.AI</span>
        </div>
        <div class="navbar-buttons">
            <button id="homeBtn">Home</button>
            <button id="chatbotBtn">Chatbot</button>
            <button id="soilBtn">Soil Monitoring</button>
            <button id="contactBtn">Contact Us</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create buttons for navbar using Streamlit's button widget
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('Home'):
            st.session_state.page = "Home"
            st.markdown('<style> #homeBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)
    with col2:
        if st.button('Chatbot'):
            st.session_state.page = "Chatbot"
            st.markdown('<style> #chatbotBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)
    with col3:
        if st.button('Soil Monitoring'):
            st.session_state.page = "Soil Prediction"
            st.markdown('<style> #soilBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)
    with col4:
        if st.button('Contact Us'):
            st.session_state.page = "Contact Us"
            st.markdown('<style> #contactBtn {background-color: #9edd6c;} </style>', unsafe_allow_html=True)

# --- Main Function to Handle Navigation ---
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    navbar()  # Display the navbar at the top
    #  home_page()  # Display the home page by default
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Chatbot":
        chatbot_page()
    elif st.session_state.page == "Soil Prediction":
        soil_prediction_page()
    # elif st.session_state.page == "Contact Us":
    #     contact_page()

# --- Run the Streamlit app ---
if __name__ == "__main__":
    main()