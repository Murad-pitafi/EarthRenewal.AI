import streamlit as st
import joblib
import os
import requests

# --- Function for Home Page ---
def home_page():
    st.title("Welcome to EarthRenewal.AI")
    st.write("About Us")
    #st.image("Frontend/logo.png", width=200)  # Example image on the homepage
    st.write("Navigate to other sections using the navigation bar above.")

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
            color: white;
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
    </style>
    <div class="header">
        <div style="display: flex; align-items: center;">
            <img src="Frontend/logo.png" alt="Logo">
            <span class="project-name">EarthRenewal.AI</span>
        </div>
        <div class="nav-links">
            <a href="javascript:void(0);" onclick="window.location.href='/Chatbot'">Chatbot</a>
            <a href="javascript:void(0);" onclick="window.location.href='/SoilPrediction'">Soil Monitoring</a>
            <a href="javascript:void(0);" onclick="window.location.href='/ContactUs'">Contact Us</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Function to Handle Navigation ---
def main():
    navbar()  # Display the navbar at the top
    page = st.selectbox("Select a Page", ["Home", "Chatbot", "Soil Prediction"])

    if page == "Home":
        home_page()
    elif page == "Chatbot":
        chatbot_page()
    elif page == "Soil Prediction":
        soil_prediction_page()

# --- Run the Streamlit app ---
if __name__ == "__main__":
    main()