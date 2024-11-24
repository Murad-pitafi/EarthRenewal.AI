import streamlit as st
import joblib
import os
# --- Function for Home Page ---
def home_page():
    st.title("Welcome to the Multi-Page App!")
    st.write("This is the home page where we introduce the app and its functionality.")
    # st.image("D:/EarthRenewal.AI/Frontend/logo.png", width=200)  # Example image on the homepage
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
# import joblib
# import os

def load_model():
    # Use a relative path to ensure portability in deployment
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = joblib.load(model_path)
    return model

# --- Function to make prediction ---
def make_prediction(model, input_data):
    prediction = model.predict([input_data])
    return prediction

# --- Function to Create a Custom Navigation Bar ---
def navbar():
    st.markdown("""
    <style>
        .navbar {
            background-color: #2d6a4f;
            padding: 10px;
            text-align: center;
        }
        .navbar a {
            color: white;
            padding: 14px 20px;
            text-decoration: none;
            font-size: 17px;
            margin: 0 10px;
        }
        .navbar a:hover {
            background-color: #1b5e20;
        }
    </style>
    <div class="navbar">
        <a href="javascript:void(0);" onclick="window.location.href='/Home'">Home</a>
        <a href="javascript:void(0);" onclick="window.location.href='/Chatbot'">Chatbot</a>
        <a href="javascript:void(0);" onclick="window.location.href='/SoilPrediction'">Soil Prediction</a>
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
