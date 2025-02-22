def detect_intent(text):
    keywords = {"weather": "Weather Inquiry", "crop": "Crop Inquiry"}
    for word, intent in keywords.items():
        if word in text.lower():
            return intent
    return "General Inquiry"
