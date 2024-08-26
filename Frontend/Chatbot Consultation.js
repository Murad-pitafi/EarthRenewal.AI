document.getElementById("send-btn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});

document.getElementById("mic-btn").addEventListener("click", function() {
    startVoiceRecognition();
});

async function sendMessage(query) {
    const userInput = document.getElementById("user-input");
    const chatBody = document.getElementById("chat-body");
    const userText = query || userInput.value.trim();

    if (userText !== "") {
        // Create and append user's message
        const userMessage = document.createElement("div");
        userMessage.classList.add("chat-message", "user-message");
        userMessage.textContent = userText;
        chatBody.appendChild(userMessage);

        chatBody.scrollTop = chatBody.scrollHeight;
        userInput.value = "";

        // Create and append typing indicator
        const typingIndicator = document.createElement("div");
        typingIndicator.classList.add("chat-message", "typing-indicator");
        typingIndicator.textContent = "Bot is typing...";
        chatBody.appendChild(typingIndicator);

        chatBody.scrollTop = chatBody.scrollHeight;

        try {
            const response = await fetch('http://127.0.0.1:5000/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: userText }),
            });

            const data = await response.json();
            chatBody.removeChild(typingIndicator);

            const botMessage = document.createElement("div");
            botMessage.classList.add("chat-message", "bot-message");
            botMessage.textContent = data.answer || "Sorry, I couldn't process your request.";
            chatBody.appendChild(botMessage);

        } catch (error) {
            chatBody.removeChild(typingIndicator);

            const botMessage = document.createElement("div");
            botMessage.classList.add("chat-message", "bot-message");
            botMessage.textContent = "There was an error connecting to the server.";
            chatBody.appendChild(botMessage);
        }

        chatBody.scrollTop = chatBody.scrollHeight;
    }
}

function startVoiceRecognition() {
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
        alert("Your browser does not support speech recognition.");
        return;
    }

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';

    recognition.onstart = function() {
        alert("Listening..."); // Notify the user that speech recognition has started
    };

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        sendMessage(transcript);
    };

    recognition.onerror = function(event) {
        alert("An error occurred during speech recognition: " + event.error);
    };

    recognition.start();
}

function goBack() {
    window.history.back();
}
