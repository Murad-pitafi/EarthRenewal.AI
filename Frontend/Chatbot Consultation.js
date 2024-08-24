document.getElementById("send-btn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevents form submission if you're using a form
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById("user-input");
    const chatBody = document.getElementById("chat-body");

    if (userInput.value.trim() !== "") {
        // Create and append user's message
        const userMessage = document.createElement("div");
        userMessage.classList.add("chat-message", "user-message");
        userMessage.textContent = userInput.value;
        chatBody.appendChild(userMessage);

        // Scroll to the bottom of the chat body
        chatBody.scrollTop = chatBody.scrollHeight;

        // Clear the input
        userInput.value = "";

        // Create and append typing indicator
        const typingIndicator = document.createElement("div");
        typingIndicator.classList.add("chat-message", "typing-indicator");
        typingIndicator.textContent = "Bot is typing...";
        chatBody.appendChild(typingIndicator);

        // Scroll to the bottom of the chat body again
        chatBody.scrollTop = chatBody.scrollHeight;

        // Remove typing indicator and show bot response after a delay
        setTimeout(() => {
            chatBody.removeChild(typingIndicator); // Remove typing indicator

            const botMessage = document.createElement("div");
            botMessage.classList.add("chat-message", "bot-message");
            botMessage.textContent = "Smart irrigation refers to the use of advanced technologies, such as sensors, AI, and automated systems, to optimize the use of water in agriculture. Unlike traditional irrigation systems that operate on fixed schedules, smart irrigation systems adapt to real-time conditions like soil moisture levels, weather forecasts, and plant water needs to deliver water precisely where and when it’s needed. This not only conserves water but also improves crop yield and health by ensuring that plants receive the right amount of water at the right time.";
            chatBody.appendChild(botMessage);

            // Scroll to the bottom of the chat body
            chatBody.scrollTop = chatBody.scrollHeight;
        }, 3000); // Delay of 3000 milliseconds (3 seconds)
    }
}
