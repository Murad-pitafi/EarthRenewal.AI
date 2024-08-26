// document.getElementById("send-btn").addEventListener("click", function() {
//     sendMessage();
// });

// document.getElementById("user-input").addEventListener("keypress", function(event) {
//     if (event.key === "Enter") {
//         event.preventDefault();
//         sendMessage();
//     }
// });

// document.getElementById("mic-btn").addEventListener("click", function() {
//     startVoiceRecognition();
// });

// async function sendMessage(query) {
//     const userInput = document.getElementById("user-input");
//     const chatBody = document.getElementById("chat-body");
//     const userText = query || userInput.value.trim();

//     if (userText !== "") {
//         // Create and append user's message
//         const userMessage = document.createElement("div");
//         userMessage.classList.add("chat-message", "user-message");
//         userMessage.textContent = userText;
//         chatBody.appendChild(userMessage);

//         chatBody.scrollTop = chatBody.scrollHeight;
//         userInput.value = "";

//         // Create and append typing indicator
//         const typingIndicator = document.createElement("div");
//         typingIndicator.classList.add("chat-message", "typing-indicator");
//         typingIndicator.textContent = "Bot is typing...";
//         chatBody.appendChild(typingIndicator);

//         chatBody.scrollTop = chatBody.scrollHeight;

//         try {
//             const response = await fetch('http://127.0.0.1:5000/api/ask', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ question: userText }),
//             });

//             const data = await response.json();
//             chatBody.removeChild(typingIndicator);

//             const botMessage = document.createElement("div");
//             botMessage.classList.add("chat-message", "bot-message");
//             botMessage.textContent = data.answer || "Sorry, I couldn't process your request.";
//             chatBody.appendChild(botMessage);

//         } catch (error) {
//             chatBody.removeChild(typingIndicator);

//             const botMessage = document.createElement("div");
//             botMessage.classList.add("chat-message", "bot-message");
//             botMessage.textContent = "There was an error connecting to the server.";
//             chatBody.appendChild(botMessage);
//         }

//         chatBody.scrollTop = chatBody.scrollHeight;
//     }
// }

// function startVoiceRecognition() {
//     if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
//         alert("Your browser does not support speech recognition.");
//         return;
//     }

//     const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
//     recognition.lang = 'en-US';

//     recognition.onstart = function() {
//         alert("Listening..."); // Notify the user that speech recognition has started
//     };

//     recognition.onresult = function(event) {
//         const transcript = event.results[0][0].transcript;
//         sendMessage(transcript);
//     };

//     recognition.onerror = function(event) {
//         alert("An error occurred during speech recognition: " + event.error);
//     };

//     recognition.start();
// }

// function goBack() {
//     window.history.back();
// }

document.getElementById("send-btn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});

document.getElementById("mic-btn").addEventListener("click", async function() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        let audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'user_audio.wav');

            try {
                const response = await fetch('http://127.0.0.1:5000/api/ask_audio', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                appendBotMessage(data.answer || "Sorry, I couldn't process your request.");
            } catch (error) {
                console.error("Error sending audio:", error);
                appendBotMessage("An error occurred while sending the audio.");
            }
        };

        mediaRecorder.start();
        setTimeout(() => {
            mediaRecorder.stop();
        }, 5000); // 5 seconds
    } catch (error) {
        console.error("Error accessing microphone:", error);
        alert("Could not access your microphone.");
    }
});

async function sendMessage() {
    const userInput = document.getElementById("user-input");
    const chatBody = document.getElementById("chat-body");

    if (userInput.value.trim() !== "") {
        appendUserMessage(userInput.value);

        const userText = userInput.value;
        userInput.value = "";

        try {
            const response = await fetch('http://127.0.0.1:5000/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: userText }),
            });

            const data = await response.json();
            appendBotMessage(data.answer || "Sorry, I couldn't process your request.");
        } catch (error) {
            appendBotMessage("There was an error connecting to the server.");
        }
    }
}

function appendUserMessage(message) {
    const chatBody = document.getElementById("chat-body");
    const userMessage = document.createElement("div");
    userMessage.classList.add("chat-message", "user-message");
    userMessage.textContent = message;
    chatBody.appendChild(userMessage);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function appendBotMessage(message) {
    const chatBody = document.getElementById("chat-body");
    const typingIndicator = document.querySelector(".typing-indicator");
    if (typingIndicator) {
        chatBody.removeChild(typingIndicator);
    }
    const botMessage = document.createElement("div");
    botMessage.classList.add("chat-message", "bot-message");
    botMessage.textContent = message;
    chatBody.appendChild(botMessage);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function goBack() {
    window.history.back();
}
