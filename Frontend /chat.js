document.getElementById('send-button').addEventListener('click', sendMessage);

function appendMessage(sender, message) {
    const messagesDiv = document.getElementById('messages');
    const messageParagraph = document.createElement('p');
    messageParagraph.textContent = `${sender}: ${message}`;
    messagesDiv.appendChild(messageParagraph);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendMessage() {
    const userInputField = document.getElementById('user-input');
    const userInput = userInputField.value.trim();

    if (userInput === '') return;

    appendMessage('You', userInput);
    userInputField.value = '';

    try {
        const response = await fetch('http://localhost:8000/api/handle_query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userInput })
        });

        const data = await response.json();
        appendMessage('Agent', data.response);
    } catch (error) {
        appendMessage('Agent', 'Sorry, there was an error processing your request.');
        console.error('Error:', error);
    }
}
