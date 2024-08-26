document.getElementById('irrigation-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const formDataObj = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('http://127.0.0.1:5000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formDataObj),
        });

        const result = await response.json();
        if (result.error) {
            document.getElementById('result').textContent = 'Error: ' + result.error;
        } else {
            document.getElementById('result').textContent = `Predicted Active Root Zone: ${result.prediction}`;
        }

    } catch (error) {
        document.getElementById('result').textContent = 'Error: ' + error.message;
    }
});
