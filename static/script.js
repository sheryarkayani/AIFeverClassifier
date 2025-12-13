document.getElementById('diagnosisForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('submitBtn');
    const resultCard = document.getElementById('resultCard');

    // Set loading state
    const originalBtnText = btn.innerHTML;
    btn.innerHTML = '<span><span class="spinner"></span> Analyzing Vitals...</span>';
    btn.style.opacity = '0.8';
    btn.disabled = true;

    // Hide previous results slightly
    if (!resultCard.classList.contains('hidden')) {
        resultCard.style.opacity = '0.5';
    }

    try {
        // Collect and parse data
        const data = {
            Age: parseInt(document.getElementById('age').value) || 0,
            Sex: parseInt(document.getElementById('sex').value),
            temperature: parseFloat(document.getElementById('temperature').value) || 0,
            wbc: parseFloat(document.getElementById('wbc').value) || 0,
            platelets: parseFloat(document.getElementById('platelets').value) || 0,
            headache: parseInt(document.getElementById('headache').value),
            joint_pain: parseInt(document.getElementById('joint_pain').value),
            rash: parseInt(document.getElementById('rash').value),
            travel_to_hot_area: parseInt(document.getElementById('travel').value),
            mosquito_exposure: parseInt(document.getElementById('mosquito').value)
        };

        // Artificial delay for UX (to show animation)
        await new Promise(r => setTimeout(r, 600));

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.statusText}`);
        }

        const result = await response.json();

        // Update Prediction Display
        const predictionEl = document.getElementById('predictionValue');
        predictionEl.textContent = result.prediction.replace(/-/g, ' ');

        // Populate Confidence Scores with Bars
        const probList = document.getElementById('probList');
        probList.innerHTML = '';

        // Sort: highest probability first
        const sortedProbs = Object.entries(result.probabilities)
            .sort(([, a], [, b]) => b - a);

        sortedProbs.forEach(([disease, prob], index) => {
            const percentage = (prob * 100).toFixed(1);
            const isTop = index === 0;

            const row = document.createElement('div');
            row.className = 'prob-row';
            if (isTop) row.style.background = 'rgba(255,255,255,0.1)';

            row.innerHTML = `
                <span class="prob-label" style="${isTop ? 'color:#4ade80' : ''}">${disease.replace(/-/g, ' ')}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: 0%; ${isTop ? 'background: #4ade80;' : ''}"></div>
                </div>
                <span class="prob-value">${percentage}%</span>
            `;
            probList.appendChild(row);

            // Animate bar after append
            setTimeout(() => {
                row.querySelector('.prob-bar').style.width = `${percentage}%`;
            }, 50);
        });

        // Show result
        resultCard.classList.remove('hidden');
        resultCard.style.opacity = '1';

        // Smooth scroll to result
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        console.error(error);
        alert('Diagnosis failed: ' + error.message);
    } finally {
        // Reset button
        btn.innerHTML = originalBtnText;
        btn.disabled = false;
        btn.style.opacity = '1';
    }
});
