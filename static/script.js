document.getElementById('diagnosisForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('submitBtn');
    const resultCard = document.getElementById('resultCard');

    // UI Loading state
    const originalText = btn.innerHTML;
    btn.innerHTML = 'Analyzing...';
    btn.disabled = true;
    resultCard.classList.add('hidden');

    try {
        // Collect Data
        // Features: 
        // 'Age', 'Sex', 'temperature', 'wbc', 'platelets', 
        // 'headache', 'joint_pain', 'rash', 'vomiting', 'fatigue', 'chills',
        // 'fever_pattern', 'travel_to_hot_area', 'mosquito_exposure', 
        // 'sun_exposure', 'hygiene_issue'

        const getCheck = (id) => document.getElementById(id).checked ? 1 : 0;
        const getVal = (id) => document.getElementById(id).value;
        const getNum = (id) => parseFloat(getVal(id)) || 0;

        const data = {
            // Demographics (Hidden/Defaulted in UI for now, preserving inputs)
            Age: parseInt(getVal('age')) || 30,
            Sex: parseInt(getVal('sex')) || 1,

            // Vitals
            temperature: getNum('temperature'),
            wbc: getVal('wbc') ? getNum('wbc') : null, // Optional
            platelets: getVal('platelets') ? getNum('platelets') : null, // Optional

            // Symptoms
            headache: getCheck('headache'),
            joint_pain: getCheck('joint_pain'),
            rash: getCheck('rash'),
            vomiting: getCheck('vomiting'),
            fatigue: getCheck('fatigue'),
            chills: getCheck('chills'),

            // Pattern
            fever_pattern: parseInt(getVal('fever_pattern')),

            // Exposure
            travel_to_hot_area: getCheck('travel'),
            mosquito_exposure: getCheck('mosquito'),
            sun_exposure: getCheck('sun_exposure'),
            hygiene_issue: getCheck('hygiene')
        };

        // Simulating processing delay
        await new Promise(r => setTimeout(r, 600));

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error(response.statusText);

        const result = await response.json();

        // Render Results

        // 1. Prediction
        document.getElementById('predictionValue').textContent = result.prediction.replace(/-/g, ' ');

        // 2. Severity Badge
        const badge = document.getElementById('severityBadge');
        badge.textContent = result.severity + " Severity";

        // Color coding
        if (result.severity === 'High') {
            badge.style.backgroundColor = '#ef4444'; // Red
            badge.style.color = 'white';
        } else if (result.severity === 'Medium') {
            badge.style.backgroundColor = '#f97316'; // Orange
            badge.style.color = 'white';
        } else {
            badge.style.backgroundColor = '#10b981'; // Green
            badge.style.color = 'white';
        }

        // 3. Advice
        document.getElementById('adviceText').textContent = result.advice;

        // 4. Probabilities
        const probList = document.getElementById('probList');
        probList.innerHTML = '';
        Object.entries(result.probabilities)
            .sort(([, a], [, b]) => b - a)
            .forEach(([key, val]) => {
                const div = document.createElement('div');
                div.className = 'prob-row';
                div.innerHTML = `<span>${key}</span><span>${(val * 100).toFixed(1)}%</span>`;
                probList.appendChild(div);
            });

        resultCard.classList.remove('hidden');
        resultCard.scrollIntoView({ behavior: 'smooth' });

    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});
