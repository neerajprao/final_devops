document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const loadingEl = document.getElementById('loading');
    const resultsEl = document.getElementById('results');
    const predictionElements = {
        'tabnet': document.getElementById('tabnet-pred'),
        'xgb': document.getElementById('xgb-pred'),
        'vqc': document.getElementById('vqc-pred'),
        'qnn': document.getElementById('qnn-pred'),
        'consensus-value': document.getElementById('consensus-value'),
        'consensus-explanation': document.getElementById('consensus-explanation')
    };

    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Show loading, hide results
        loadingEl.classList.remove('hidden');
        resultsEl.classList.add('hidden');

        const formData = {
            creditScore: document.getElementById('creditScore').value,
            age: document.getElementById('age').value,
            tenure: document.getElementById('tenure').value,
            balance: document.getElementById('balance').value,
            numProducts: document.getElementById('numProducts').value,
            hasCard: document.getElementById('hasCard').value,
            isActive: document.getElementById('isActive').value,
            salary: document.getElementById('salary').value,
            geography: document.getElementById('geography').value,
            gender: document.getElementById('gender').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Unknown error occurred');
            }

            // Hide loading
            loadingEl.classList.add('hidden');

            // Update predictions
            updatePredictions(result.predictions);

            // Show results
            resultsEl.classList.remove('hidden');

            // Scroll to results
            resultsEl.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            loadingEl.classList.add('hidden');
            alert('Failed to get predictions: ' + error.message);
        }
    });

    function updatePredictions(predictions) {
        for (const model in predictionElements) {
            if (predictions.hasOwnProperty(model) && model !== 'consensus-value' && model !== 'consensus-explanation') {
                predictionElements[model].textContent = formatPrediction(predictions[model]);
            } else if (!predictions.hasOwnProperty(model) && model !== 'consensus-value' && model !== 'consensus-explanation') {
                predictionElements[model].textContent = "Not available";
            }
        }

        // Calculate consensus
        const consensusEl = predictionElements['consensus-value'];
        const consensusExplanationEl = predictionElements['consensus-explanation'];

        const modelPredictions = {};
        for (const key in predictions) {
            if (predictionElements.hasOwnProperty(key) && key !== 'consensus-value' && key !== 'consensus-explanation') {
                modelPredictions[key] = predictions[key];
            }
        }

        // Get valid predictions (numerical only) from the models
        const validPredictions = Object.values(modelPredictions).filter(p => typeof p === 'number');

        if (validPredictions.length > 0) {
            let stayCount = 0;
            let churnCount = 0;

            validPredictions.forEach(pred => {
                if (pred === 0) stayCount++;
                else if (pred === 1) churnCount++;
            });

            const total = stayCount + churnCount;
            const stayPercentage = total > 0 ? Math.round((stayCount / total) * 100) : 0;
            const churnPercentage = total > 0 ? Math.round((churnCount / total) * 100) : 0;

            if (stayCount > churnCount) {
                consensusEl.className = 'stay';
                consensusEl.textContent = `STAY (${stayPercentage}%)`;
                consensusExplanationEl.textContent = `${stayCount} out of ${total} models predict the customer will stay with the bank. ${churnCount} models predict the customer will leave.`;
            } else if (churnCount > stayCount) {
                consensusEl.className = 'churn';
                consensusEl.textContent = `CHURN (${churnPercentage}%)`;
                consensusExplanationEl.textContent = `${churnCount} out of ${total} models predict the customer will leave the bank. ${stayCount} models predict the customer will stay.`;
            } else if (total > 0) {
                consensusEl.className = '';
                consensusEl.textContent = `UNDECIDED (Stay: ${stayPercentage}%, Churn: ${churnPercentage}%)`;
                consensusExplanationEl.textContent = `The models are evenly split or have an equal number of stay and churn predictions: ${stayCount} predict stay and ${churnCount} predict churn.`;
            } else {
                consensusEl.className = '';
                consensusEl.textContent = 'No valid predictions for consensus';
                consensusExplanationEl.textContent = '';
            }
        } else {
            consensusEl.textContent = 'No valid predictions available';
            consensusExplanationEl.textContent = '';
        }
    }

    function formatPrediction(prediction) {
        if (typeof prediction === 'number') {
            return prediction === 0 ? 'Stay (0)' : 'Churn (1)';
        } else {
            return prediction;
        }
    }

    // Set default values for form fields
    function setDefaultValues() {
        document.getElementById('creditScore').value = 650;
        document.getElementById('age').value = 35;
        document.getElementById('tenure').value = 5;
        document.getElementById('balance').value = 75000;
        document.getElementById('numProducts').value = 2;
        document.getElementById('hasCard').value = 1;
        document.getElementById('isActive').value = 1;
        document.getElementById('salary').value = 65000;
        document.getElementById('geography').value = 'France';
        document.getElementById('gender').value = 'Male';
    }

    // Set default values on page load
    //setDefaultValues();
});

// For the models page
function toggleModel(element) {
    const content = element.nextElementSibling;
    content.classList.toggle('active');
    const arrow = element.querySelector('.arrow');
    arrow.style.transform = content.classList.contains('active') ? 'rotate(180deg)' : '';
}