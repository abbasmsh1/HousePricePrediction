// Page Navigation
function showPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Remove active class from all nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected page
    document.getElementById(`${pageName}-page`).classList.add('active');
    
    // Activate corresponding nav button
    event.target.classList.add('active');
    
    // Load metrics if metrics page is shown
    if (pageName === 'metrics') {
        loadMetrics();
    }
}

// Form Submission
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading overlay
    document.getElementById('loading-overlay').classList.remove('hidden');
    
    // Collect form data
    const formData = {
        bedrooms: parseFloat(document.getElementById('bedrooms').value),
        bathrooms: parseFloat(document.getElementById('bathrooms').value),
        sqft_living: parseFloat(document.getElementById('sqft_living').value),
        sqft_lot: parseFloat(document.getElementById('sqft_lot').value),
        floors: parseFloat(document.getElementById('floors').value),
        waterfront: parseInt(document.getElementById('waterfront').value),
        view: parseInt(document.getElementById('view').value),
        condition: parseInt(document.getElementById('condition').value),
        sqft_above: parseFloat(document.getElementById('sqft_above').value),
        sqft_basement: parseFloat(document.getElementById('sqft_basement').value),
        yr_built: parseInt(document.getElementById('yr_built').value),
        yr_renovated: parseInt(document.getElementById('yr_renovated').value),
        city: document.getElementById('city').value
    };
    
    try {
        // Send prediction request
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
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        // Hide loading overlay
        document.getElementById('loading-overlay').classList.add('hidden');
    }
});

// Display Prediction Results
function displayResults(result) {
    // Show results section
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');
    
    // Display ensemble prediction
    const ensemblePrice = document.getElementById('ensemble-price');
    ensemblePrice.textContent = formatPrice(result.ensemble_prediction);
    
    // Display confidence
    const confidenceLevel = document.getElementById('confidence-level');
    confidenceLevel.textContent = result.confidence;
    confidenceLevel.style.color = getConfidenceColor(result.confidence);
    
    // Display individual model predictions
    const modelsGrid = document.getElementById('models-grid');
    modelsGrid.innerHTML = '';
    
    for (const [modelName, prediction] of Object.entries(result.individual_predictions)) {
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        modelCard.innerHTML = `
            <div class="model-name">${modelName}</div>
            <div class="model-prediction">${formatPrice(prediction)}</div>
        `;
        modelsGrid.appendChild(modelCard);
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Load Metrics
async function loadMetrics() {
    try {
        const response = await fetch('/metrics');
        
        if (!response.ok) {
            throw new Error('Failed to load metrics');
        }
        
        const data = await response.json();
        
        // Display model metrics
        displayModelMetrics(data.model_metrics);
        
        // Display error distributions
        displayErrorDistributions(data.error_distributions);
        
    } catch (error) {
        console.error('Error loading metrics:', error);
        alert('Failed to load metrics');
    }
}

// Display Model Metrics
function displayModelMetrics(metrics) {
    const metricsGrid = document.getElementById('metrics-grid');
    metricsGrid.innerHTML = '';
    
    for (const [modelName, modelMetrics] of Object.entries(metrics)) {
        const metricCard = document.createElement('div');
        metricCard.className = 'metric-card';
        metricCard.innerHTML = `
            <h3>${modelName}</h3>
            <div class="metric-item">
                <span class="metric-label">Test R² Score</span>
                <span class="metric-value">${modelMetrics.test_r2.toFixed(4)}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Test RMSE</span>
                <span class="metric-value">$${modelMetrics.test_rmse.toFixed(2)}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Train R² Score</span>
                <span class="metric-value">${modelMetrics.train_r2.toFixed(4)}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Train RMSE</span>
                <span class="metric-value">$${modelMetrics.train_rmse.toFixed(2)}</span>
            </div>
        `;
        metricsGrid.appendChild(metricCard);
    }
}

// Display Error Distributions
function displayErrorDistributions(errorDistributions) {
    const chartsGrid = document.getElementById('charts-grid');
    chartsGrid.innerHTML = '';
    
    const colors = {
        'Linear Regression': '#A8D8EA',
        'Decision Tree': '#FFE082',
        'Random Forest': '#7FB3D5',
        'XGBoost': '#FFD54F',
        'KNN': '#D4E6F1'
    };
    
    for (const [modelName, errors] of Object.entries(errorDistributions)) {
        const chartCard = document.createElement('div');
        chartCard.className = 'chart-card';
        
        const canvasId = `chart-${modelName.replace(/\s+/g, '-').toLowerCase()}`;
        chartCard.innerHTML = `
            <h4>${modelName} - Error Distribution</h4>
            <canvas id="${canvasId}"></canvas>
        `;
        chartsGrid.appendChild(chartCard);
        
        // Create histogram
        const ctx = document.getElementById(canvasId).getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: errors.map((_, i) => i + 1),
                datasets: [{
                    label: 'Prediction Error ($)',
                    data: errors,
                    backgroundColor: colors[modelName] || '#A8D8EA',
                    borderColor: colors[modelName] || '#7FB3D5',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Error: $' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Error ($)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Sample'
                        },
                        ticks: {
                            maxTicksLimit: 10
                        }
                    }
                }
            }
        });
    }
}

// Utility Functions
function formatPrice(price) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(price);
}

function getConfidenceColor(confidence) {
    switch(confidence) {
        case 'High':
            return '#4CAF50';
        case 'Medium':
            return '#FF9800';
        case 'Low':
            return '#F44336';
        default:
            return '#9E9E9E';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('House Price Predictor loaded successfully!');
});
