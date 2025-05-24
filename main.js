// Global variables
let selectedState = null;
let statesData = {};
let yieldHistoryChart = null;
let factorsChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initMap();
    setupEventListeners();
    document.getElementById('data-alert').style.display = 'block';
});

// Initialize India map with fixed view
function initMap() {
    const mapContainer = document.getElementById('india-map');
    const width = mapContainer.clientWidth;
    const height = 700;

    // Clear previous SVG
    d3.select('#india-map svg').remove();

    const svg = d3.select('#india-map')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    document.getElementById('map-loading').style.display = 'block';

    Promise.all([
        d3.json('/static/data/india_states.geojson'),
        fetch('/static/data/state_data.json').then(r => r.json())
    ]).then(([geoData, stateData]) => {
        statesData = stateData;
        document.getElementById('data-alert').style.display = 'none';
        document.getElementById('map-loading').style.display = 'none';
        renderMap(svg, geoData, width, height);
    }).catch(error => {
        console.error('Error loading data:', error);
        document.getElementById('map-loading').innerHTML = 
            '<div class="alert alert-danger">Failed to load map data. Please refresh.</div>';
    });
}

// Render map with proper fixed projection
function renderMap(svg, geoData, width, height) {
    // Optimized fixed projection for full India view
    const projection = d3.geoMercator()
        .center([83, 23])  // India's geographic center
        .scale(1200)
        .translate([width/2.2, height/1.8]);

    const path = d3.geoPath().projection(projection);

    // Validate state names
    geoData.features.forEach(f => {
        const normalized = normalizeStateName(f.properties.NAME_1);
        if (!statesData[normalized]) {
            console.warn('No data for state:', f.properties.NAME_1);
        }
    });

    // Create color scale
    const validYields = Object.values(statesData)
        .filter(d => !isNaN(d.avg_yield))
        .map(d => d.avg_yield);
    const colorScale = d3.scaleLinear()
        .domain([d3.min(validYields), d3.max(validYields)])
        .range(['#e8f5e9', '#2e7d32']);

    // Draw states
    svg.selectAll('.state')
        .data(geoData.features)
        .enter().append('path')
        .attr('class', d => {
            const stateName = normalizeStateName(d.properties.NAME_1);
            return statesData[stateName] ? 'state' : 'state state-no-data';
        })
        .attr('d', path)
        .style('fill', d => {
            const stateName = normalizeStateName(d.properties.NAME_1);
            return statesData[stateName] ? colorScale(statesData[stateName].avg_yield) : '#f5f5f5';
        })
        .on('mouseover', handleMouseOver)
        .on('mouseout', handleMouseOut)
        .on('click', handleStateClick);

    // Add state labels
    svg.selectAll('.state-label')
        .data(geoData.features)
        .enter().append('text')
        .attr('class', 'state-label')
        .attr('transform', d => `translate(${path.centroid(d)})`)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px')
        .style('fill', '#2e7d32')
        .text(d => d.properties.NAME_1)
        .style('display', d => {
            const stateName = normalizeStateName(d.properties.NAME_1);
            return statesData[stateName] ? 'block' : 'none';
        });
}

// Simplified state name normalization
// Change normalizeStateName function to:
function normalizeStateName(stateName) {
    return stateName
        .replace(/\s+/g, ' ')  // Handle multiple spaces
        .trim()
        .replace(/(?:^|\s)\S/g, a => a.toUpperCase());  // Proper capitalization
}

function handleMouseOver(event, d) {
    const stateName = d.properties.NAME_1;
    const panel = document.getElementById('info-panel');
    
    if (!statesData[stateName]) {
        panel.innerHTML = `<strong>${stateName}</strong><br>Data collection in progress`;
        panel.style.backgroundColor = '#ffebee';
        return;
    }
    
    const stateInfo = statesData[stateName];
    panel.innerHTML = `
        <strong>${stateName}</strong><br>
        Avg Yield: ${stateInfo.avg_yield} t/ha<br>
        Top Crop: ${stateInfo.top_crop}<br>
        Rainfall: ${stateInfo.avg_rainfall} mm
    `;
    panel.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';

    d3.select(event.currentTarget).style('fill', '#1b5e20');
}

function handleMouseOut(event, d) {
    document.getElementById('info-panel').innerHTML = 'Hover over a state to see details';
    panel.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
    
    if (selectedState !== d.properties.NAME_1) {
        const stateName = d.properties.NAME_1;
        d3.select(event.currentTarget).style('fill', 
            statesData[stateName] ? colorScale(statesData[stateName].avg_yield) : '#f0f0f0'
        );
    }
}

function handleStateClick(event, d) {
    const clickedState = d.properties.NAME_1;
    if (selectedState === clickedState) return;
    
    // Remove previous selection
    if (selectedState) {
        d3.selectAll(`.state[data-state="${selectedState}"]`)
            .classed('state-selected', false)
            .style('fill', colorScale(statesData[selectedState].avg_yield));
    }
    
    // Set new selection
    selectedState = clickedState;
    d3.select(event.currentTarget)
        .classed('state-selected', true)
        .style('fill', '#1b5e20');
    
    // Update UI
    document.getElementById('state-info').style.display = 'block';
    document.getElementById('prediction-form').style.display = 'none';
    document.getElementById('result-container').style.display = 'none';
    
    // Update state info
    fetchStateData(selectedState);
}

function setupEventListeners() {
    document.getElementById('show-prediction-form').addEventListener('click', () => {
        document.getElementById('prediction-form').style.display = 'block';
        document.getElementById('result-container').style.display = 'none';
    });
    
    document.getElementById('crop-form').addEventListener('submit', function(e) {
        e.preventDefault();
        submitPredictionForm();
    });
    
    document.getElementById('new-prediction-btn').addEventListener('click', () => {
        document.getElementById('prediction-form').style.display = 'block';
        document.getElementById('result-container').style.display = 'none';
    });
    
    document.getElementById('export-btn').addEventListener('click', exportToCSV);
}

function fetchStateData(stateName) {
    document.getElementById('loading').style.display = 'block';
    
    fetch(`/get_state_data/${encodeURIComponent(stateName)}`)
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            document.getElementById('loading').style.display = 'none';
            if (data.success) {
                updateStateInfo(data);
            } else {
                showError(data.message);
            }
        })
        .catch(error => {
            document.getElementById('loading').style.display = 'none';
            showError(error.message);
        });
}

function updateStateInfo(data) {
    document.getElementById('state-name').textContent = data.state;
    document.getElementById('avg-yield').textContent = data.avg_yield;
    document.getElementById('top-crop').textContent = data.top_crop;
    document.getElementById('avg-rainfall').textContent = data.avg_rainfall;
    document.getElementById('crop-count').textContent = data.crops.length;
    document.getElementById('form-state-name').textContent = data.state;
    document.getElementById('state').value = data.state;

    populateDropdown('crop', data.crops);
    populateDropdown('season', data.seasons);
    document.getElementById('rainfall').value = data.avg_rainfall;

    updateYieldHistoryChart(data.historic_data);
}

function populateDropdown(id, options) {
    const select = document.getElementById(id);
    select.innerHTML = `<option value="" selected disabled>Select ${id.charAt(0).toUpperCase() + id.slice(1)}</option>`;
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}

function updateYieldHistoryChart(historicData) {
    const ctx = document.getElementById('yield-history-chart').getContext('2d');
    
    if (yieldHistoryChart) {
        yieldHistoryChart.destroy();
    }
    
    const years = Object.keys(historicData).sort();
    const yields = years.map(year => historicData[year]);
    
    yieldHistoryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Historical Yield (tonnes/hectare)',
                data: yields,
                borderColor: '#2e7d32',
                backgroundColor: 'rgba(46, 125, 50, 0.1)',
                borderWidth: 2,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Yield (t/ha)'
                    }
                }
            }
        }
    });
}

function submitPredictionForm() {
    const formData = new FormData(document.getElementById('crop-form'));
    document.getElementById('loading').style.display = 'block';
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error('Prediction failed');
        return response.json();
    })
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        if (data.success) {
            showPredictionResult(data);
        } else {
            showError(data.message);
        }
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        showError(error.message);
    });
}

function showPredictionResult(data) {
    document.getElementById('prediction-form').style.display = 'none';
    document.getElementById('result-container').style.display = 'block';
    
    document.getElementById('prediction-result').innerHTML = `
        <strong>Predicted Yield:</strong> ${data.prediction} tonnes/hectare
    `;
    
    generateRecommendations(data.prediction, data.feature_importance);
    updateFactorsChart(data.feature_importance);
}


function updateFactorsChart(importanceData) {
    const ctx = document.getElementById('factors-chart').getContext('2d');
    
    // Completely destroy previous chart
    if (factorsChart) {
        factorsChart.destroy();
        factorsChart = null;
    }

    // Validate and format data
    const validData = Object.entries(importanceData)
        .filter(([key, value]) => typeof value === 'number' && value > 1) // Filter small values
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5); // Show top 5 factors

    // Create meaningful labels
    const labelMap = {
        'Annual_Rainfall': '🌧️ Rainfall',
        'Fertilizer': '🌱 Fertilizer',
        'Pesticide': '🐜 Pesticide',
        'Area': '🌾 Cultivation Area',
        'Crop': '🌻 Crop Type',
        'Season': '☀️ Season',
        'State': '📍 Region'
    };

    factorsChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: validData.map(([key]) => labelMap[key] || key),
            datasets: [{
                data: validData.map(([_, val]) => val),
                backgroundColor: [
                    '#2e7d32', // Dark green
                    '#4caf50', // Medium green
                    '#81c784', // Light green
                    '#a5d6a7', // Very light green
                    '#c8e6c9'  // Lightest green
                ],
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 20,
                        padding: 15,
                        font: {
                            size: 14
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const label = context.label || '';
                            const value = context.formattedValue || '';
                            return `${label}: ${value}%`;
                        }
                    }
                }
            },
            animation: {
                duration: 0 // Disable animations for stability
            }
        }
    });
}

function generateRecommendations(prediction, factors) {
    const recommendations = document.getElementById('recommendations');
    recommendations.innerHTML = '';
    
    // Add recommendation logic based on factors and prediction
    const topFactor = Object.entries(factors).reduce((a, b) => a[1] > b[1] ? a : b)[0];
    
    const recommendationItems = [
        `Focus on improving ${topFactor} management`,
        `Maintain optimal irrigation practices`,
        `Monitor soil health regularly`,
        `Consider crop rotation strategies`
    ];
    
    recommendationItems.forEach(item => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.textContent = item;
        recommendations.appendChild(li);
    });
}

function showHistoricalData() {
    if (!selectedState) {
        showError('Please select a state first');
        return;
    }
    
    document.getElementById('loading').style.display = 'block';
    
    fetch(`/get_state_data/${selectedState}`)
        .then(response => {
            if (!response.ok) throw new Error('Failed to fetch historical data');
            return response.json();
        })
        .then(data => {
            document.getElementById('loading').style.display = 'none';
            if (data.success) {
                displayHistoricalData(data.historic_data);
            } else {
                showError(data.message);
            }
        })
        .catch(error => {
            document.getElementById('loading').style.display = 'none';
            showError(error.message);
        });
}

function displayHistoricalData(data) {
    const years = Object.keys(data).sort();
    const yields = years.map(year => data[year]);
    
    const ctx = document.createElement('canvas');
    const modalContent = document.getElementById('history-content');
    
    // Clear previous content
    modalContent.innerHTML = '';
    modalContent.appendChild(ctx);
    
    // Destroy previous chart instance
    if (historyChart) {
        historyChart.destroy();
    }
    
    historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: years,
            datasets: [{
                label: 'Historical Yield (tonnes/hectare)',
                data: yields,
                borderColor: '#2e7d32',
                backgroundColor: 'rgba(46, 125, 50, 0.1)',
                borderWidth: 2,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Yield (t/ha)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Year'
                    }
                }
            }
        }
    });
    
    // Update modal title and show
    document.getElementById('history-state-name').textContent = selectedState;
    new bootstrap.Modal(document.getElementById('history-modal')).show();
}

// Replace exportToCSV function with:
function exportToCSV() {
    if (!factorsChart?.data?.labels) {
        showError('Complete a prediction first');
        return;
    }

    const prediction = document.getElementById('prediction-result').textContent;
    const factors = factorsChart.data.labels.map((label, index) => 
        `${label}: ${factorsChart.data.datasets[0].data[index]}%`
    );
    
    const recommendations = Array.from(document.querySelectorAll('#recommendations li'))
        .map(li => li.textContent);

    const csvContent = [
        ['State', 'Predicted Yield', 'Factors', 'Recommendations'],
        [
            selectedState || 'N/A',
            prediction.match(/[\d.]+/)?.[0] + ' tonnes/hectare' || 'N/A',
            `"${factors.join('; ')}"`,
            `"${recommendations.join('; ')}"`
        ]
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.href = url;
    link.download = `CropYield_${selectedState || 'prediction'}_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger position-fixed top-0 end-0 m-3';
    errorDiv.role = 'alert';
    errorDiv.innerHTML = `
        <strong>Error:</strong> ${message}
        <button type="button" class="btn-close float-end" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}
