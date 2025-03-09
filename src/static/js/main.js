// main.js

// References to HTML elements
const startDateInput = document.getElementById('startDate');
const endDateInput = document.getElementById('endDate');
const applyFilterBtn = document.getElementById('applyFilterBtn');
const refreshAlertsBtn = document.getElementById('refreshAlertsBtn');
const alertBanner = document.getElementById('alertBanner');
const alertsList = document.getElementById('alertsList');

// Chart contexts
const metricsChartCtx = document.getElementById('metricsChart').getContext('2d');
const predictionChartCtx = document.getElementById('predictionChart').getContext('2d');

// 1) Multi-Metric Line Chart (CPU, memory, GPU)
// Multi-Metric Line Chart (updated time scale configuration)
const metricsChart = new Chart(metricsChartCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'CPU (%)',
        data: [],
        borderColor: 'blue',
        backgroundColor: 'rgba(0, 0, 255, 0.1)',
        fill: true,
        yAxisID: 'y'
      },
      {
        label: 'Memory (%)',
        data: [],
        borderColor: 'green',
        backgroundColor: 'rgba(0, 255, 0, 0.1)',
        fill: true,
        yAxisID: 'y'
      },
      {
        label: 'GPU (%)',
        data: [],
        borderColor: 'orange',
        backgroundColor: 'rgba(255, 165, 0, 0.1)',
        fill: true,
        yAxisID: 'y'
      }
    ]
  },
  options: {
    responsive: true,
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day', // Change unit to 'day' (or 'month' if preferred)
          stepSize: 1, // One tick per day
          displayFormats: {
            day: 'MMM D' // e.g., "Mar 9"
          }
        }
      },
      y: {
        beginAtZero: true
      }
    }
  }
});


// 2) Doughnut Chart for Predicted CPU Usage
const predictionChart = new Chart(predictionChartCtx, {
  type: 'doughnut',
  data: {
    labels: ['Predicted CPU', 'Remaining'],
    datasets: [{
      label: 'CPU Prediction',
      data: [0, 100],
      backgroundColor: ['red', 'lightgray']
    }]
  },
  options: {
    responsive: true,
    cutout: '50%',
    plugins: {
      legend: { position: 'top' },
      title: {
        display: true,
        text: 'Real-Time CPU Prediction'
      }
    }
  }
});

/* -----------------------
   FETCH FUNCTIONS
------------------------*/

// Fetch metrics with optional date range
async function fetchMetrics(startDate, endDate) {
  try {
    let url = '/api/metrics';
    const params = [];
    if (startDate) params.push(`start=${startDate}`);
    if (endDate) params.push(`end=${endDate}`);
    if (params.length > 0) {
      url += '?' + params.join('&');
    }

    const response = await fetch(url);
    const data = await response.json();

    // data is an array of objects like:
    // [{ timestamp: "2025-03-07T03:19:33.434000", cpu: 20, memory: 50, gpu: "N/A" }, ...]

    // Reset chart data
    metricsChart.data.labels = [];
    metricsChart.data.datasets[0].data = []; // CPU
    metricsChart.data.datasets[1].data = []; // Memory
    metricsChart.data.datasets[2].data = []; // GPU

    data.forEach(item => {
      metricsChart.data.labels.push(item.timestamp);
      metricsChart.data.datasets[0].data.push(item.cpu);
      metricsChart.data.datasets[1].data.push(item.memory);
      // Convert "N/A" to 0 or parse if numeric
      const gpuVal = (item.gpu === 'N/A') ? 0 : parseFloat(item.gpu);
      metricsChart.data.datasets[2].data.push(gpuVal);
    });

    metricsChart.update();
  } catch (err) {
    console.error('Error fetching metrics:', err);
  }
}

// Fetch real-time prediction
async function fetchPrediction() {
  try {
    const response = await fetch('/api/prediction');
    const data = await response.json();

    const predictedCpu = data.predicted_cpu || 0;
    const clamped = Math.max(0, Math.min(predictedCpu, 100));

    // Update doughnut chart
    predictionChart.data.datasets[0].data = [clamped, 100 - clamped];
    predictionChart.update();

    // Check for alert
    if (data.alert_triggered) {
      alertBanner.style.display = 'block';
      alertBanner.textContent = data.alert_message;
    } else {
      alertBanner.style.display = 'none';
      alertBanner.textContent = '';
    }

  } catch (err) {
    console.error('Error fetching prediction:', err);
  }
}

// Fetch recent alerts
async function fetchAlerts() {
  try {
    const response = await fetch('/api/alerts');
    const alertsData = await response.json();

    // Clear existing list
    alertsList.innerHTML = '';
    alertsData.forEach(alert => {
      const li = document.createElement('li');
      li.textContent = `${alert.timestamp} - ${alert.message} (CPU: ${alert.predicted_cpu.toFixed(2)}%)`;
      alertsList.appendChild(li);
    });
  } catch (err) {
    console.error('Error fetching alerts:', err);
  }
}

/* -----------------------
   EVENT LISTENERS
------------------------*/

// Apply filter button
applyFilterBtn.addEventListener('click', () => {
  const startVal = startDateInput.value; // e.g. "2025-03-07"
  const endVal = endDateInput.value;     // e.g. "2025-03-09"
  // ISO-ify them if needed, or pass them directly if your server expects YYYY-MM-DD
  fetchMetrics(startVal, endVal);
});

// Refresh alerts button
refreshAlertsBtn.addEventListener('click', () => {
  fetchAlerts();
});

/* -----------------------
   INTERVAL UPDATES
------------------------*/
setInterval(() => {
  fetchPrediction();
}, 5000); // Update prediction every 5 seconds

// On page load, fetch initial data
fetchMetrics();
fetchPrediction();
fetchAlerts();
