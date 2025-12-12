// --- THEME MANAGEMENT ---
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const target = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', target);
    localStorage.setItem('theme', target);
    updateChartColor(target);
}

// Load saved theme immediately
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);

// --- CHART CONFIGURATION ---
// We initialize Chart.js after the DOM is ready
let scoreChart;

document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('scoreChart').getContext('2d');
    
    scoreChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: ['#e2e8f0', '#f1f5f9'],
                borderWidth: 0,
                cutout: '88%',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { tooltip: { enabled: false } },
            animation: { animateScale: true, animateRotate: true }
        }
    });

    // Set initial chart colors based on theme
    updateChartColor(savedTheme);
});

function updateChartColor(theme) {
    if (!scoreChart) return;
    const emptyColor = theme === 'dark' ? '#334155' : '#e2e8f0';
    scoreChart.data.datasets[0].backgroundColor[1] = emptyColor;
    scoreChart.update();
}

// --- ANALYSIS LOGIC ---
async function analyzePrompt() {
    const input = document.getElementById('promptInput').value;
    const btn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('btnSpinner');
    const pill = document.getElementById('statusPill');

    // Basic Validation
    if (!input.trim()) {
        alert("Please enter a prompt first.");
        return;
    }

    // UI Loading State
    btn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'block';
    pill.textContent = "Processing...";
    pill.style.color = "var(--text-secondary)";

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: input })
        });

        const data = await response.json();

        if (data.error) {
            pill.textContent = "Error: " + data.error;
            pill.style.color = "var(--error-color)";
        } else if (data.status === "REJECTED") {
            updateResultUI(data.final_score, data.bert_score, 0, false);
            pill.textContent = "Status: Rejected (Low Quality)";
            pill.style.color = "var(--error-color)";
        } else {
            updateResultUI(data.final_score, data.bert_score, data.llm_score, true);
            pill.textContent = "Status: Accepted";
            pill.style.color = "var(--success-color)";
        }

    } catch (err) {
        console.error(err);
        pill.textContent = "System Error";
        pill.style.color = "var(--error-color)";
    } finally {
        btn.disabled = false;
        btnText.style.display = 'block';
        spinner.style.display = 'none';
    }
}

function updateResultUI(final, bert, llm, isSuccess) {
    // Determine Color
    let color = '#ef4444'; // Red default
    if (isSuccess) {
        if (final > 75) color = '#10b981'; // Green
        else if (final > 40) color = '#f59e0b'; // Orange
    }

    // Update Chart
    const theme = document.documentElement.getAttribute('data-theme');
    const emptyColor = theme === 'dark' ? '#334155' : '#e2e8f0';
    
    scoreChart.data.datasets[0].backgroundColor = [color, emptyColor];
    scoreChart.data.datasets[0].data = [final, 100 - final];
    scoreChart.update();

    // Update Text
    animateValue("finalScore", 0, Math.round(final), 800);
    document.getElementById('bertScore').textContent = bert.toFixed(1);
    document.getElementById('llmScore').textContent = llm ? llm.toFixed(1) : "N/A";
}

function animateValue(id, start, end, duration) {
    if (start === end) return;
    const range = end - start;
    const stepTime = Math.abs(Math.floor(duration / range));
    const obj = document.getElementById(id);
    let current = start;
    const timer = setInterval(function() {
        current += end > start ? 1 : -1;
        obj.textContent = current;
        if (current == end) clearInterval(timer);
    }, stepTime);
}