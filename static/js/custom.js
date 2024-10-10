// static/js/custom.js

// Interactive Plotly features for D3.js and Plotly
document.addEventListener('DOMContentLoaded', function() {
    var fairnessData = {
        x: ['Population Equality', 'Minority Representation', 'Political Fairness'],
        y: [0.9, 0.85, 0.75],
        type: 'bar',
        marker: {
            color: ['#4a90e2', '#ff6f61', '#6b4e71']
        }
    };
    Plotly.newPlot('fairnessMetricsChart', [fairnessData]);

    // D3.js Dynamic Bar Chart Example
    var svg = d3.select('#populationDistributionChart')
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%');

    svg.selectAll('rect')
        .data([50000, 48000, 51000, 49500, 50500])
        .enter()
        .append('rect')
        .attr('width', function(d) { return d / 1000; })
        .attr('height', 30)
        .attr('y', function(d, i) { return i * 40; })
        .attr('fill', '#2563eb');
});
