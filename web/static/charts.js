"use strict";

class LossChart {
  constructor() {
    this.ctx = document.getElementById("loss-chart")?.getContext("2d");
    if (!this.ctx) return;
    this.chart = new Chart(this.ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [{
          label: t("chart.loss"),
          data: [],
          borderColor: "#e94560",
          backgroundColor: "rgba(233,69,96,0.1)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.1,
        }],
      },
      options: {
        responsive: true,
        animation: false,
        scales: {
          x: { display: true, title: { display: true, text: t("monitor.step"), color: "#aaa" }, grid: { color: "#2a2a4a" }, ticks: { color: "#aaa" } },
          y: { display: true, title: { display: true, text: t("chart.loss"), color: "#aaa" }, grid: { color: "#2a2a4a" }, ticks: { color: "#aaa" } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  applyTranslations() {
    if (!this.chart) return;
    this.chart.options.scales.x.title.text = t("monitor.step");
    this.chart.options.scales.y.title.text = t("chart.loss");
    this.chart.data.datasets[0].label = t("chart.loss");
    this.chart.update("none");
  }

  update(step, loss) {
    if (!this.chart) return;
    this.chart.data.labels.push(step);
    this.chart.data.datasets[0].data.push(loss);
    if (this.chart.data.labels.length > 200) {
      this.chart.data.labels.shift();
      this.chart.data.datasets[0].data.shift();
    }
    this.chart.update("none");
  }

  reset() {
    if (!this.chart) return;
    this.chart.data.labels = [];
    this.chart.data.datasets[0].data = [];
    this.chart.update("none");
  }
}

window.chart = new LossChart();
