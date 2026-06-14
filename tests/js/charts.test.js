import { describe, it, expect } from 'vitest'

describe('Charts', () => {
  it('should create chart instance', () => {
    expect(window.chart).toBeDefined()
  })

  it('should update chart data', () => {
    const chart = window.chart
    if (chart && chart.chart) {
      chart.update(1, 5.5)
      expect(chart.chart.data.labels.length).toBeGreaterThanOrEqual(1)
      expect(chart.chart.data.datasets[0].data.length).toBeGreaterThanOrEqual(1)
    }
  })

  it('should reset chart data', () => {
    const chart = window.chart
    if (chart && chart.chart) {
      chart.update(2, 4.5)
      chart.reset()
      expect(chart.chart.data.labels.length).toBe(0)
    }
  })

  it('should not exceed 200 data points', () => {
    const chart = window.chart
    if (chart && chart.chart) {
      for (let i = 0; i < 250; i++) chart.update(i, Math.random())
      expect(chart.chart.data.labels.length).toBeLessThanOrEqual(200)
    }
  })
})
