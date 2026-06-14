import { describe, it, expect } from 'vitest'

describe('Settings', () => {
  it('should have defaults', () => {
    expect(aiSettings).toBeDefined()
    expect(aiSettings.data.api_url).toBe('https://api.deepseek.com/anthropic')
    expect(aiSettings.data.model).toBe('deepseek-v4-flash')
  })

  it('should save and load from localStorage', () => {
    aiSettings.save({ api_key: 'test-key-123' })
    const loaded = aiSettings.load()
    expect(loaded.api_key).toBe('test-key-123')
  })

  it('should merge with defaults on partial save', () => {
    aiSettings.save({ model: 'custom-model' })
    const loaded = aiSettings.load()
    expect(loaded.model).toBe('custom-model')
    expect(loaded.api_url).toBe('https://api.deepseek.com/anthropic')
  })

  it('should escape HTML in modal', () => {
    const escaped = aiSettings.escape('<script>alert("xss")</script>')
    expect(escaped).not.toContain('<script>')
  })
})
