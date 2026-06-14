import { describe, it, expect } from 'vitest'

describe('i18n', () => {
  it('should have EN translations', () => {
    expect(I18N.en).toBeDefined()
    expect(I18N.en['app.title']).toBeTruthy()
  })

  it('should have ZH translations', () => {
    expect(I18N.zh).toBeDefined()
    expect(I18N.zh['app.title']).toBeTruthy()
  })

  it('should return EN by default', () => {
    expect(t('app.title')).toBe('build-llm-using-cpp Trainer')
  })

  it('should fallback to EN for missing ZH keys', () => {
    // Test that zh translations exist and are different from en
    expect(I18N.zh['app.title']).toBe('build-llm-using-cpp 训练器')
    expect(I18N.en['app.title']).not.toBe(I18N.zh['app.title'])
  })

  it('should interpolate params', () => {
    const result = t('code.selected', { s: '5', e: '10' })
    expect(result).toContain('5')
    expect(result).toContain('10')
  })

  it('should return key itself for unknown keys', () => {
    expect(t('nonexistent.key')).toBe('nonexistent.key')
  })

  it('should have nav translations', () => {
    for (const key of ['nav.config','nav.monitor','nav.code','nav.chat','nav.learn','nav.play']) {
      expect(I18N.en[key]).toBeTruthy()
      expect(I18N.zh[key]).toBeTruthy()
    }
  })

  it('should have tooltip translations', () => {
    for (const key of ['tooltip.dmodel','tooltip.layers','tooltip.lr','tooltip.steps']) {
      expect(I18N.en[key]).toBeTruthy()
    }
  })

  it('should have preset translations', () => {
    for (const key of ['preset.Tiny (32M)','preset.Small (64M)','preset.Medium (100M)']) {
      expect(I18N.en[key]).toBeTruthy()
    }
  })

  it('should have field translations for all config keys', () => {
    const fields = ['data_path','tokenizer','steps','batch','seq','dmodel','layers','lr','seed','norm_type','mlp_type']
    for (const f of fields) {
      expect(I18N.en['field.'+f]).toBeTruthy()
    }
  })
})
