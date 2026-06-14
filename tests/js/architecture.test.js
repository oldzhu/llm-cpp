import { describe, it, expect } from 'vitest'

describe('Architecture Diagram', () => {
  it('should have all components defined', () => {
    const comps = ['tokenizer','embedding','layernorm','attention','mlp','residual','lm_head','loss']
    for (const c of comps) {
      expect(ARCH_COMPONENTS[c]).toBeDefined()
    }
  })

  it('each component should have EN and ZH labels', () => {
    for (const [key, comp] of Object.entries(ARCH_COMPONENTS)) {
      expect(comp.label_en).toBeTruthy()
      expect(comp.label_zh).toBeTruthy()
    }
  })

  it('each component should have description', () => {
    for (const [key, comp] of Object.entries(ARCH_COMPONENTS)) {
      expect(comp.desc_en).toBeTruthy()
      expect(comp.desc_zh).toBeTruthy()
    }
  })

  it('attention component should have formula', () => {
    expect(ARCH_COMPONENTS.attention.formula).toBeTruthy()
    expect(ARCH_COMPONENTS.attention.formula).toContain('Q')
    expect(ARCH_COMPONENTS.attention.formula).toContain('K')
  })

  it('should render SVG into container', () => {
    const container = document.getElementById('arch-diagram-container')
    if (container) {
      archDiagram.render(container)
      const svg = container.querySelector('svg')
      expect(svg).not.toBeNull()
    }
  })
})
