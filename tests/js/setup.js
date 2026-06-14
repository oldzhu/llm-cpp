// Vitest setup — initialize jsdom with required DOM elements, expose globals

import { JSDOM } from 'jsdom'
import fs from 'fs'
import path from 'path'

const htmlPath = path.resolve('web/templates/base.html')
let html = fs.readFileSync(htmlPath, 'utf-8')

const templatesDir = path.resolve('web/templates')
for (const name of ['config.html','monitor.html','code.html','chat.html','learn.html','play.html']) {
  const content = fs.readFileSync(path.join(templatesDir, name), 'utf-8')
  html = html.replace(`{% include '${name}' %}`, content)
}

const dom = new JSDOM(html, { url: 'http://localhost:8080', runScripts: 'dangerously' })

// Expose browser globals to Node.js
global.window = dom.window
global.document = dom.window.document
global.localStorage = {
  _data: {},
  getItem(key) { return this._data[key] || null },
  setItem(key, value) { this._data[key] = String(value) },
  removeItem(key) { delete this._data[key] },
  clear() { this._data = {} },
}

// Load JS files in order via eval and capture globals
const staticDir = path.resolve('web/static')
const scripts = ['i18n.js', 'settings.js', 'architecture.js', 'charts.js', 'app.js', 'code.js', 'chat.js', 'play.js']

for (const script of scripts) {
  const src = fs.readFileSync(path.join(staticDir, script), 'utf-8')
  try {
    dom.window.eval(src)
  } catch (e) {
    console.warn(`Script ${script} load warning: ${e.message.slice(0, 80)}`)
  }
}

// Expose globals from jsdom window to Node global for tests
const exposed = ['I18N', 't', 'currentLang', 'toggleLang', 'applyI18n', 'aiSettings', 'AISettings',
                  'ARCH_COMPONENTS', 'archDiagram', 'ui', 'codeUI', 'chatUI', 'playUI']
for (const name of exposed) {
  if (dom.window[name] !== undefined && dom.window[name] !== null) {
    global[name] = dom.window[name]
  }
}
