"""E2E tests: Web UI pages load, tabs switch, key elements visible."""

import pytest


class TestPageLoads:
    def test_home_page_loads(self, page):
        assert page.title() != ""
        assert "build-llm-using-cpp" in page.content()

    def test_status_bar_visible(self, page):
        assert page.locator("#status-bar").is_visible()
        assert page.locator("#status-indicator").is_visible()


class TestTabSwitching:
    def test_config_tab_active_by_default(self, page):
        assert page.locator("#tab-config").is_visible()

    def test_monitor_tab_switches(self, page):
        page.locator('[data-tab="monitor"]').click()
        page.wait_for_timeout(300)
        assert page.locator("#tab-monitor").is_visible()

    def test_code_tab_switches(self, page):
        page.locator('[data-tab="code"]').click()
        page.wait_for_timeout(500)
        assert page.locator("#tab-code").is_visible()
        # File tree should load
        page.wait_for_timeout(500)
        assert page.locator("#file-tree").is_visible()

    def test_chat_tab_switches(self, page):
        page.locator('[data-tab="chat"]').click()
        page.wait_for_timeout(300)
        assert page.locator("#tab-chat").is_visible()

    def test_learn_tab_switches(self, page):
        page.locator('[data-tab="learn"]').click()
        page.wait_for_timeout(300)
        assert page.locator("#tab-learn").is_visible()

    def test_play_tab_switches(self, page):
        page.locator('[data-tab="play"]').click()
        page.wait_for_timeout(300)
        assert page.locator("#tab-play").is_visible()


class TestConfigForm:
    def test_start_button_visible(self, page):
        assert page.locator("#btn-start").is_visible()

    def test_preset_buttons_visible(self, page):
        assert page.locator(".preset-btn").first.is_visible()

    def test_preset_click_updates_fields(self, page):
        preset = page.locator('.preset-btn[data-preset="Tiny (32M)"]')
        preset.click()
        page.wait_for_timeout(300)
        dmodel = page.locator('[name="dmodel"]')
        assert dmodel.input_value() == "32"


class TestArchitectureDiagram:
    def test_diagram_container_visible(self, page):
        page.locator('[data-tab="code"]').click()
        page.wait_for_timeout(500)
        assert page.locator("#arch-diagram-container").is_visible()

    def test_diagram_has_svg(self, page):
        page.locator('[data-tab="code"]').click()
        page.wait_for_timeout(500)
        svg = page.locator(".arch-svg")
        assert svg.count() > 0

    def test_diagram_component_clickable(self, page):
        page.locator('[data-tab="code"]').click()
        page.wait_for_timeout(800)
        components = page.locator(".arch-component")
        assert components.count() > 0
        # Click first component — should trigger file load
        components.first.click()
        page.wait_for_timeout(500)
        # Code filename should update
        filename = page.locator("#code-filename")
        assert filename.text_content() != "Select a file to view"


class TestI18N:
    def test_language_toggle_exists(self, page):
        assert page.locator("#lang-toggle").is_visible()

    def test_toggle_changes_labels(self, page):
        toggle = page.locator("#lang-toggle")
        original_text = toggle.text_content()
        toggle.click()
        page.wait_for_timeout(500)
        new_text = toggle.text_content()
        assert new_text != original_text  # "中文" ↔ "English"
