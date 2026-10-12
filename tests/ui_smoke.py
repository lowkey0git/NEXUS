"""Browser smoke test: drives the real UI (Chromium via Playwright) against the real server with a scripted
model. Run:  python tests/ui_smoke.py [screenshot_dir]   (needs `pip install playwright` + a Chromium install)."""
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import uvicorn
from playwright.sync_api import expect, sync_playwright

from app.config import Settings
from app.main import create_app
from tests.helpers import ScriptedProvider, text, use

expect.set_options(timeout=15000)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
OUT.mkdir(parents=True, exist_ok=True)


def find_file_id(messages):
    return re.search(r"id=([0-9a-f]{32})", str(messages)).group(1)


sheet = {"name": "Expenses", "columns": ["category", "total"], "rows": [["food", 4200], ["transport", 1500]]}
script = [
    text("**Before you go to the bank**\n\n- [ ] Valid ID\n- [ ] Account number\n- [ ] Reference for the failed transfer\n\n| Task | Online or in branch |\n|---|---|\n| Deposit | In branch |\n| Failed transfer | Call first |\n\nConfirm hours at `your bank's website`. <script>alert(1)</script>"),
    [use("read_document", file="expenses.csv")],
    [use("create_spreadsheet", name="expense-totals", sheets=[sheet])],
    text("Created **expense-totals.xlsx** with totals per category."),
    lambda m: [use("delete_file", file_id=find_file_id(m))],
    text("Deleted the CSV as you approved."),
    text("Streaming check: this sentence is deliberately long so that it arrives in many small pieces while the page is open and being watched."),
]
settings = Settings(data_dir=Path(tempfile.mkdtemp()), approval_timeout=30)
prov = ScriptedProvider(script, chunk=3, delay=0.001)
app = create_app(settings, prov)
server = uvicorn.Server(uvicorn.Config(app, port=8124, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
time.sleep(1.5)

csv = Path(tempfile.mkdtemp()) / "expenses.csv"
csv.write_text("category,amount\nfood,2000\nfood,2200\ntransport,1500\n")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1100, "height": 780})
    pg.set_default_timeout(15000)
    dialogs = []
    pg.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    pg.goto("http://127.0.0.1:8124/")
    pg.click("#authSwitch")
    pg.fill("#email", "ui@example.com"); pg.fill("#password", "password123"); pg.click("#authBtn")
    expect(pg.locator(".empty h2")).to_be_visible()
    pg.screenshot(path=str(OUT / "1-empty.png"))

    pg.check("input[value=advise]", force=True)
    pg.fill("#input", "I need to go to the bank tomorrow. What should I prepare?"); pg.click("#send")
    expect(pg.locator(".a table")).to_be_visible()
    assert pg.locator(".a input[type=checkbox]").count() == 3
    assert pg.locator(".a script").count() == 0 and not dialogs, "script tag must be inert"
    pg.screenshot(path=str(OUT / "2-advice.png"))

    pg.click("#newConv")
    pg.check("input[value=execute]", force=True)
    pg.set_input_files("#fileInput", str(csv))
    expect(pg.locator(".chip")).to_contain_text("expenses.csv")
    pg.fill("#input", "Total my expenses by category into a spreadsheet"); pg.click("#send")
    expect(pg.locator(".a").last).to_contain_text("expense-totals.xlsx")
    expect(pg.locator("details.log summary")).to_contain_text("2 steps")
    pg.click("details.log summary")
    pg.screenshot(path=str(OUT / "3-automation.png"))
    pg.click("#openFiles")
    expect(pg.locator("#fileList")).to_contain_text("expense-totals.xlsx")
    with pg.expect_download() as dl:
        pg.locator("[data-dl]").first.click()
    print("downloaded:", dl.value.suggested_filename)
    pg.click("#closeFiles")

    pg.fill("#input", "Delete expenses.csv please"); pg.click("#send")
    expect(pg.locator(".approval")).to_be_visible()
    expect(pg.locator(".approval")).to_contain_text("Permanently delete")
    pg.screenshot(path=str(OUT / "4-approval.png"))
    pg.click(".approval button[data-v='1']")
    expect(pg.locator(".a").last).to_contain_text("Deleted the CSV")
    pg.click("#openFiles")
    expect(pg.locator("#fileList")).not_to_contain_text("expenses.csv")  # retries until the list reloads
    expect(pg.locator("#fileList")).to_contain_text("expense-totals.xlsx")
    pg.click("#closeFiles")

    # live streaming: partial text is visible in #live before the final stored message replaces it
    full = "Streaming check: this sentence is deliberately long so that it arrives in many small pieces while the page is open and being watched."
    prov.delay = 0.04  # slow the scripted model down so progressive rendering is observable
    pg.fill("#input", "stream please"); pg.click("#send")
    pg.wait_for_selector("#live", state="attached")
    partials = set()
    for _ in range(40):
        if pg.locator("#live").count():
            partials.add(len(pg.inner_text("#live")))
        time.sleep(0.05)
    assert len(partials) >= 3 and min(partials) < len(full), f"text did not grow progressively: {sorted(partials)}"
    expect(pg.locator(".a").last).to_contain_text("being watched.")
    expect(pg.locator("#live")).to_have_count(0)
    assert pg.locator(".a", has_text="Streaming check").count() == 1, "live text must not be duplicated by the stored message"
    pg.screenshot(path=str(OUT / "5a-streamed.png"))

    pg.fill("#input", "one more thing"); pg.click("#send")
    expect(pg.locator(".banner.bad")).to_contain_text("script exhausted")
    expect(pg.locator("#retry")).to_be_visible()
    pg.screenshot(path=str(OUT / "5-failure.png"))

    m = b.new_page(viewport={"width": 390, "height": 760})
    m.set_default_timeout(15000)
    m.goto("http://127.0.0.1:8124/")
    m.fill("#email", "ui@example.com"); m.fill("#password", "password123"); m.click("#authBtn")
    expect(m.locator("#menu")).to_be_visible()
    m.screenshot(path=str(OUT / "6-mobile.png"))
    b.close()
print("UI SMOKE TEST PASSED; screenshots in", OUT)
