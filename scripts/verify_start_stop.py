"""Regression: Stop during a delayed start must not allow overlapping sessions."""
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])
    page=browser.new_page(permissions=['camera'])
    pending=[]
    page.route('**/api/start',lambda route:pending.append(route))
    page.goto('http://127.0.0.1:8876')
    page.get_by_text('모델 준비 완료',exact=True).wait_for()
    page.get_by_role('button',name='카메라 시작').click()
    for _ in range(100):
        if pending: break
        page.wait_for_timeout(20)
    assert len(pending)==1,'Expected an in-flight start request'
    page.get_by_role('button',name='중지',exact=True).click()
    assert page.get_by_role('button',name='카메라 시작').is_disabled(), 'New start enabled while old start remains in flight'
    pending[0].continue_()
    expect(page.get_by_role('button',name='카메라 시작')).to_be_enabled()
    assert page.locator('#video').evaluate('(v)=>v.srcObject===null')
    page.unroute('**/api/start')
    page.get_by_role('button',name='카메라 시작').click()
    expect(page.locator('#risk-value')).not_to_have_text('—')
    page.get_by_role('button',name='중지',exact=True).click()
    browser.close()
    print('PASS: delayed start -> stop -> safe restart; physical camera unused')
