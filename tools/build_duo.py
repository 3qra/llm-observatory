"""Derive the two-screen markup and shared visualization code from the original UI."""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "static"
html = (root / "index.html").read_text(encoding="utf-8")
js = (root / "app.js").read_text(encoding="utf-8")
chat = html.split('  <section class="chat-panel panel"', 1)[1].split('\n  </section>', 1)[0]
chat = '  <section class="chat-panel panel"' + chat + '\n  </section>'
dashboard = html.split('  <section class="dashboard"', 1)[1].split('\n  </section>\n</main>', 1)[0]
dashboard = '  <section class="dashboard"' + dashboard + '\n  </section>'
settings = dashboard.split('        <section class="panel settings-panel"', 1)[1].split('</section>', 1)[0]
settings = '        <section class="panel settings-panel"' + settings + '</section>'
dashboard = dashboard.replace(settings, '')
chat = chat.replace('    <div id="errorBox"', '''    <div class="duo-controls">
      <div class="d-flex gap-2 align-items-center flex-wrap"><button class="btn btn-outline-primary" id="pauseButton" type="button" disabled>一時停止</button><button class="btn btn-outline-primary" id="stepButton" type="button" disabled>1トークン進める</button><span id="runStatus" class="run-status" role="status">入力待ち</span></div>
      <div class="compare-line"><label for="compareTemperature">比較用 Temperature B</label><input id="compareTemperature" class="form-control form-control-sm" type="number" min="0" max="2" step="0.1" value="1.3"><button id="compareButton" class="btn btn-outline-primary" type="button">同じ質問でA/B比較</button></div>
    </div>
    <details class="chat-settings"><summary>生成設定を開く</summary>''' + settings + '</details>\n    <div id="errorBox"')
chat = chat.replace('<button class="btn btn-primary" type="submit" id="sendButton">', '<button class="btn btn-primary" type="submit" id="sendButton">')
dashboard = dashboard.replace('<div class="dashboard-heading">', '<div class="dashboard-heading">', 1)
dashboard = dashboard.replace('<span id="runStatus" class="run-status" role="status">入力待ち</span>', '<span id="runStatus" class="run-status" role="status">接続待ち</span>')
dashboard = dashboard.replace('    <div class="top-grid">', '''    <div class="viz-controls"><button id="pauseButton" class="btn btn-sm btn-outline-primary" type="button" disabled>一時停止</button><button id="stepButton" class="btn btn-sm btn-outline-primary" type="button" disabled>1トークン進める</button></div>
    <div class="compare-head"><button id="laneA" class="btn btn-sm btn-outline-primary" type="button">A · Temperature —</button><button id="laneB" class="btn btn-sm btn-outline-primary" type="button">B · Temperature —</button><span id="comparisonNote">通常生成</span></div>
    <div class="top-grid">''', 1)
dashboard = dashboard.replace('    <div class="middle-grid">', '''    <section class="panel pca-panel"><div class="panel-heading"><h3><span class="step-number">P</span>入力トークンのPCA</h3><span id="pcaSelected" class="meta">トークンをクリックして観察</span></div><div class="pca-layout"><svg id="pcaChart" role="img" aria-label="入力トークンの最終層ベクトルを2次元に射影した散布図" viewBox="0 0 600 180"></svg><div id="pcaDetails" class="pca-details">生成を始めると実データを表示します。</div></div><div class="panel-foot">最終層の入力位置ベクトルを主成分分析で2次元に射影。距離はこの質問内での相対値です。</div></section>
    <section id="comparePanel" class="panel compare-panel" hidden><div class="panel-heading"><h3>同じ質問の Temperature 比較</h3><span class="meta">同じ入力・履歴・seed</span></div><div id="compareResults" class="compare-results"></div></section>
    <div class="middle-grid">''', 1)
dashboard = dashboard.replace('      <div class="side-stack">', '      <div class="side-stack">', 1)
dashboard = dashboard.replace('    <footer class="dashboard-footer">', '    <div id="errorBox" class="alert alert-danger" role="alert" hidden></div>\n    <footer class="dashboard-footer">', 1)
head = '''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM Observatory</title><link rel="stylesheet" href="/static/vendor/bootstrap.min.css"><link rel="stylesheet" href="/static/style.css"><link rel="stylesheet" href="/static/duo.css"></head>'''
header = '''<header class="topbar"><div class="brand"><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><div><h1>LLM <strong>Observatory</strong></h1><span class="brand-caption">生成の過程を、2画面で観察する。</span></div></div><div class="top-actions"><span id="sessionLabel" class="session-label"></span><a class="btn btn-sm btn-outline-secondary" href="/">画面の選択</a></div></header>'''
(root / "chat.html").write_text(head + '<body class="duo-page chat-view">' + header + '<main class="duo-main">' + chat + '</main><script src="/static/chat.js" defer></script></body></html>', encoding="utf-8")
(root / "viz.html").write_text(head + '<body class="duo-page viz-view">' + header + '<main class="duo-main">' + dashboard + '</main><script src="/static/viz.js" defer></script></body></html>', encoding="utf-8")
shared = js.split('function handleEvent(event) {', 1)[0]
(root / "viz.js").write_text(shared + '\n' + (root / "viz-tail.js").read_text(encoding="utf-8"), encoding="utf-8")

