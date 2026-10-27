const sessionId = new URLSearchParams(location.search).get('session');
const runs = new Map();
let lane = 'single', selectedPca = 0, playing = true;
function setupSession() {
  if (!sessionId || !/^[a-zA-Z0-9_-]{8,64}$/.test(sessionId)) {
    error('画面の選択ページから、同じセッションのチャット画面と可視化画面を開いてください。');
    status('セッション未設定'); return;
  }
  $('sessionLabel').textContent = `SESSION ${sessionId.slice(0, 8)}`;
  const feed = new EventSource(`/api/events/${encodeURIComponent(sessionId)}`);
  feed.onmessage = e => { try { handleBroadcast(JSON.parse(e.data)); } catch (err) { error(`表示エラー: ${err.message}`); } };
  feed.onerror = () => status('再接続しています…');
  fetch('/api/system').then(r => r.json()).then(s => { $('gpuName').textContent = s.device; }).catch(() => {});
}
function currentRun() { return runs.get(lane); }
function renderLane() {
  const run = currentRun();
  clearCharts(); $('pcaChart').replaceChildren(); $('pcaDetails').textContent = 'PCAを計算中…';
  if (!run) { status('入力待ち'); return; }
  startEvent(run.start);
  state.steps = run.steps;
  state.busy = !run.done;
  if (run.steps.length) showStep(run.steps.length - 1);
  if (run.pca) renderPca();
  renderCompare();
  if (run.done) status(run.done.reason === 'length' ? '上限まで生成しました' : '生成完了');
  $('laneA').classList.toggle('active', lane === 'A');
  $('laneB').classList.toggle('active', lane === 'B');
}
function updateLaneButtons() {
  for (const name of ['A', 'B']) {
    const run = runs.get(name);
    $(name === 'A' ? 'laneA' : 'laneB').textContent = run ? `${name} · T=${run.start.settings.temperature.toFixed(1)} · ${run.steps.length} tokens` : `${name} · 未実行`;
  }
  $('comparisonNote').textContent = runs.has('A') || runs.has('B') ? '同じ入力・履歴・seedで順次生成' : '通常生成';
  renderCompare();
}
function renderCompare() {
  const panel=$('comparePanel'), holder=$('compareResults');
  panel.hidden=!(runs.has('A')||runs.has('B'));
  if(panel.hidden)return;
  const step=Math.max(0,state.selected);
  const fragment=document.createDocumentFragment();
  for(const name of ['A','B']) {
    const run=runs.get(name), box=el('div','compare-run');
    box.append(el('strong','',run?`${name} · Temperature ${run.start.settings.temperature.toFixed(1)}`:`${name} · 待機中`));
    if(run?.steps.length) {
      const sample=run.steps[Math.min(step,run.steps.length-1)];
      box.append(el('p','compare-answer',run.steps.at(-1).text.slice(0,180)||'生成中…'));
      const barTitle=el('small','',`Step ${sample.step} · 次トークン候補`);box.append(barTitle);
      sample.candidates.slice(0,4).forEach(c=>{const row=el('div','compare-candidate');row.append(el('span','',visible(c.text)),el('span','',percent(c.prob)));const track=el('i');track.style.width=`${c.prob*100}%`;row.append(track);box.append(row);});
    } else box.append(el('p','compare-answer','生成を待っています…'));
    fragment.append(box);
  }
  holder.replaceChildren(fragment);
}
function handleBroadcast(event) {
  if (event.type === 'control') {
    playing = event.playing;
    $('pauseButton').textContent = playing ? '一時停止' : '再開';
    status(playing ? '生成しています' : '一時停止中', playing);
    return;
  }
  if (event.type === 'status') { status(event.message, true); return; }
  const key = event.run_label || 'single';
  if (event.type === 'start') {
    if (key === 'single') runs.clear();
    if (key === 'A') runs.delete('B');
    runs.set(key, {start:event, steps:[], pca:null, done:null});
    lane = key; selectedPca = 0; renderLane(); updateLaneButtons();
    $('pauseButton').disabled = false; $('stepButton').disabled = false;
    return;
  }
  const run = runs.get(key);
  if (!run) return;
  if (event.type === 'pca') { run.pca = event; if (lane === key) renderPca(); }
  if (event.type === 'token') {
    run.steps.push(event);
    if (lane === key) {
      state.steps = run.steps;
      if (state.follow) showStep(state.steps.length - 1);
      else { $('timeline').max = state.steps.length; $('timelineCount').textContent = `${state.selected+1} / ${state.steps.length}`; $('nextStep').disabled = false; }
    }
    updateLaneButtons();
  }
  if (event.type === 'done') { run.done = event; if (lane === key) status('生成完了'); $('pauseButton').disabled = true; $('stepButton').disabled = true; }
  if (event.type === 'error') { error(event.message); status('生成エラー'); $('pauseButton').disabled = true; $('stepButton').disabled = true; }
}
function renderPca() {
  const points = currentRun()?.pca?.points || [];
  const svg = $('pcaChart'); svg.replaceChildren();
  if (!points.length) { $('pcaDetails').textContent = '入力トークンの位置を特定できませんでした。'; return; }
  const maxX = Math.max(1e-5, ...points.map(p=>Math.abs(p.x)));
  const maxY = Math.max(1e-5, ...points.map(p=>Math.abs(p.y)));
  const make = (tag, attrs) => { const n=document.createElementNS('http://www.w3.org/2000/svg',tag); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,String(v));return n; };
  svg.append(make('line',{x1:300,y1:8,x2:300,y2:172,stroke:'#dce3ec'}),make('line',{x1:12,y1:90,x2:588,y2:90,stroke:'#dce3ec'}));
  points.forEach((p,i) => {
    const x=300+p.x/maxX*270, y=90-p.y/maxY*70;
    const dot=make('circle',{cx:x,cy:y,r:i===selectedPca?8:5,fill:i===selectedPca?'#ed8f47':'#335ff0',tabindex:0,role:'button','aria-label':`トークン ${i+1} ${visible(p.text)} のPCA位置`});
    dot.addEventListener('click',()=>pickPca(i));
    dot.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pickPca(i);}});
    const title=make('title',{});title.textContent=`${i+1}: ${p.text} / ID ${p.id}`;dot.append(title);svg.append(dot);
    if (i===selectedPca) { const label=make('text',{x:Math.min(550,x+10),y:Math.max(18,y-10),fill:'#a94e18','font-size':12});label.textContent=visible(p.text).slice(0,16);svg.append(label); }
  });
  const p=points[Math.min(selectedPca,points.length-1)];
  $('pcaDetails').textContent=`${p.index+1} / ${points.length} · ${visible(p.text)} · ID ${p.id}\nPC1 ${p.x.toFixed(3)} / PC2 ${p.y.toFixed(3)}`;
  $('pcaSelected').textContent=`選択 ${p.index+1} / ${points.length}`;
  [...$('inputTokens').children].forEach((child,i)=>child.classList.toggle('picked',i===selectedPca));
}
function pickPca(index) { selectedPca=index; renderPca(); }
$('inputTokens').addEventListener('click',e=>{const chip=e.target.closest('.token-chip');if(chip)pickPca([...$('inputTokens').children].indexOf(chip));});
for(const name of ['A','B']) $(name==='A'?'laneA':'laneB').addEventListener('click',()=>{if(runs.has(name)){lane=name;renderLane();}});
$('layerSelect').addEventListener('change',()=>{const s=state.steps[state.selected];if(s){renderAttention(s);renderHidden(s);}});
$('timeline').addEventListener('input',()=>{state.follow=false;showStep(Number($('timeline').value)-1);});
$('prevStep').addEventListener('click',()=>{state.follow=false;showStep(state.selected-1);});
$('nextStep').addEventListener('click',()=>{state.follow=false;showStep(state.selected+1);});
$('liveButton').addEventListener('click',()=>{state.follow=true;showStep(state.steps.length-1);});
async function controlAction(action) {
  try { const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId,action})});if(!r.ok)throw new Error((await r.json()).detail); }
  catch(e){error(e.message);}
}
$('pauseButton').addEventListener('click',()=>controlAction(playing?'pause':'resume'));
$('stepButton').addEventListener('click',()=>controlAction('step'));
setupSession();
