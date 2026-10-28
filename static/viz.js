'use strict';
const embedded = new URLSearchParams(location.search).get('embedded')==='1';
document.body.classList.add(embedded?'embedded-viz':'detail-viz');
if(embedded)document.body.classList.add('overview-viz');
document.querySelector('.dashboard-heading h2').textContent=embedded?'入力から次のトークンまで':'モデル内部の動き';
function fitViewport(){
  const width=embedded?window.parent.innerWidth:window.innerWidth;
  document.body.classList.toggle('fit-viz',width>=1241&&window.innerHeight>=(embedded?760:820));
}
window.addEventListener('resize',fitViewport);
if(embedded)window.parent.addEventListener('resize',fitViewport);
fitViewport();
const $ = id => document.getElementById(id);
const state = {models: [], messages: [], steps: [], selected: -1, follow: true, busy: false, controller: null, layers: 0, assistant: null};
const el = (tag, cls, text) => { const n = document.createElement(tag); if(cls) n.className = cls; if(text !== undefined) n.textContent = text; return n; };
const visible = text => String(text).replace(/\n/g, '↵').replace(/\t/g, '⇥').replace(/^ +$/, m => '␠'.repeat(m.length));
const percent = p => `${(p * 100).toFixed(1)}%`;
const color = value => {
  const v = Math.min(1, Math.max(0, value));
  const stops = [[237,242,255],[81,125,238],[237,143,71]];
  const [a,b,t] = v < .65 ? [stops[0],stops[1],v/.65] : [stops[1],stops[2],(v-.65)/.35];
  return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*t)).join(',')})`;
};
function error(message) { $('errorBox').textContent = message; $('errorBox').hidden = !message; }
function status(text, active=false) { $('runStatus').textContent=text; $('runStatus').classList.toggle('active',active); }
function busy(value) {
  state.busy = value;
  $('sendButton').hidden=value; $('stopButton').hidden=!value;
  $('modelSelect').disabled=value; $('resetButton').disabled=value; $('settingsFields').disabled=value;
  $('prompt').disabled=value;
  document.querySelectorAll('.example-button').forEach(b=>b.disabled=value);
}
function message(role, content) {
  $('welcome')?.remove();
  const box=el('article',`message ${role}`), label=el('div','message-label');
  label.append(el('span','avatar',role==='user'?'U':'Ai'),el('span','',role==='user'?'YOU':$('modelSelect').selectedOptions[0]?.textContent || 'MODEL'));
  box.append(label,el('div','bubble',content)); $('chatMessages').append(box);
  $('chatMessages').scrollTop=$('chatMessages').scrollHeight;
  return box;
}
function clearCharts() {
  state.steps=[]; state.selected=-1; state.follow=true;
  for(const id of ['hiddenChart','attentionChart','probabilityChart','logitsTable']) {
    $(id).replaceChildren(el('div','chart-empty','生成データを待っています…'));
  }
  $('inputTokens').replaceChildren(el('span','empty-inline','入力を処理しています…'));
  $('inputCount').textContent='— tokens'; $('currentToken').textContent='—';
  $('tokenId').textContent='TOKEN ID —'; $('selectedProb').textContent='確率 —';
  $('stepCount').textContent='STEP —'; $('speed').textContent='—';
  $('timeline').disabled=true; $('timeline').max=1; $('timeline').value=1;
  $('timelineCount').textContent='0 / 0'; $('prevStep').disabled=true; $('nextStep').disabled=true; $('liveButton').disabled=true;
}
function startEvent(data) {
  const fragment=document.createDocumentFragment();
  data.tokens.forEach(t=>{ const n=el('span','token-chip'); n.title=`${t.text}\nToken ID: ${t.id}`; n.append(el('b','',visible(t.text)),el('small','',String(t.id))); fragment.append(n); });
  $('inputTokens').replaceChildren(fragment); $('inputCount').textContent=`${data.tokens.length} tokens`;
  $('gpuName').textContent=data.device; $('precision').textContent=data.dtype.toUpperCase();
  $('layerCount').textContent=`${data.layers} layers · ${data.hidden_size} dims`;
  $('contextCount').textContent=`${data.prompt_tokens} tokens`;
  state.layers=data.layers;
  $('layerSelect').replaceChildren(...Array.from({length:data.layers},(_,i)=>{const o=el('option','',`Layer ${i+1}`);o.value=i;return o;}));
  $('layerSelect').value=data.layers-1; $('layerSelect').disabled=false;
  if(data.dropped_messages) error(`文脈長に収まるよう、古い${data.dropped_messages}件のメッセージを今回の推論から除きました。`);
  status('生成しています',true);
}
function renderHidden(s) {
  const fragment=document.createDocumentFragment();
  const sampled=new Set([0,s.hidden.length-1,Number($('layerSelect').value)+1]);
  for(let i=1;i<s.hidden.length;i+=Math.ceil((s.hidden.length-1)/8))sampled.add(i);
  s.hidden.forEach((row,index)=>{
    if(!sampled.has(index))return;
    const n=el('div',`hidden-row${index===Number($('layerSelect').value)+1?' active':''}`);
    n.title=`${index===0?'Embedding':`Layer ${index}`}${index===s.hidden.length-1?' (final norm)':''} / RMS最大値 ${s.hidden_scale[index].toFixed(4)}`;
    n.append(el('label','',index===0?'Embed':`L ${String(index).padStart(2,'0')}`));
    const cells=el('div','cells');
    row.forEach((v,j)=>{const c=el('span');c.style.backgroundColor=color(v);c.title=`区間 ${j+1} / RMS ${(v*s.hidden_scale[index]).toFixed(4)}`;cells.append(c);});
    n.append(cells,el('small','',s.hidden_scale[index].toFixed(s.hidden_scale[index]<.1?2:1)));
    if(index>0) {n.tabIndex=0;n.setAttribute('role','button');n.setAttribute('aria-label',`Layer ${index}のアテンションを表示`);const pick=()=>{$('layerSelect').value=index-1;renderAttention(s);renderHidden(s);};n.addEventListener('click',pick);n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick();}});}
    fragment.append(n);
  });
  const scroll=$('hiddenChart').scrollTop;
  $('hiddenChart').replaceChildren(fragment); $('hiddenChart').scrollTop=scroll;
}
function renderAttention(s) {
  const layer=Number($('layerSelect').value);
  const recent=state.steps.slice(Math.max(0,state.selected-7),state.selected+1);
  const table=el('table','attention-table'); table.setAttribute('aria-label',`Layer ${layer+1}のアテンション重み`);
  const head=el('tr'); head.append(el('th','','STEP ↓ / 文脈 →'));
  s.keys.forEach(k=>{const th=el('th');th.title=`位置 ${k.position} / ID ${k.id}: ${k.text}`;th.append(el('span','',visible(k.text)));head.append(th);});
  const thead=el('thead');thead.append(head);table.append(thead);
  const peak=Math.max(.001,...recent.flatMap(r=>r.attention[layer]));
  $('attentionScale').textContent=peak.toFixed(2);
  const tbody=el('tbody');
  recent.forEach(r=>{
    const tr=el('tr'); const name=el('td','',`${r.step} · ${visible(r.token)}`);
    name.title=`このトークンを選ぶ直前の計算。query: ${r.query}`;tr.append(name);
    const weights=new Map(r.keys.map((k,i)=>[k.position,r.attention[layer][i]]));
    s.keys.forEach(k=>{const td=el('td');const v=weights.get(k.position);td.style.backgroundColor=v===undefined?'#f4f6f9':color(v/peak);td.title=v===undefined?'このステップの表示範囲外':`位置 ${k.position}「${k.text}」への重み: ${v.toFixed(6)}`;tr.append(td);});
    tbody.append(tr);
  }); table.append(tbody);
  const wrap=$('attentionChart'), atEnd=wrap.scrollWidth-wrap.clientWidth-wrap.scrollLeft<25;
  const oldScroll=wrap.scrollLeft; wrap.replaceChildren(table);
  wrap.scrollLeft=atEnd ? wrap.scrollWidth : oldScroll;
}
function renderProbabilities(s) {
  if(embedded){
    const fragment=document.createDocumentFragment();
    const rows=[...s.candidates,{text:'その他',prob:s.other_prob,other:true}];
    rows.forEach(c=>{const row=el('div',`candidate-row${c.id===s.id?' selected':''}${c.other?' other-row':''}`);
      const label=el('span','candidate-text',visible(c.text));label.title=c.id===undefined?c.text:`${c.text} / Token ID ${c.id}`;
      const track=el('div','prob-track'),bar=el('div','prob-bar');bar.style.width=`${c.prob*100}%`;track.append(bar);
      row.append(label,track,el('span','prob-number',percent(c.prob)));fragment.append(row);
    }); $('probabilityChart').replaceChildren(fragment);
  } else {
    const table=document.createDocumentFragment(), heading=el('div','logit-head');
    heading.append(el('span','','候補'),el('span','','Logit'),el('span','','確率'));table.append(heading);
    s.candidates.slice(0,5).forEach(c=>{const row=el('div','logit-row');row.append(el('span','',visible(c.text)),el('span','',c.logit.toFixed(2)),el('span','',percent(c.prob)));table.append(row);});
    $('logitsTable').replaceChildren(table);
  }
}
function showStep(index) {
  if(!state.steps.length)return;
  state.selected=Math.min(Math.max(index,0),state.steps.length-1);
  const s=state.steps[state.selected];
  $('currentToken').textContent=visible(s.token); $('currentToken').title=s.token;
  $('tokenId').textContent=`TOKEN ID ${s.id}`; $('selectedProb').textContent=`確率 ${percent(s.selected_prob)}`;
  $('stepCount').textContent=`STEP ${s.step}${s.eos?' · EOS':''}`;
  $('contextCount').textContent=`${s.context_tokens} tokens`;
  $('speed').textContent=s.compute_tps.toFixed(1);
  $('speed').title='モデル計算・可視化データ抽出時間で計測。表示間隔と初回ロードを除きます。';
  $('timeline').disabled=false; $('timeline').max=state.steps.length; $('timeline').value=state.selected+1;
  $('timelineCount').textContent=`${state.selected+1} / ${state.steps.length}`;
  $('prevStep').disabled=state.selected===0; $('nextStep').disabled=state.selected===state.steps.length-1;
  $('liveButton').disabled=false; $('liveButton').textContent=state.follow&&state.busy?'追従中':'最新へ';
  if(!embedded){renderHidden(s);renderAttention(s);}
  renderProbabilities(s);
}

const sessionId = new URLSearchParams(location.search).get('session');
const selectionChannel = sessionId && 'BroadcastChannel' in window ? new BroadcastChannel(`llm-observatory-selection-${sessionId}`) : null;
const runs = new Map();
let lane = 'single', selectedPca = 0, playing = true;
if(selectionChannel)selectionChannel.onmessage = ({data}) => {
  if(data?.lane && runs.has(data.lane) && lane!==data.lane){lane=data.lane;renderLane();}
  if(data?.step==='live'){state.follow=true;if(state.steps.length)showStep(state.steps.length-1);}
  else if(Number.isInteger(data?.step) && state.steps.length){state.follow=false;showStep(data.step);}
};
function broadcastSelection(step){selectionChannel?.postMessage({lane,step});}
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
  if (run.pca && embedded) renderPca();
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
  if (event.type === 'pca') { run.pca = event; if (lane === key && embedded) renderPca(); }
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
for(const name of ['A','B']) $(name==='A'?'laneA':'laneB').addEventListener('click',()=>{if(runs.has(name)){lane=name;renderLane();broadcastSelection(state.follow?'live':state.selected);}});
$('layerSelect').addEventListener('change',()=>{const s=state.steps[state.selected];if(s){renderAttention(s);renderHidden(s);}});
$('timeline').addEventListener('input',()=>{state.follow=false;showStep(Number($('timeline').value)-1);broadcastSelection(state.selected);});
$('prevStep').addEventListener('click',()=>{state.follow=false;showStep(state.selected-1);broadcastSelection(state.selected);});
$('nextStep').addEventListener('click',()=>{state.follow=false;showStep(state.selected+1);broadcastSelection(state.selected);});
$('liveButton').addEventListener('click',()=>{state.follow=true;showStep(state.steps.length-1);broadcastSelection('live');});
async function controlAction(action) {
  try { const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId,action})});if(!r.ok)throw new Error((await r.json()).detail); }
  catch(e){error(e.message);}
}
$('pauseButton').addEventListener('click',()=>controlAction(playing?'pause':'resume'));
$('stepButton').addEventListener('click',()=>controlAction('step'));
setupSession();
