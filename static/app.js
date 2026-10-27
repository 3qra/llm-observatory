'use strict';
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
  const fragment=document.createDocumentFragment();
  const rows=[...s.candidates,{text:'その他',prob:s.other_prob,other:true}];
  rows.forEach(c=>{const row=el('div',`candidate-row${c.id===s.id?' selected':''}${c.other?' other-row':''}`);
    const label=el('span','candidate-text',visible(c.text));label.title=c.id===undefined?c.text:`${c.text} / Token ID ${c.id}`;
    const track=el('div','prob-track'),bar=el('div','prob-bar');bar.style.width=`${c.prob*100}%`;track.append(bar);
    row.append(label,track,el('span','prob-number',percent(c.prob)));fragment.append(row);
  }); $('probabilityChart').replaceChildren(fragment);
  const table=document.createDocumentFragment(), heading=el('div','logit-head');
  heading.append(el('span','','候補'),el('span','','Logit'),el('span','','確率'));table.append(heading);
  s.candidates.slice(0,5).forEach(c=>{const row=el('div','logit-row');row.append(el('span','',visible(c.text)),el('span','',c.logit.toFixed(2)),el('span','',percent(c.prob)));table.append(row);});
  $('logitsTable').replaceChildren(table);
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
  renderHidden(s);renderAttention(s);renderProbabilities(s);
}
function handleEvent(event) {
  if(event.type==='status')status(event.message,true);
  if(event.type==='start')startEvent(event);
  if(event.type==='token') {
    state.steps.push(event);
    state.assistant.querySelector('.bubble').textContent=event.text;
    const chat=$('chatMessages');chat.scrollTop=chat.scrollHeight;
    if(state.follow)showStep(state.steps.length-1);
    else {$('timeline').max=state.steps.length;$('timelineCount').textContent=`${state.selected+1} / ${state.steps.length}`;$('nextStep').disabled=false;}
  }
  if(event.type==='error')throw new Error(event.message);
  if(event.type==='done') {
    status(event.reason==='length'?'上限まで生成しました':'生成完了');
    state.assistant.append(el('div','message-note',`${event.tokens} tokens · ${event.seconds.toFixed(1)}秒${event.reason==='length'?' · 最大生成数に到達':''}`));
  }
}
async function submit(event) {
  event.preventDefault();if(state.busy)return;
  const prompt=$('prompt').value.trim();if(!prompt)return;
  for(const id of ['topK','topP','maxTokens','delay','seed'])if(!$(id).reportValidity())return;
  error('');clearCharts();
  const userMessage={role:'user',content:prompt};
  const payload={model:$('modelSelect').value,messages:[...state.messages.slice(-22),userMessage],temperature:Number($('temperature').value),top_k:Number($('topK').value),top_p:Number($('topP').value),max_tokens:Number($('maxTokens').value),delay_ms:Number($('delay').value),seed:Number($('seed').value)};
  message('user',prompt);state.assistant=message('assistant','');state.assistant.classList.add('streaming');
  $('prompt').value='';busy(true);status('モデルを準備しています',true);
  state.controller=new AbortController();let completed=false;
  try {
    const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:state.controller.signal});
    if(!response.ok){const body=await response.json();throw new Error(typeof body.detail==='string'?body.detail:'入力設定を確認してください。');}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines)if(line.trim()){const data=JSON.parse(line);handleEvent(data);if(data.type==='done')completed=true;}if(done)break;}
    if(buffer.trim()){const data=JSON.parse(buffer);handleEvent(data);if(data.type==='done')completed=true;}
    if(!completed)throw new Error('生成中に接続が切れました。もう一度送信してください。');
  }catch(err){
    if(err.name==='AbortError')status('停止しました');
    else{error(err.message);status('生成を確認してください');state.controller.abort();}
    state.assistant.classList.add('interrupted');
    state.assistant.append(el('div','message-note',err.name==='AbortError'?'生成を停止しました':'生成が完了しませんでした'));
    if(!state.steps.length)$('prompt').value=prompt;
  }finally{
    const text=state.steps.at(-1)?.text||'';
    const bubble=state.assistant.querySelector('.bubble');
    if(text){bubble.replaceChildren(...text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map(part=>part.startsWith('**')&&part.endsWith('**')?el('strong','',part.slice(2,-2)):part.startsWith('`')&&part.endsWith('`')?el('code','',part.slice(1,-1)):document.createTextNode(part)));}
    if(text)state.messages.push(userMessage,{role:'assistant',content:text});
    state.assistant.classList.remove('streaming');state.controller=null;busy(false);$('liveButton').textContent='最新へ';$('prompt').focus();
  }
}
async function init() {
  try {
    const [modelsResponse,systemResponse]=await Promise.all([fetch('/api/models'),fetch('/api/system')]);
    if(!modelsResponse.ok||!systemResponse.ok)throw new Error('サーバーに接続できません。ページを再読み込みしてください。');
    const data=await modelsResponse.json(),system=await systemResponse.json();state.models=data.models;
    $('modelSelect').replaceChildren(...state.models.map(m=>{const o=el('option','',m.name);o.value=m.id;return o;}));
    const preferred=state.models.find(m=>m.id==='Qwen/Qwen3-8B');if(preferred)$('modelSelect').value=preferred.id;
    modelChange();$('gpuName').textContent=system.device;
  }catch(e){error(e.message);$('sendButton').disabled=true;status('接続できません');}
}
function modelChange(){const model=state.models.find(m=>m.id===$('modelSelect').value);$('modelDescription').textContent=model?`${model.description} · ${model.layers} layers`:'';}
$('chatForm').addEventListener('submit',submit);
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('chatForm').requestSubmit();}});
$('stopButton').addEventListener('click',()=>state.controller?.abort());
$('modelSelect').addEventListener('change',modelChange);
$('temperature').addEventListener('input',()=>$('temperatureValue').textContent=Number($('temperature').value).toFixed(1));
$('layerSelect').addEventListener('change',()=>{const s=state.steps[state.selected];if(s){renderAttention(s);renderHidden(s);}});
$('timeline').addEventListener('input',()=>{state.follow=false;showStep(Number($('timeline').value)-1);});
$('prevStep').addEventListener('click',()=>{state.follow=false;showStep(state.selected-1);});
$('nextStep').addEventListener('click',()=>{state.follow=false;showStep(state.selected+1);});
$('liveButton').addEventListener('click',()=>{state.follow=true;showStep(state.steps.length-1);});
$('resetButton').addEventListener('click',()=>{state.messages=[];clearCharts();$('chatMessages').replaceChildren(el('div','chart-empty','新しい会話を始めましょう。'));$('inputTokens').replaceChildren(el('span','empty-inline','入力がモデルの処理単位に分かれます'));status('入力待ち');error('');$('prompt').value='';$('prompt').focus();});
$('helpButton').addEventListener('click',()=>{$('help').hidden=!$('help').hidden;$('helpButton').setAttribute('aria-expanded',String(!$('help').hidden));});
document.querySelectorAll('.example-button').forEach(b=>b.addEventListener('click',()=>{$('prompt').value=b.dataset.prompt;$('chatForm').requestSubmit();}));
init();
