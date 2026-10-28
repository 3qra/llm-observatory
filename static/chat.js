'use strict';
const $ = id => document.getElementById(id);
const el = (tag, cls, value) => { const n=document.createElement(tag); if(cls)n.className=cls; if(value!==undefined)n.textContent=value;return n; };
const randomId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), b=>b.toString(16).padStart(2,'0')).join('');
const params = new URLSearchParams(location.search);
const sessionId = /^[a-zA-Z0-9_-]{8,64}$/.test(params.get('session') || '') ? params.get('session') : randomId();
if (params.get('session') !== sessionId) history.replaceState(null,'',`/chat?session=${encodeURIComponent(sessionId)}`);
const state = {messages:[],busy:false,controller:null,playing:true,models:[],generatedSteps:0};
$('sessionLabel').textContent = `SESSION ${sessionId.slice(0,8)}`;
$('vizFrame').src=`/viz?session=${encodeURIComponent(sessionId)}&embedded=1`;
const vizLink=el('a','btn btn-sm btn-outline-primary','内部状態の画面を開く ↗');vizLink.href=`/viz?session=${encodeURIComponent(sessionId)}`;vizLink.target='_blank';vizLink.rel='noopener';
$('modelDescription').after(vizLink);
function status(value, active=false) {$('runStatus').textContent=value;$('runStatus').classList.toggle('active',active);}
function error(value) {$('errorBox').textContent=value;$('errorBox').hidden=!value;}
function busy(value) {
  state.busy=value;
  $('sendButton').hidden=value;$('stopButton').hidden=!value;
  $('modelSelect').disabled=value;$('resetButton').disabled=value;$('settingsFields').disabled=value;
  $('modeAuto').disabled=value;$('manualMode').disabled=value;$('compareButton').disabled=value||$('manualMode').checked;$('prompt').readOnly=value;
  $('pauseButton').disabled=!value;$('stepButton').disabled=!value;$('backButton').disabled=!value||!$('manualMode').checked||state.generatedSteps===0;
  document.querySelectorAll('.example-button').forEach(b=>b.disabled=value);
}
function message(role, value, label) {
  $('welcome')?.remove();
  const box=el('article',`message ${role}`),head=el('div','message-label');
  head.append(el('span','avatar',role==='user'?'U':'Ai'),el('span','',label|| (role==='user'?'YOU':$('modelSelect').selectedOptions[0]?.textContent||'MODEL')));
  box.append(head,el('div','bubble',value));$('chatMessages').append(box);
  $('chatMessages').scrollTop=$('chatMessages').scrollHeight;
  return box;
}
function payload(prompt, temperature, runLabel, comparisonId, prior) {
  return {session_id:sessionId,run_label:runLabel,comparison_id:comparisonId,
    model:$('modelSelect').value,messages:[...prior.slice(-22),{role:'user',content:prompt}],
    temperature,top_k:Number($('topK').value),top_p:Number($('topP').value),
    max_tokens:Number($('maxTokens').value),delay_ms:Number($('delay').value),seed:Number($('seed').value),
    manual_mode:runLabel==='single'&&$('manualMode').checked};
}
async function streamRun(data, box) {
  state.controller=new AbortController();
  const response=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:state.controller.signal});
  if(!response.ok) { const detail=await response.json();throw new Error(typeof detail.detail==='string'?detail.detail:'生成を開始できませんでした。'); }
  let {cursor}=await response.json(),done=false,last='';
  while(!done) {
    const poll=await fetch(`/api/events/${encodeURIComponent(sessionId)}/poll?cursor=${cursor}`,{signal:state.controller.signal,cache:'no-store'});
    if(!poll.ok)throw new Error('生成イベントを取得できませんでした。');
    const batch=await poll.json();cursor=batch.cursor;
    for(const event of batch.events) {
      if(event.type==='status')status(event.message,true);
      if(event.type==='start')status(data.manual_mode&&!state.playing?'次のトークン待ち':'生成しています',state.playing);
      if(event.type==='token') {last=event.text;state.generatedSteps=event.step;box.querySelector('.bubble').textContent=last;$('backButton').disabled=!data.manual_mode||state.generatedSteps===0;$('chatMessages').scrollTop=$('chatMessages').scrollHeight;if(!state.playing)status('次のトークン待ち');}
      if(event.type==='rewind') {last=event.text;state.generatedSteps=event.step;box.querySelector('.bubble').textContent=last;$('backButton').disabled=state.generatedSteps===0;status('1トークン戻しました');}
      if(event.type==='error')throw new Error(event.message);
      if(event.type==='cancelled'){const err=new Error('生成を停止しました。');err.name='AbortError';throw err;}
      if(event.type==='done') {done=true;status('生成完了');box.append(el('div','message-note',`${event.tokens} tokens · ${event.seconds.toFixed(1)}秒`));}
    }
  }
  return last;
}
async function send(compare=false) {
  if(state.busy)return;
  const prompt=$('prompt').value.trim();if(!prompt)return;
  for(const id of ['topK','topP','maxTokens','delay','seed'])if(!$(id).reportValidity())return;
  if(compare&&!$('compareTemperature').reportValidity())return;
  const temperature=Number($('temperature').value), temperatureB=Number($('compareTemperature').value);
  const prior=[...state.messages];
  const comparisonId=compare?randomId():null;
  state.playing=!$('manualMode').checked||compare;state.generatedSteps=0;
  $('pauseButton').textContent=state.playing?'一時停止':'自動再開';
  error('');message('user',prompt);$('prompt').value='';busy(true);if($('manualMode').checked&&!compare)$('prompt').focus();status('モデルを準備しています',true);
  try {
    if(compare) {
      const a=message('assistant','',`A · ${$('modelSelect').selectedOptions[0].textContent} · T=${temperature.toFixed(1)}`);
      a.classList.add('streaming');
      await streamRun(payload(prompt,temperature,'A',comparisonId,prior),a);
      a.classList.remove('streaming');
      const b=message('assistant','',`B · ${$('modelSelect').selectedOptions[0].textContent} · T=${temperatureB.toFixed(1)}`);
      b.classList.add('streaming');
      await streamRun(payload(prompt,temperatureB,'B',comparisonId,prior),b);
      b.classList.remove('streaming');
    } else {
      const assistant=message('assistant','');assistant.classList.add('streaming');
      const text=await streamRun(payload(prompt,temperature,'single',null,prior),assistant);
      assistant.classList.remove('streaming');
      if(text)state.messages.push({role:'user',content:prompt},{role:'assistant',content:text});
    }
  } catch(e) {
    error(e.name==='AbortError'?'生成を停止しました。':e.message);
    status(e.name==='AbortError'?'停止しました':'生成エラー');
    document.querySelector('.message.assistant:last-of-type')?.classList.remove('streaming');
  } finally {state.controller=null;busy(false);state.playing=true;$('pauseButton').textContent='一時停止';$('prompt').focus();}
}
async function control(action) {
  try {const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sessionId,action})});if(!r.ok)throw new Error((await r.json()).detail);
    const result=await r.json();if(action==='stop'){status('停止しています');return;}state.playing=result.playing;$('pauseButton').textContent=state.playing?'一時停止':'自動再開';status(action==='step'?'1トークン進めています':action==='back'?'1トークン戻しています':state.playing?'生成しています':'次のトークン待ち',state.playing);
  }catch(e){error(e.message);}
}
async function init() {
  try {const r=await fetch('/api/models');if(!r.ok)throw new Error('モデル一覧を取得できません。');
    const data=await r.json();state.models=data.models;
    $('modelSelect').replaceChildren(...data.models.map(m=>{const o=el('option','',m.name+(m.available?'':' · 未導入'));o.value=m.id;o.disabled=!m.available;return o;}));
    const preferred=data.models.find(m=>m.id==='openai/gpt-oss-20b'&&m.available)||data.models.find(m=>m.available);
    if(preferred)$('modelSelect').value=preferred.id;else throw new Error('利用可能なモデルがありません。サーバーにモデルを導入してください。');
    modelChange();
  }catch(e){error(e.message);$('sendButton').disabled=true;$('compareButton').disabled=true;status('接続できません');}
}
function modelChange(){const m=state.models.find(m=>m.id===$('modelSelect').value);$('modelDescription').textContent=m?`${m.description} · ${m.layers} layers`:'';}
$('chatForm').addEventListener('submit',e=>{e.preventDefault();send(false);});
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();e.stopPropagation();if(state.busy&&$('manualMode').checked)control('step');else if(!state.busy)$('chatForm').requestSubmit();}});
$('compareButton').addEventListener('click',()=>send(true));
$('stopButton').addEventListener('click',()=>control('stop'));
$('pauseButton').addEventListener('click',()=>control(state.playing?'pause':'resume'));
$('stepButton').addEventListener('click',()=>control('step'));
$('backButton').addEventListener('click',()=>control('back'));
document.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.ctrlKey&&!e.altKey&&!e.metaKey&&!e.repeat&&state.busy&&$('manualMode').checked&&!e.target.closest('input,select,textarea,[contenteditable="true"]')){e.preventDefault();control('step');}});
$('temperature').addEventListener('input',()=>$('temperatureValue').textContent=Number($('temperature').value).toFixed(1));
$('modelSelect').addEventListener('change',modelChange);
function modeChange(){const manual=$('manualMode').checked;$('modeHint').textContent=manual?'送信後、Enterかボタンで最初のトークンから進めます':'送信すると自動で生成します';$('compareButton').disabled=manual||state.busy;$('compareButton').title=manual?'A/B比較は自動で進める場合に使えます。':'';}
$('modeAuto').addEventListener('change',modeChange);
$('manualMode').addEventListener('change',modeChange);
$('resetButton').addEventListener('click',()=>{state.messages=[];$('chatMessages').replaceChildren();error('');status('入力待ち');$('prompt').value='';$('prompt').focus();});
window.addEventListener('pagehide',()=>{if(state.busy)navigator.sendBeacon('/api/control',new Blob([JSON.stringify({session_id:sessionId,action:'stop'})],{type:'application/json'}));});
document.querySelectorAll('.example-button').forEach(b=>b.addEventListener('click',()=>{$('prompt').value=b.dataset.prompt;send(false);}));
init();
