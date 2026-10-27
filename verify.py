"""Integration checks against the running server; real models, no mocked inference."""
import json
import math
import time
from pathlib import Path
import httpx

URL = 'http://150.89.227.185:8090'
results = []
client = httpx.Client(base_url=URL, timeout=180)

def payload(model='Qwen/Qwen3-0.6B', **kwargs):
    return dict(model=model, messages=[{'role':'user','content':'大阪の名物を3つ、名前だけ教えて。'}], max_tokens=32, delay_ms=0, **kwargs)

for model, layers in [('Qwen/Qwen3-0.6B',28), ('Qwen/Qwen3-8B',36)]:
    ticks = []
    kinds = []
    start = time.monotonic()
    with client.stream('POST', '/api/generate', json=payload(model)) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            event=json.loads(line)
            kinds.append(event['type'])
            assert event['type'] != 'error', event
            if event['type']=='token':
                assert len(event['hidden'])==layers+1
                assert all(len(row)==48 for row in event['hidden'])
                assert len(event['attention'])==layers
                assert all(len(row)==len(event['keys']) for row in event['attention'])
                assert all(0 <= v <= 1.001 for row in event['attention'] for v in row)
                assert all(sum(row)<=1.02 for row in event['attention'])
                assert math.isclose(sum(c['prob'] for c in event['candidates'])+event['other_prob'],1,abs_tol=1e-5)
                assert 0 <= event['selected_prob'] <= 1.001
                assert all(math.isfinite(v) for row in event['hidden'] for v in row)
                ticks.append(event)
    assert kinds[-1]=='done' and ticks, kinds
    result={'test':'real_generation','model':model,'passed':True,'steps':len(ticks),'seconds':round(time.monotonic()-start,2),'text':ticks[-1]['text'],'compute_tps':ticks[-1]['compute_tps']}
    print(json.dumps(result,ensure_ascii=False),flush=True);results.append(result)

assert client.post('/api/generate',json=payload('unlisted/model')).status_code==400
assert client.post('/api/generate',json=payload(),headers={'Origin':'https://example.com'}).status_code==403
bad=payload();bad['temperature']=-1
assert client.post('/api/generate',json=bad).status_code==422
results.append({'test':'validation_and_origin','passed':True})

request=payload('Qwen/Qwen3-8B');request.update(max_tokens=256,delay_ms=500)
with client.stream('POST','/api/generate',json=request) as response:
    for line in response.iter_lines():
        event=json.loads(line)
        assert event['type']!='error',event
        if event['type']=='token':
            assert client.post('/api/generate',json=payload()).status_code==409
            break
for _ in range(100):
    if not client.get('/api/system').json()['busy']:
        break
    time.sleep(.1)
else:
    raise AssertionError('Cancellation did not release GPU lock')
results.append({'test':'concurrency_and_cancellation','passed':True})

request=payload('Qwen/Qwen3-8B',temperature=0);request['max_tokens']=4
with client.stream('POST','/api/generate',json=request) as response:
    for line in response.iter_lines():
        event=json.loads(line)
        assert event['type']!='error',event
        if event['type']=='token':
            assert event['selected_prob']==1.0
            assert len(event['candidates'])==1
results.append({'test':'greedy_temperature_zero','passed':True})
Path('verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
