# LLM Observatory

DGX Spark上のローカルモデルを、チャットと可視化の2画面で観察するアプリです。Bootstrap 5、FastAPI、Transformers、PyTorchを使用します。可視化の数値は実際の推論結果から取得します。

## 使い方

VPN接続後、`http://150.89.227.185:8090/` を開きます。同じセッションIDの「チャット・トークン画面」と「内部状態画面」を、それぞれのモニターに開いてください。前者は左に会話、右に入力トークン・現在のトークン・PCA・候補確率を表示します。後者は層の活性・アテンション・スコア・システム情報・履歴を表示します。両画面は同じ生成イベントと選択ステップに同期し、通常のPC画面ではページ全体をスクロールせずに収まります。`/single` では従来の1画面版も使えます。

チャットまたは内部状態画面で「一時停止」「1トークン進める」「再開」を操作できます。チャット画面の「同じ質問でA/B比較」は、同じ質問・会話履歴・乱数シードでTemperature A/Bの生成を順番に実行します。両画面でA/Bを切り替えられ、内部状態画面では出力と候補確率を並べて比較できます。生成中のチャットを閉じると推論は停止します。

入力トークンをクリックすると、最終層の各入力位置のhidden stateから計算したPCA散布図で該当点を選択します。PCAの距離は同じ質問内での相対値です。アテンションは選択層の全ヘッド平均で、生成トークンを選ぶ直前の最後の文脈位置が過去に向けた重みです。意味の説明や因果を表すものではありません。

## モデル

選択肢は `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-8B`、`openai/gpt-oss-20b` です。モデルファイルがローカルキャッシュに揃っていないものは選択できません。推論は `local_files_only=True`、サービスは `HF_HUB_OFFLINE=1` で動作し、外部AI APIには送信しません。GPUには同時に1モデル・1生成だけをロードします。gpt-oss 20BはOpenAIのオープンウェイトモデルで、Transformersのeager attentionから可視化データを取得します。導入と実機確認はDGX Spark上で行ってください。

## DGX Sparkでの導入

既存配置先は `/home/sakurai/llm-observatory`、ユーザーサービスは `llm-observatory.service` です。既存のPyTorch/CUDAを維持し、仮想環境に不足する依存関係を追加します。`download_models.py` はgpt-ossの重みとMXFP4用Tritonカーネルを取得し、オフライン推論に備えます。

```bash
cd /home/sakurai/llm-observatory
.venv/bin/python -m pip install -r requirements.txt
HF_HUB_OFFLINE=0 .venv/bin/python download_models.py openai/gpt-oss-20b
systemctl --user restart llm-observatory
.venv/bin/python verify_duo.py
```

新規環境では適合するCUDA版PyTorchを先に入れ、`python -m venv .venv`、`pip install -r requirements.txt`、`python download_models.py` の順に実施します。systemdのユーザーサービスのパスと待受IPは実環境に合わせます。モデルは数十GBあるため十分な空き容量が必要です。

## APIと検証

- `GET /api/models`: 導入済みモデル一覧。
- `POST /api/generate`: NDJSONの生成イベント。`session_id` と `run_label`（`single`/`A`/`B`）を指定すると画面同期に配信します。
- `GET /api/events/{session_id}`: 画面同期用SSE。後から開いた画面にもセッション内の最近のイベントを再送します。
- `POST /api/control`: 対象セッションの`pause`、`step`、`resume`。
- `GET /api/system`: GPUと推論状態。

`python verify.py` はQwenの推論・確率・アテンション等の旧統合テストです。`python verify_duo.py` は2画面プロトコル、PCA、一時停止、A/B比較を稼働中のサーバーで検査します。ブラウザーの表示とgpt-ossの実推論も別途確認してください。

## 制限

展示用のVPN内アプリで、ユーザー認証はありません。セッションIDはランダムに生成します。入力文脈は512トークン、最大生成は256トークンです。全層のattention・hidden stateを取得するため、通常のチャット推論より計算量・メモリ使用量が大きくなります。gpt-oss 20Bの初回ロードは時間がかかる可能性があります。

参考: [OpenAI gpt-oss](https://openai.com/open-models/)、[Hugging Face TransformersのGptOss](https://huggingface.co/docs/transformers/model_doc/gpt_oss)、[Bootstrap](https://getbootstrap.com/docs/5.3/getting-started/introduction/)
