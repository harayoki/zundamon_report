# ReportVox（Preview）

RRWADME 

※ 口調変換は作成中

## ライセンス
未定です。利用するプロダクトの中で一番厳しい物に合わせる事になります。MITにしたいでがGPLなどになるかもしれません。構成が確定したら決定します。

## ⚠️ 依存関係に関する重要な注意事項
ffmpeg が ファイルフォーマット変換と 話者分離処理に使われます。
現在、Windows 環境において pyannote.audio と PyTorch 周辺ライブラリの間に深刻な互換性問題が複数確認されています。 本ツールではこれらを回避するためのパッチを内蔵していますが、以下の環境構築が必須となります。

### 1. NumPy のバージョン固定
NumPy 2.0 以降では話者分離処理時にエラーが起こるため、1.26.x 系を強制してください。
```bash
pip install "numpy<2"
```

### 2. FFmpeg Shared 版の導入 (Windows 必須)
話者分離（torchcodec）の動作には、Shared 版（共有DLL版）の FFmpeg が必須です。
- 取得元: gyan.dev から "release full shared" を選択。
- 設定: 解凍した bin フォルダのパスを環境変数のPATHに設定します。それでもエラーが起こる場合は FFMPEG_SHARED_PATH 環境変数も同じ内容で設定してください。
  ```bash
  # 例 (Git Bash)
  export FFMPEG_SHARED_PATH="F:/Program Files/ffmpeg-7.1.1-full_build-shared/bin"
  ```
  ※本ツールは起動時にこの環境変数を参照し、自動的に DLL 探索パスへ追加します。

### 3. PyTorch 2.6+ セキュリティ制限の回避
最新の PyTorch では weights_only=True がデフォルトとなり、モデル読み込みで UnpicklingError が発生します。 本ツールは起動時に torch.load をパッチし、この制限を自動で回避します。

## インストール
Python 3.11+ を推奨します。
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "numpy<2"
```

## 依存サービス
### OpenAI API で利用するモデルの指定
LLM バックエンドに `openai` を選択した場合、呼び出すモデルは環境変数 `OPENAI_MODEL` で指定できます。未設定時のデフォルトは **gpt-4o-mini** です。

例: 高精度な `gpt-4o` を使う場合

```bash
export OPENAI_MODEL=gpt-4o
```

主な選択肢（OpenAI 側の提供状況に応じて利用可否が変わるため、最新情報は公式ドキュメントを確認してください）:

- `gpt-4o-mini`（デフォルト / 高速・低コスト）
- `gpt-4o`（高精度）
- `gpt-4.1` / `gpt-4.1-mini`（最新系モデル。利用可否は API の提供状況を参照）

`OPENAI_API_KEY` と合わせて設定することで、`--llm openai` 実行時に希望するモデルで推論できます。

### pyannote.audio の認証
話者分離機能（--speakers auto/2）を利用するには、Hugging Face の Classic Token (Read) が必要です。

1. 以下のリポジトリですべて利用規約に同意（Accept）してください。
   - pyannote/speaker-diarization-3.1
   - pyannote/segmentation-3.0
2. その他認証が求められるエラーが表示された場合も各ページで利用規約に同意してください。
3.  Hugging Face Settings でトークンを作成。
4. トークンを環境変数 PYANNOTE_TOKEN または HF_TOKEN で渡してください。

## ローカルLLM (Ollama) との連携
文字起こし結果の自動校正 (`--review-transcript-llm`) などで、ローカルで動作するLLMと連携することができます。本ツールではOllama経由での利用を想定しています。

### 1. Ollamaのインストール
以下の公式サイトからOllamaをダウンロードし、インストールしてください。
- [https://ollama.com/](https://ollama.com/)

### 2. モデルのダウンロード
次に、利用したいLLMモデルをダウンロードします。ターミナル（コマンドプロンプトやPowerShell）で以下のコマンドを実行します。
性能とPCスペックのバランスに応じて選択してください。

- **高精度モデル (推奨)**:
  ```bash
  ollama pull llama3
  ```
- **軽量モデル**:
  ```bash
  ollama pull gemma:2b
  ```

### 3. モデルの指定
使用するモデルは `--ollama-model` 引数で指定するのが最も簡単です。

```bash
# gemma:2b モデルを使って実行
python -m reportvox input.wav --llm local --ollama-model gemma:2b
```

引数を指定しない場合は、従来通り環境変数 `LOCAL_LLM_MODEL` が参照されます。

- **Windows (コマンドプロンプト) の場合:**
  ```shell
  set LOCAL_LLM_MODEL=llama3
  ```
- **Windows (PowerShell) の場合:**
  ```powershell
  $env:LOCAL_LLM_MODEL="llama3"
  ```
- **Linux / macOS の場合:**
  ```bash
  export LOCAL_LLM_MODEL=llama3
  ```

どちらも指定されていない場合のデフォルトは `llama3` です。

### 4. 実行
コマンド実行時に `--llm local` オプションを追加して、ローカルLLMを指定します。

```bash
# 例: ローカルLLMで文字起こしを自動校正
python -m reportvox input.wav --review-transcript-llm --llm local
```

### VOICEVOX Engine の起動

本ツールはVOICEVOXアプリケーションではなく、VOICEVOX Engine（コアとなる合成エンジン）と連携します。

1.  **Engineのダウンロード:**
    VOICEVOX公式サイトから「VOICEVOX Engine」をダウンロードしてください。
    *   [VOICEVOX公式サイト](https://voicevox.hiroshiba.jp/)
    （お使いのPC環境に合わせてCPU版またはGPU版を選択してください。）

2.  **Engineの起動:**
    ダウンロードしたzipファイルを解凍後、中にある `run.exe` を実行してください。
    実行するとコマンドプロンプトの画面が開き、以下のようなログが表示されれば起動成功です。
    ```
    INFO:     Uvicorn running on http://127.0.0.1:50021 (Press CTRL+C to quit)
    ```
    本ツールを実行する際は、このEngineが起動している状態を維持してください。

3.  **URLの指定（任意）:**
    Engineがデフォルト（`http://127.0.0.1:50021`）以外のURLやポートで起動している場合、本ツール実行時に `--voicevox-url` オプションで指定してください。

## 使い方 (コマンド例)
```bash
# 自動話者判定、デフォルト（ずんだもん+めたん）
python -m reportvox input.wav

# ずんだもんの自己紹介（役割を指定）
python -m reportvox input.m4a --zunda-senior-job エンジニア --zunda-junior-job 自動化係

# 話者 2 人を強制し、キャラ指定
python -m reportvox input.wav --speakers 2 --speaker1 zundamon --speaker2 metan

# 中断したワークフローを再開（work/<run_id> を指定）
python -m reportvox input.wav --resume 20240101-120000

# mp3 を作ったあとに mp4 だけを生成（work/<run_id> を指定）
python -m reportvox.video_cli --run-id 20240101-120000

# 作業ディレクトリを指定せず、必要なファイルを個別指定して mp4 を生成
python -m reportvox.video_cli \
  --stylized work/20240101-120000/stylized.json \
  --placements work/20240101-120000/placements.json \
  --segments-dir work/20240101-120000 \
  --audio out/input_report.mp3

# work/<run_id> を残して設定変更や再出力に使う（開発用）
python -m reportvox input.wav --keep-work

# 出力ファイル名を指定（mp3/字幕にも同名を適用）
python -m reportvox input.wav --output-name meeting_2024

# 既存の出力がある場合も確認せず上書き
python -m reportvox input.wav -f

# 字幕 (SRT) の出力設定（デフォルトで2話者を1つにまとめたSRTを out/ に保存）
python -m reportvox input.wav --subtitles off    # SRT を出力しない
python -m reportvox input.wav --subtitles split  # 話者ごとに別ファイルを追加で出力
# 動画へ焼き込む字幕は常にキャラクター別レイアウトで生成され、--subtitles は out/ に保存する SRT の形式だけを切り替えます。

# 字幕付きの動画 (mp4/mov) を出力
python -m reportvox input.wav --mp4  # mp4 を生成
python -m reportvox input.wav --mov --video-width 1280 --video-height 720 --video-fps 30  # 透明movをHD/30fpsで生成
# 動画上に複数画像を順番に表示（デフォルトでは尺を等分して表示）
python -m reportvox input.wav --mp4 --video-images image1.png image2.png image3.png
# 画像の開始秒や位置・拡大率をまとめて指定
python -m reportvox input.wav --mp4 --video-images image1.png image2.png --video-image-times 5 18 --video-image-pos 900,520 --video-image-scale 0.4

# 動画ファイルも入力可、MP3も生成
python -m reportvox input.mp4 --speakers auto --mp3
```

## 主なオプション
- **音声入力と話者設定**
  - `--voicevox-url`: VOICEVOX Engine のベースURL（デフォルト: `http://127.0.0.1:50021`）。
  - `--speakers {auto,1,2}`: 話者数の扱い（auto=自動判定/1=単一話者/2=2人固定）。
  - `--diarization-threshold`: 話者分離のクラスタリング閾値（0.0-1.0、デフォルト 0.8）。
  - `--speaker1` / `--speaker2`: 話者に割り当てるキャラクターID（デフォルト: ずんだもん/めたん）。
  - `--color1` / `--color2`: 各話者で使用するカラーコード（`#RRGGBB`）。
  - `--intro1` / `--intro2`: 最初の挨拶文の上書き、`--no-intro` で挨拶自動挿入をオフ。
  - `--zunda-senior-job` / `--zunda-junior-job`: ずんだもんの役割設定（未指定時は自動決定）。

- **出力形式とファイル操作**
  - `--mp3` / `--bitrate`: mp3 を生成し、ビットレートを指定（デフォルト `192k`）。
  - `--mp4` / `--mov`: 字幕付きの mp4 / 透明 mov を生成。
  - `--output-name`: 出力ファイル名のベースを指定（拡張子不要）。
  - `-f, --force`: 既存の出力を確認なしで上書き。
  - `--ffmpeg-path`: ffmpeg 実行ファイルのパス。
  - `--keep-work`: `work/` 以下の中間ファイルを残す。
  - `--max-pause`: セリフ間の無音を詰める秒数（負の値で詰めない）。

- **音声認識と LLM 設定**
  - `--model`: Whisper モデルサイズ（デフォルト `large-v3`）。
  - `--llm {none,openai,ollama}`: 口調変換に使う LLM バックエンド。
  - `--ollama-host` / `--ollama-port` / `--ollama-model`: ローカルLLM接続先とモデル指定。
  - `--no-style-with-llm`: 口調変換を LLM で行わない。
  - `--no-linebreak-with-llm`: 長いセリフへの改行挿入を無効化、`--linebreak-min-chars` で改行検討の長さを調整。
  - `--kana-level`: 漢字のふりがなレベル（none/elementary/junior/high/college）。

- **文字起こしのレビュー**
  - `--review-transcript`: 文字起こし保存後に処理を停止し手動修正。
  - `--review-transcript-llm`: 文字起こし後に LLM で誤字脱字を校正して続行。
  - `--skip-review-transcript`: 自動校正をスキップ。

- **字幕・動画出力**
  - `--subtitles {off,all,split}`: SRT の出力形式（生成なし/全発話/話者別）。
  - `--subtitle-max-chars` / `--subtitle-font` / `--subtitle-font-size`: 字幕の1行文字数・フォント・サイズ設定。
  - `--video-width` / `--video-height` / `--video-fps`: 動画の解像度とフレームレート。
  - `--video-images` / `--video-image-times` / `--video-image-pos` / `--video-image-scale`: 動画上に配置する画像、開始時刻・位置・拡大率。

- **再実行・キャッシュ**
  - `--resume`: 既存の `work/<run_id>` を指定して処理を再開。
  - `--resume-from`: 再開時に特定ステップから実行。
  - `--force-transcribe` / `--force-diarize`: 文字起こし・話者分離のキャッシュを無視して再実行。
## カスタム話者・口癖の追加
新しいキャラクターの追加や口癖の調整方法は「[キャラクター追加・口癖設定ガイド](docs/characters.md)」にまとめています。

## トラブルシューティング（パッチ対応済み事項）
以下の問題は diarize.py 内のモンキーパッチにより自動修正されます：

- TypeError (use_auth_token): huggingface_hub の引数名変更エラーを修正。
- WeightsUnpickler error: PyTorch 2.6+ のセキュリティ制限を自動解除。
- AttributeError (with_character): AlignedSegment クラスのメソッド不足を解消。
- FFmpeg DLL Path: 環境変数 FFMPEG_SHARED_PATH からの動的ロードに対応。
