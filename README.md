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
### pyannote.audio の認証
話者分離機能（--speakers auto/2）を利用するには、Hugging Face の Classic Token (Read) が必要です。

1. 以下のリポジトリですべて利用規約に同意（Accept）してください。
   - pyannote/speaker-diarization-3.1
   - pyannote/segmentation-3.0
2. その他認証が求められるエラーが表示された場合も各ページで利用規約に同意してください。
3.  Hugging Face Settings でトークンを作成。
4. トークンを環境変数 PYANNOTE_TOKEN または --hf-token で渡してください。

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

# work/<run_id> を残して設定変更や再出力に使う（開発用）
python -m reportvox input.wav --keep-work

# 出力ファイル名を指定（mp3/字幕にも同名を適用）
python -m reportvox input.wav --output-name meeting_2024

# 既存の出力がある場合も確認せず上書き
python -m reportvox input.wav -f

# 字幕 (SRT) を同時出力
python -m reportvox input.wav --subtitles all  # 2話者を1つのファイルにまとめて出力
python -m reportvox input.wav --subtitles split  # 話者ごとに別ファイルを出力

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
- --speaker1: 主話者に割り当てるキャラクターID。
- --speaker2: 副話者に割り当てるキャラクターID。
- --intro1: 話者1の最初の挨拶文を指定します。
- --intro2: 話者2の最初の挨拶文を指定します。
- --no-intro: 最初の挨拶文の自動挿入を無効化します。
- --zunda-senior-job, --zunda-junior-job: speaker1がずんだもんの場合のデフォルト挨拶を生成します（--intro1が指定されている場合はそちらが優先されます）。
- --speakers {auto,1,2}: 話者数の扱い。
- --diarization-threshold: 話者分離のクラスタリング閾値（0.0-1.0）。値が低いほど、声質が似ている話者も別人として分離されやすくなります。デフォルトは 0.8。
- --model: Whisper のモデルサイズ。
- --voicevox-url: VOICEVOX Engine のベース URL。
- --speed-scale: VOICEVOX での読み上げ速度（デフォルト 1.1）。発話の間隔は維持されるため、音声全体の長さはほぼ変わりません。
- --output-name: 出力ファイル名のベース（拡張子不要）。mp3/字幕にも同名を適用。
- -f, --force: 出力の上書き確認をスキップ。
- --mp3: mp3 を生成（out/ には mp3 だけを出力）。
- --bitrate: mp3 出力時のビットレート。
- --resume <run_id>: 中断した工程から再開。
- --keep-work: work/<run_id> を削除せず保持（開発/再出力向け）。
- --llm {none,openai,ollama,gemini}: LLMバックエンドの指定。
- --ollama-host: Ollamaのホスト名。
- --ollama-port: Ollamaのポート番号。
- --ollama-model: Ollamaで使用するモデル名。デフォルトは環境変数 LOCAL_LLM_MODEL または 'llama3'。
- --hf-token: Hugging Face Token。
- --subtitles {off,all,split}: SRT 字幕の出力モード。
- --subtitle-max-chars: 字幕1枚あたりの最大文字数（デフォルト 25、0 で無制限）。
- --subtitle-font: 動画用ASS字幕に使用するフォント名（libass で解決可能なもの）。
- --subtitle-font-size: 動画用ASS字幕のフォントサイズ。デフォルトは 96 pt。
- --review-transcript: 文字起こし結果を保存したあとで処理を停止し、transcript.json を手動修正したうえで表示される再開コマンドを実行して続行できるようにする。
- --review-transcript-llm: 文字起こし結果を保存したあと、LLM で明らかな誤字脱字を自動校正してから続行する（--llm でバックエンド指定）。
- --mp4: 字幕焼き込み済みの mp4 を生成。
- --mov: 透明背景の mov (ProRes 4444) を生成。
- --video-width / --video-height: 動画の解像度をピクセルで指定（デフォルト 1920x1080）。
- --video-fps: 動画のフレームレート。デフォルト 24 fps。

## カスタム話者・口癖の追加
新しいキャラクターの追加や口癖の調整方法は「[キャラクター追加・口癖設定ガイド](docs/characters.md)」にまとめています。

## トラブルシューティング（パッチ対応済み事項）
以下の問題は diarize.py 内のモンキーパッチにより自動修正されます：

- TypeError (use_auth_token): huggingface_hub の引数名変更エラーを修正。
- WeightsUnpickler error: PyTorch 2.6+ のセキュリティ制限を自動解除。
- AttributeError (with_character): AlignedSegment クラスのメソッド不足を解消。
- FFmpeg DLL Path: 環境変数 FFMPEG_SHARED_PATH からの動的ロードに対応。
