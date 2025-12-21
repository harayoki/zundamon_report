# ReportVox

ReportVox は、音声ファイルを「文字起こし → 話者分離 → 口調変換 → VOICEVOX での音声生成 → WAV/MP3 出力」まで自動化する CLI ツールです。 NotebookLM などが生成する m4a を扱うため ffmpeg は必須です。

## ⚠️ 依存関係に関する重要な注意事項
現在、Windows 環境において pyannote.audio と PyTorch 周辺ライブラリの間に深刻な互換性問題が複数確認されています。 本ツールではこれらを回避するためのパッチを内蔵していますが、以下の環境構築が必須となります。

### 1. NumPy のバージョン固定
NumPy 2.0 以降では話者分離がクラッシュするため、1.26.x 系を強制してください。
```bash
pip install "numpy<2"
```

### 2. FFmpeg Shared 版の導入 (Windows 必須)
話者分離（torchcodec）の動作には、Shared 版（共有DLL版）の FFmpeg が必須です。
- 取得元: gyan.dev から "release full shared" を選択。
- 設定: 解凍した bin フォルダのパスを環境変数 FFMPEG_SHARED_PATH に設定してください。
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

1. 以下のリポジトリですべて利用規約に同意（Accept）してください：
   - pyannote/speaker-diarization-3.1
   - pyannote/segmentation-3.0
2. Hugging Face Settings でトークンを作成。
3. トークンを環境変数 PYANNOTE_TOKEN または --hf-token で渡してください。

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

# 動画ファイルも入力可、MP3も生成
python -m reportvox input.mp4 --speakers auto --mp3
```

## 主なオプション
- --speakers {auto,1,2}: 話者数の扱い。
- --model: Whisper のモデルサイズ。
- --voicevox-url: VOICEVOX Engine のベース URL。
- --speed-scale: VOICEVOX での読み上げ速度（デフォルト 1.0）。
- --resume <run_id>: 中断した工程から再開。
- --hf-token: Hugging Face Token。

## トラブルシューティング（パッチ対応済み事項）
以下の問題は diarize.py 内のモンキーパッチにより自動修正されます：

- TypeError (use_auth_token): huggingface_hub の引数名変更エラーを修正。
- WeightsUnpickler error: PyTorch 2.6+ のセキュリティ制限を自動解除。
- AttributeError (with_character): AlignedSegment クラスのメソッド不足を解消。
- FFmpeg DLL Path: 環境変数 FFMPEG_SHARED_PATH からの動的ロードに対応。
