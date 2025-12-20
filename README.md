# ReportVox

ReportVox は、音声ファイルを「文字起こし → 話者分離 → 口調変換 → VOICEVOX での音声生成 → WAV/MP3 出力」まで自動化する CLI ツールです。NotebookLM などが生成する m4a を扱うため ffmpeg は必須で、`ffmpeg -version` が通る状態にしてください。動画ファイル（mp4 など）の入力も許可していますが、音声トラックがあることが前提で、最初に ffmpeg で WAV に正規化してから処理します。

## ワークフロー
1. 入力音声を `work/<run_id>/` にコピー
2. ffmpeg で WAV に正規化（以降は WAV のみ使用）
3. Whisper で文字起こし（セグメント JSON 保存）
4. pyannote.audio で話者分離（1/2人自動判定対応）
5. Whisper セグメントと突合して話者ラベルを付与
6. 1人相当ならずんだもん全編、2人なら発話量の多い方を speaker1 としてキャラ割当
7. （オプション）口調変換、定型句の自動挿入
8. VOICEVOX Engine HTTP API で行ごとに合成
9. wav を結合して `out/<stem>_report.wav` を生成
10. `--mp3` 指定時に MP3 を追加生成（ffmpeg 使用）

## インストール
Python 3.11+ を想定しています。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### ffmpeg 導入例
- Windows: https://www.gyan.dev/ffmpeg/builds/ から取得し、`ffmpeg/bin` を PATH に追加
- macOS: `brew install ffmpeg`
- Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
- インストール後、`ffmpeg -version` が成功することを確認してください。

## 依存サービス
- VOICEVOX Engine（ローカル起動を想定）
  - 例: `VOICEVOX_ENGINE_HOME=/path/to/engine ./run --host 127.0.0.1 --port 50021`
- pyannote.audio の実行には Hugging Face Token が必要な場合があります。
  - 環境変数 `PYANNOTE_TOKEN` または CLI 引数 `--hf-token` で指定してください。

## 使い方
```bash
# 自動話者判定、デフォルト（ずんだもん+めたん）
python -m reportvox input.wav

# ずんだもんの自己紹介（憧れ/現在の役割を指定）
python -m reportvox input.m4a --zunda-senior-job エンジニア --zunda-junior-job 自動化係

# 話者を強制 1 人（ずんだもんのみ）
python -m reportvox input.wav --speakers 1

# 話者 2 人を強制し、キャラ指定
python -m reportvox input.wav --speakers 2 --speaker1 zundamon --speaker2 metan

# MP3 も生成
python -m reportvox input.wav --mp3 --bitrate 192k

# 中断したワークフローを再開（work/<run_id> を指定）
python -m reportvox input.wav --resume 20240101-120000

# 動画ファイル（音声トラックあり）も入力可
python -m reportvox input.mp4 --speakers auto --mp3

# VOICEVOX の URL を指定
python -m reportvox input.wav --voicevox-url http://127.0.0.1:50021
```

## characters の追加方法
`characters/<id>/` ディレクトリを作り、`meta.yaml` と `examples.json` を配置します。`meta.yaml` では VOICEVOX の `speaker_id` を設定し、口調情報や定型句を登録してください。`examples.json` には短文例を配列で追加します。

## 出力
- 常に: `out/<入力ファイル名>_report.wav`
- `--mp3` 指定時: `out/<入力ファイル名>_report.mp3`

## 中断からの再開
`work/<run_id>/` に残っている中間生成物を利用して、途中から処理を続行できます。

- `--resume <run_id>` を指定すると、既存の `transcript.json`/`diarization.json`/`stylized.json`/`seg_*.wav` を再利用し、足りない工程だけ再実行します。
- 失敗時や `--keep-work` を付けて終了した場合、`work/` に残った run_id を指定してください。
- run_id は実行開始時に `[reportvox] run id: <run_id>` という形で表示されます。

## 注意事項
- VOICEVOX が起動していない場合、合成でエラーとなります。
- ffmpeg が無い場合は実行開始時にエラーとなります。
- LLM 口調変換は差し替えやすい構造ですが、デフォルトではスキップされます。
