# ReportVox

ReportVox は、音声ファイルを「文字起こし → 話者分離 → 口調変換 → VOICEVOX での音声生成 → WAV/MP3 出力」まで自動化する CLI ツールです。NotebookLM などが生成する m4a を扱うため ffmpeg は必須で、`ffmpeg -version` が通る状態にしてください。動画ファイル（mp4 など）の入力も許可していますが、音声トラックがあることが前提で、最初に ffmpeg で WAV に正規化してから処理します。

## ワークフロー
1. 入力音声を `work/<run_id>/` にコピー（run_id は重複しないよう自動採番）
2. ffmpeg で WAV に正規化（以降は WAV のみ使用）
3. Whisper で文字起こし（セグメント JSON を保存）
4. pyannote.audio で話者分離（`--speakers` が `auto/2` のときに実行）
5. Whisper セグメントと突合して話者ラベルを付与
6. 1 人相当なら speaker1 のキャラクターで全編、2 人なら発話量が多い話者を speaker1 としてキャラ割当
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
  - pyannote.audio + TorchCodec を Windows で使う場合は「shared」と書かれた共有DLL版を選択してください。`bin` フォルダーを PATH か DLL 検索パスに入れないと libtorchcodec_core*.dll が見つからないエラーになります。
- macOS: `brew install ffmpeg`
- Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
- インストール後、`ffmpeg -version` が成功することを確認してください。

## 依存サービス
- VOICEVOX Engine（ローカル起動を想定）
  - アプリ版ではなくエンジンを事前にダウンロードし、ポートを指定して起動しておいてください。
  - 例: `VOICEVOX_ENGINE_HOME=/path/to/engine ./run --host 127.0.0.1 --port 50021`
- pyannote.audio の実行には Hugging Face Token が必要な場合があります。
  - 環境変数 `PYANNOTE_TOKEN` または CLI 引数 `--hf-token` で指定してください。
  - 事前に以下の手順でアクセス権を付与してください:
    1. https://huggingface.co/pyannote/speaker-diarization と https://huggingface.co/pyannote/segmentation で利用規約に同意する
    2. https://huggingface.co/settings/tokens でアクセストークンを作成する
    3. 作成したトークンを `PYANNOTE_TOKEN` 環境変数または `--hf-token` で渡す
  - 環境変数の設定例:
    - Windows PowerShell: `setx PYANNOTE_TOKEN "hf_xxx"`
    - Windows コマンドプロンプト（DOS 窓）: `set PYANNOTE_TOKEN=hf_xxx`
    - macOS (csh): `setenv PYANNOTE_TOKEN hf_xxx`
  - 上記が未設定のまま実行すると、最初に「文字起こしまでで終了する」旨を対話的に確認します。OK を選んだ場合は文字起こしのみ行い、`work/<run_id>/` に結果を残した上で終了します。
  - `--speakers 1` を指定した場合は話者分離をスキップするため、Token 無しでも最後まで進みます。

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

### 主なオプション
- `--speakers {auto,1,2}`: 話者数の扱い。`auto` では pyannote.audio で 1/2 人を判定し、全発話の 93% 以上を占める話者がいれば単一話者として処理します。
- `--speaker1 / --speaker2`: キャラクター ID。デフォルトは `zundamon` と `metan`。発話量が多い話者が `speaker1` に割り当てられます。
- `--model`: Whisper のモデルサイズ（例: `base`, `small`, `medium`, `large`）。
- `--voicevox-url`: VOICEVOX Engine のベース URL。
- `--llm`: 口調変換バックエンドの選択。`none`（変換なし）、`openai`、`local` から指定できます（デフォルトは `none`）。
- `--mp3 / --bitrate`: MP3 も生成する場合のフラグとビットレート。
- `--ffmpeg-path`: ffmpeg 実行ファイルへのパス（デフォルトは `ffmpeg`。環境変数の PATH に無い場合に指定）。
- `--keep-work`: ワークディレクトリ `work/<run_id>/` を削除せずに残します。
- `--resume <run_id>`: 中断/完了済みのワークディレクトリを再利用して処理を再開します。`input.*` や `transcript.json`、`diarization.json`、`stylized.json` が既存の場合は再利用し、欠けている工程のみ実行します。
- `--hf-token`: pyannote.audio 用の Hugging Face Token を明示指定（環境変数 `PYANNOTE_TOKEN` でも可）。
- `--zunda-senior-job` / `--zunda-junior-job`: ずんだもんの自己紹介（憧れの職業/現在の役割）を挿入します。

## characters の追加方法
`characters/<id>/` ディレクトリを作り、`meta.yaml` と `examples.json` を配置します。`meta.yaml` では VOICEVOX の `speaker_id` を設定し、口調情報や定型句を登録してください。`examples.json` には短文例を配列で追加します。

## 出力
- 常に: `out/<入力ファイル名>_report.wav`
- `--mp3` 指定時: `out/<入力ファイル名>_report.mp3`

各ステップの中間生成物は `work/<run_id>/` に保存されます。`seg_*.wav` には VOICEVOX で生成した発話ごとの音声、`stylized.json` にはキャラクターや挿入フレーズを反映したセグメントが含まれます。

## 中断からの再開
`work/<run_id>/` に残っている中間生成物を利用して、途中から処理を続行できます。

- `--resume <run_id>` を指定すると、既存の `transcript.json`/`diarization.json`/`stylized.json`/`seg_*.wav` を再利用し、足りない工程だけ再実行します。
- 失敗時や `--keep-work` を付けて終了した場合、`work/` に残った run_id を指定してください。
- run_id は実行開始時に `[reportvox] run id: <run_id>` という形で表示されます。

## 注意事項
- VOICEVOX が起動していない場合、合成でエラーとなります。
- ffmpeg が無い場合は実行開始時にエラーとなります。
- LLM 口調変換は差し替えやすい構造ですが、デフォルトではスキップされます。

## トラブルシューティング
### pyannote.audio の話者分離で `torchcodec` が DLL を見つけられない/TypeError が出る
```
[reportvox +00:00] diarizing speakers (auto)...
torchcodec is not installed correctly so built-in audio decoding will fail. Solutions are:
...
TypeError: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'
```

上記のようなログが出て話者分離に失敗する場合は、以下を順に確認してください（特に Windows 環境で libtorchcodec_core*.dll が見つからない場合の定番対処です）。

1. **pyannote.audio を 3 系以降に更新する。** 3 系では `use_auth_token` が廃止されていますが、アプリ側で自動的に対応します。古い 2 系を使っている場合はアップグレードしてください。
   ```bash
   pip install -U "pyannote.audio>=3.0"
   ```
2. **PyTorch と torchcodec の互換バージョンを使う。** PyTorch 2.8.0 を利用する場合は torchcodec 0.6/0.7 系が必要です。例:
   ```bash
   pip uninstall -y torchcodec
   pip install "torchcodec==0.7.*"
   ```
   互換表: https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec
3. **FFmpeg は共有DLL版（shared build）を導入する。** Gyan のビルドなら「shared」と書かれたものを選び、`bin` フォルダーを PATH か DLL 検索パスに追加してください。PATH だけで足りない場合は、Python 実行前に次のように DLL 探索パスへ追加します（例は PowerShell で実行するスクリプトに追記する形）:
   ```python
   import os
   os.add_dll_directory(r"F:\Program Files\ffmpeg-shared\bin")
   ```
4. **依存DLLの不足を確認する。** VC++ 再頒布可能パッケージなどが欠けていると libtorchcodec_core*.dll のロードに失敗します。Dependencies で DLL を開くと不足 DLL が一覧されます。
5. **緊急回避として TorchCodec を使わない経路を試す。** `soundfile` / `librosa` で WAV を読み込み、`{"waveform": tensor, "sample_rate": int}` を `Pipeline.from_pretrained()` に渡せば TorchCodec を経由せずに実行できます。

上記を順に試しても解消しない場合は、暫定措置として `--speakers 1` で話者分離をスキップできます。
