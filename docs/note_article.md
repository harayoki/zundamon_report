# ReportVox の処理フローまとめ（note.com 下書き）

本稿は note.com での紹介記事向けに、ReportVox が音声から動画までを自動生成する流れを整理したものです。入力音声が何で、どのモジュール・LLMがどの順番で処理するかを俯瞰できるようにしています。動画処理は末尾に専用セクションを設けました。

## 入力と前提条件
- **入力ファイル**: 音声（wav/mp3/m4a など）もしくは動画。実行時に `work/<run_id>` 配下へコピーされ、必要に応じて WAV へ正規化されます。【F:reportvox/pipeline.py†L634-L708】
- **必須ツール**: ffmpeg を起動前に検査し、パスが未指定でも自動解決します。【F:reportvox/pipeline.py†L654-L658】
- **run_id の決定と再開**: 実行ごとに `work/yyyymmdd-hhmmss` を生成し、`--resume` 時は既存ディレクトリを再利用します。【F:reportvox/pipeline.py†L1181-L1199】
- **LLM バックエンド**: OpenAI/ローカルLLMを選択でき、ローカル指定時は `LOCAL_LLM_BASE_URL` を自動で組み立てます。【F:reportvox/pipeline.py†L1188-L1193】

## 全体フロー（音声〜字幕まで）
1. **入力準備**
   - 既存の出力上書き確認、作業用ディレクトリへのコピー、WAV 正規化を実施します。【F:reportvox/pipeline.py†L664-L707】

2. **文字起こし（Whisper）**
   - `transcribe.transcribe_audio` で文字起こしし、キャッシュがあれば再利用。必要に応じて LLM 校正（手動/自動）を挟みます。【F:reportvox/pipeline.py†L710-L791】

3. **話者分離（pyannote.audio）**
   - Hugging Face トークンを使って diarization を実行し、キャッシュ/再開を考慮。TorchCodec 警告も検知してログします。【F:reportvox/pipeline.py†L794-L859】

4. **口調変換と定型句挿入**
   - 話者アライン後にキャラクター設定を適用し、LLM で口調変換や改行調整を行います（プロンプトはログへ記録可能）。【F:reportvox/pipeline.py†L862-L930】

5. **音声合成（VOICEVOX Engine）**
   - スタイリング済みセグメントを VOICEVOX へ送り、合成進捗を表示しながら WAV を生成します。【F:reportvox/pipeline.py†L933-L971】

6. **結合・タイムライン配置**
   - 各セグメントの目標尺を計算し、`audio.join_wavs` で一つの WAV にまとめて配置情報を保存します。【F:reportvox/pipeline.py†L974-L1013】

7. **仕上げ（字幕・音声書き出し）**
   - 字幕データを音声に合わせて整列し、SRT/ASS を出力。mp3 変換もここで実施します。【F:reportvox/pipeline.py†L1016-L1085】

## 動画処理セクション
- **字幕焼き込み用 ASS 生成**: 動画出力を選んだ場合、キャラクター色やフォント設定を反映した ASS を `work/<run_id>` に作成します。【F:reportvox/pipeline.py†L1091-L1106】
- **画像オーバーレイ**: オプションで指定した画像を動画尺に合わせて配置し、拡大率や座標も指定可能です。【F:reportvox/pipeline.py†L1108-L1115】
- **レンダリング**: ffmpeg を用いて mp4（不透明）/mov（アルファ付き）を生成し、字幕や画像オーバーレイを合成します。【F:reportvox/pipeline.py†L1117-L1158】
- **作業ディレクトリ整理**: `--keep-work` なしの場合は動画生成後に中間成果物を自動削除します。【F:reportvox/pipeline.py†L1160-L1166】

## 処理のポイント
- Whisper や pyannote の結果は `out/` 配下にキャッシュされ、同じ音声を再利用する際の高速化に寄与します。【F:reportvox/pipeline.py†L725-L859】
- 口調変換・改行挿入で用いた LLM プロンプトは `work/<run_id>/prompt_*.log` に保存されるため、note 記事でプロンプト設計を紹介する際の素材にもできます。【F:reportvox/pipeline.py†L893-L928】
- すべてのステップは `_PIPELINE_STEPS` に明示されており、途中再開時もステップ名/番号を指定できます。【F:reportvox/pipeline.py†L1170-L1199】

この流れをベースに、note 記事では「入力 → 文字起こし → 話者分離 → 口調変換 → 合成 → 字幕・動画」の順でスクリーンショットやログを交えつつ説明すると、読者が実装イメージを掴みやすくなります。
