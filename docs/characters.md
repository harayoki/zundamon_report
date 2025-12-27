# キャラクター追加・口癖設定ガイド

> **メモ:** 本リポジトリに含まれるキャラ設定は暫定的な内容です。利用するプロジェクトや好みに合わせて、各自でカスタマイズしてご利用ください。

ReportVox に新しい話者（キャラクター）を追加したり、口癖を調整する手順をまとめています。CLI から参照できるキャラクターは `characters/<id>/` ディレクトリにあるメタデータで定義します。

## 基本構成
各キャラクターは以下の 2 ファイルで構成します。

- `characters/<id>/meta.yaml`: 必須。キャラクターの名前や VOICEVOX の `speaker_id`、口癖などのメタ情報。
- `characters/<id>/examples.json`: 任意。口調の例文を配列で列挙（外部の口調変換バックエンドを追加する際などのヒントに利用）。

> 例: `characters/zundamon/meta.yaml` と `characters/zundamon/examples.json`

## meta.yaml の書式
```yaml
id: zundamon              # キャラクターID。ディレクトリ名と一致させることを推奨
display_name: ずんだもん   # 表示用の名前
voicevox:
  speaker_id: 3           # VOICEVOX の話者ID。合成には必須
style:
  first_person: ぼく      # 一人称（任意）
  endings: ["なのだ"]      # 語尾リスト（任意）
  role: 解説役            # 役割メモ（任意）
phrases:
  idea_intro: ["ここでぼくの出番なのだ"]  # 最初の発話で使う定型句候補
```

### 設定項目の補足
- `voicevox.speaker_id` を省略または空にすると合成時にエラーになります。VOICEVOX Engine の UI や API で ID を確認して設定してください。
- `phrases.idea_intro` に指定した文は、そのキャラクターが初めて話すセグメントにランダムで 1 文挿入されます（`reportvox/style_convert.py` のヒューリスティック）。他のキーは現状使用されませんが、将来拡張用に保持しておけます。
- `style.endings` や `style.first_person` は現状強制変換には使われませんが、LLM バックエンドを有効化したときのヒントとして保持されています。

## 口癖・定型句の調整
1. 既存キャラクターを編集する場合は `characters/<id>/meta.yaml` 内の `phrases.idea_intro` を増減してください。`phrases.default` に書いた口癖リストは **`--style-with-llm` オプションで LLM バックエンドを有効にした場合のみ** 口調変換のプロンプトに渡されます（デフォルトで有効、無効にする場合は `--no-style-with-llm` を指定）。`phrases.idea_intro` に指定した定型句は、LLM を使わない場合に初回のみヒューリスティックで挿入されます。
2. 1 文あたり 12 文字以上の発話のみが定型句挿入の対象です。短い文では口癖が挿入されない点に注意してください。
3. 定型句は 1 回だけ挿入されます。複数回の挿入を避けたい場合、候補を短くしても自動的に 1 回で止まります。

## 新しいキャラクターを追加する
1. `characters/<id>/` を作成します（例: `characters/ao/`）。
2. 上記フォーマットで `meta.yaml` を用意し、必ず `voicevox.speaker_id` を設定します。
3. 任意で `examples.json` を追加し、口調例を配列で記載します。
4. CLI から次のように指定すると新キャラクターを利用できます。
   ```bash
   python -m reportvox input.wav --speakers 2 --speaker1 ao --speaker2 metan
   ```

## よくあるチェックポイント
- 文字コードは UTF-8 を使用してください。
- YAML のインデントはスペース 2 個を推奨します。
- ディレクトリ名と `id` を揃えておくと混乱が減ります。
- VOICEVOX の話者 ID を間違えると別の声になるか、エラーになります。テスト音声で確認してください。
