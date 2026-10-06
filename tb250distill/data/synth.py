"""決定的 seed による raw item 生成（常駐エージェントの decision engine 向け）。

構成:
  1. 語彙バンク（ROLES: 候補文言を train / para / unseen の 3 層で持つ、スロット語彙、filler）
  2. シナリオ生成 make_scenario(category, lang, seed): 同じ seed なら同じ Scn（状況・正解・候補役割）
  3. 描画 render(scn, variant): Scn を文字列にする。robust variant は同じ Scn を別の描画オプションで再描画する
  4. データセット組み立て build_dataset / CLI

候補文字列プール:
  - train 層: train/val/test の通常 item と robust の perm/ctx_paraphrase/irrelevant_ctx/ambiguous が使う
  - para 層 : cand_paraphrase variant だけが使う（train 層の同義語辞書。役割ごとに手書き）
  - unseen 層: unseen_cand variant だけが使う。train 層・para 層と文字列が重ならない（tests/test_synth.py で検証）
gold はテンプレート上明確に決まる場合のみ設定し、曖昧カテゴリ・矛盾する証拠・好み衝突は NULL。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable

CATEGORY_RATIOS = {
    "nli": 20, "intent": 20, "state_action": 20, "ranking": 15,
    "sentiment": 10, "ambiguous": 10, "agent_gate": 5,
}
LANG_RATIOS = {"ja": 60, "en": 40}
K_DIST = {2: 0.10, 3: 0.35, 4: 0.30, 5: 0.25}
VARIANT_KINDS = ("perm", "cand_paraphrase", "ctx_paraphrase", "irrelevant_ctx", "ambiguous", "unseen_cand")

# context 長の目安（ja は文字数、en は語数）。teacher の prompt 上限 256 tokens に収まる範囲に抑える。
CTX_LEN_RANGE = {"ja": (30, 200), "en": (25, 120)}

# ---------------------------------------------------------------------------
# 1. 候補役割の文言（train / para / unseen）
# ---------------------------------------------------------------------------


def _r(ja: tuple, en: tuple) -> dict:
    """ja=(train, para, unseen), en=(train, para, unseen)。"""
    return {
        "ja": {"train": list(ja[0]), "para": list(ja[1]), "unseen": list(ja[2])},
        "en": {"train": list(en[0]), "para": list(en[1]), "unseen": list(en[2])},
    }


ROLES: dict[str, dict] = {
    "continue": _r(
        (["続行する", "そのまま続ける", "処理を継続する", "作業を進める", "このまま実行を続ける"],
         ["継続する", "引き続き進める", "やり続ける"],
         ["今の手順で先へ進める", "中断せずに最後まで走らせる", "現在の作業を維持する"]),
        (["continue", "keep going", "continue with the current task", "proceed as planned", "carry on"],
         ["go on", "keep it running", "press on"],
         ["let the current run finish", "leave the job running to completion", "stay the course"])),
    "stop": _r(
        (["停止する", "中止する", "処理を止める", "作業を打ち切る", "ここで終了する"],
         ["やめる", "中断する", "ストップする"],
         ["実行を取りやめる", "以降の処理を行わない", "いったん手を止めて終わりにする"]),
        (["stop", "abort the task", "halt here", "cancel the current job", "stop and exit"],
         ["quit", "terminate it", "call it off"],
         ["shut the job down", "do not run any further steps", "end the run now"])),
    "answer": _r(
        (["そのまま答える", "手元の情報で回答する", "すぐに回答する", "今ある知識で答える"],
         ["直接答える", "即答する", "持っている情報で返答する"],
         ["調べずにこの場で返事をする", "既知の内容だけで応答する", "追加確認なしで説明する"]),
        (["answer directly", "answer from what we know", "reply right away", "respond with the current information"],
         ["give the answer now", "respond immediately", "just answer"],
         ["reply without looking anything up", "give an answer based on existing knowledge only", "explain it on the spot"])),
    "search": _r(
        (["先に検索する", "ウェブで調べてから答える", "情報を調べる", "検索してから回答する", "まず外部情報を確認する"],
         ["ネットで調べる", "先に調査する", "調べ直す"],
         ["回答の前に最新情報を探す", "オンラインで裏取りしてから返す", "外部ソースを当たってみる"]),
        (["search the web first", "look it up before answering", "do a search", "check external sources first"],
         ["google it first", "research it first", "run a web lookup"],
         ["verify online before replying", "find up-to-date information first", "consult outside sources before responding"])),
    "ask": _r(
        (["ユーザーに確認する", "本人に質問する", "ユーザーへ聞き返す", "確認を取る", "ユーザーの意向を尋ねる"],
         ["本人に聞く", "問い合わせる", "確かめてもらう"],
         ["実行前に相手の了承をもらう", "どうしたいか相手に尋ねる", "判断をユーザーに委ねる"]),
        (["ask the user", "check with the user", "ask a clarifying question", "confirm with the user first", "ask for the user's preference"],
         ["query the user", "ask them first", "get the user's confirmation"],
         ["hand the decision back to the user", "request approval before doing anything", "find out what the user wants"])),
    "act": _r(
        (["自分で実行する", "そのまま実行する", "確認せずに進める", "自律的に処理する", "すぐに実行に移す"],
         ["自力で対応する", "勝手に進める", "直ちに実施する"],
         ["相談なしで作業を完了させる", "自分の判断で手を動かす", "問い合わせず先に実施する"]),
        (["act on it", "just do it", "do it without asking", "handle it autonomously", "execute immediately"],
         ["take care of it yourself", "go ahead and do it", "carry it out right now"],
         ["complete it on your own judgment", "make the change without checking in", "take the action first and report later"])),
    "remember": _r(
        (["記憶に保存する", "覚えておく", "メモリに残す", "長期記憶に記録する", "後で使えるよう保存する"],
         ["記録しておく", "保管する", "メモに残す"],
         ["今後の参照用に書き留める", "この内容を忘れないよう保持する", "永続メモリへ追加する"]),
        (["save to memory", "remember it", "store it in long-term memory", "keep a note of this", "record it for later"],
         ["memorize it", "write it down", "commit it to memory"],
         ["retain this for future reference", "persist this fact", "add it to the memory store"])),
    "discard": _r(
        (["破棄する", "保存せず捨てる", "覚えない", "記録しない", "忘れてよい"],
         ["捨てる", "無視して消す", "残さない"],
         ["この内容は保持せず流す", "メモリには加えない", "一時的な情報として扱い消去する"]),
        (["discard it", "do not store this", "forget it", "don't save it", "drop it"],
         ["throw it away", "ignore and delete it", "leave it unrecorded"],
         ["treat it as transient and let it go", "keep it out of memory", "do not retain this"])),
    "retrieve": _r(
        (["記憶を検索する", "メモリから関連情報を取り出す", "過去の記録を参照する", "記憶を引き出す", "関連する記憶を呼び出す"],
         ["過去のメモを調べる", "メモリを引く", "保存済みの情報を確認する"],
         ["以前のやり取りを掘り起こす", "蓄積した知識から関連項目を探す", "履歴の中から該当する内容を探す"]),
        (["retrieve from memory", "look up related memories", "check past records", "recall relevant memories", "query the memory store"],
         ["pull up stored notes", "search memory", "consult saved memories"],
         ["dig through earlier interactions", "find matching entries in the knowledge store", "look in the history for the relevant part"])),
    "skip": _r(
        (["記憶の参照は省略する", "メモリは見ない", "検索せず進む", "参照をスキップする", "記憶に頼らず応答する"],
         ["メモリ確認を飛ばす", "参照しない", "過去記録を使わない"],
         ["履歴の照会は行わない", "保存済み情報には触れない", "今回は蓄積データを使わず対応する"]),
        (["skip memory lookup", "don't check memory", "proceed without retrieval", "skip the recall step", "respond without consulting memory"],
         ["bypass the memory step", "leave memory untouched", "no retrieval this time"],
         ["do not consult the stored history", "go ahead without querying past notes", "leave the knowledge store alone"])),
    "use_tool": _r(
        (["{tool}を使う", "{tool}を呼び出す", "{tool}で処理する"],
         ["{tool}を利用する", "{tool}に任せる", "{tool}で対応する"],
         ["{tool}へ処理を回す", "{tool}の機能を使って進める", "{tool}経由で実行する"]),
        (["use the {tool}", "call the {tool}", "run the {tool}"],
         ["make use of the {tool}", "invoke the {tool}", "rely on the {tool}"],
         ["route this through the {tool}", "hand it to the {tool}", "go with the {tool}"])),
    "no_tool": _r(
        (["ツールは使わない", "ツールなしで答える", "ツール呼び出しは不要"],
         ["ツールに頼らない", "道具を使わず対応する", "ツールを呼ばない"],
         ["外部機能には触れず処理する", "何も呼び出さずに終える", "そのまま手元で片付ける"]),
        (["use no tool", "answer without a tool", "no tool call is needed"],
         ["skip the tools", "don't invoke any tool", "avoid tools"],
         ["handle it without calling anything external", "finish without touching any function", "settle it locally, no calls"])),
    "safe_action": _r(
        (["安全な方法で実行する", "影響の小さい操作だけ行う", "安全側の手順で進める", "元に戻せる操作を選ぶ", "慎重に実行する"],
         ["低リスクな手順で進める", "可逆な操作のみ実施する", "安全策を取って実行する"],
         ["ロールバック可能な手段で片付ける", "副作用のない範囲にとどめて動く", "危険の少ないやり方で処理する"]),
        (["take the safe action", "do only the low-impact operation", "proceed the safe way", "choose a reversible step", "execute carefully"],
         ["go with the low-risk option", "do the undoable thing only", "play it safe and proceed"],
         ["pick the rollback-friendly approach", "limit actions to side-effect-free ones", "handle it in the least risky manner"])),
    "defer": _r(
        (["後回しにする", "判断を保留する", "あとで対応する", "いったん見送る", "保留にする"],
         ["先送りする", "様子を見る", "一旦置いておく"],
         ["今は手を出さず後で判断する", "結論を持ち越す", "もう少し状況が固まるまで待つ"]),
        (["defer it", "hold off on the decision", "handle it later", "postpone for now", "put it on hold"],
         ["push it back", "wait and see", "set it aside for now"],
         ["revisit once the situation settles", "carry the decision over", "wait until more is known"])),
    "local": _r(
        (["ローカルの小型モデルで処理する", "手元のモデルで答える", "小さいモデルで済ませる", "オンデバイスのモデルを使う"],
         ["軽量モデルで対応する", "端末内のモデルに任せる", "ローカルモデルを使う"],
         ["この場の小さなモデルで片付ける", "外に出さず内蔵モデルだけで返す", "省リソースのモデルで回答する"]),
        (["use the local model", "answer with the small on-device model", "handle it with the lightweight model", "keep it on the local model"],
         ["stay on the local model", "let the small model take it", "use the on-device model"],
         ["settle it with the built-in small model", "process locally without calling out", "rely on the compact model"])),
    "larger": _r(
        (["大型モデルに回す", "より大きいモデルへ委ねる", "上位モデルに問い合わせる", "高性能モデルで処理する"],
         ["大きなモデルに任せる", "強力なモデルを呼ぶ", "大規模モデルで対応する"],
         ["重い推論は大きなモデルへ引き継ぐ", "上位の頭脳に判断を仰ぐ", "より賢いモデルへエスカレーションする"]),
        (["escalate to the larger model", "hand it to the bigger model", "ask the stronger model", "use the high-capacity model"],
         ["send it to the large model", "defer to the more powerful model", "call the big model"],
         ["pass the hard part to a more capable model", "consult the heavyweight model", "bump this up to the larger LLM"])),
    "speak": _r(
        (["ユーザーに話しかける", "通知して知らせる", "声をかけて伝える", "今すぐ報告する"],
         ["ユーザーへ知らせる", "発話して伝える", "割り込んで報告する"],
         ["こちらから切り出して伝える", "画面に出して注意を促す", "直ちに一言伝える"]),
        (["speak up now", "notify the user", "tell the user right away", "send an alert"],
         ["let the user know", "say something now", "raise it with the user"],
         ["bring it up proactively", "pop up a notice for the user", "flag it to them immediately"])),
    "silent": _r(
        (["黙っておく", "何も言わない", "通知しない", "邪魔をしない"],
         ["声をかけない", "静かに見守る", "伝えずにおく"],
         ["発言せず様子を見る", "今は話しかけない", "ユーザーの集中を妨げない"]),
        (["stay silent", "say nothing", "don't notify", "don't interrupt"],
         ["keep quiet", "hold your tongue", "leave the user alone"],
         ["refrain from speaking up", "let it pass unmentioned", "avoid disturbing the user"])),
    "do": _r(
        (["{x}を実行する", "{x}を行う", "{x}に取りかかる"],
         ["{x}を処理する", "{x}を進める", "{x}をやる"],
         ["{x}の作業へ移る", "{x}を片付ける", "{x}に着手する"]),
        (["do the {x}", "work on the {x}", "start on the {x}"],
         ["take care of the {x}", "carry out the {x}", "go ahead with the {x}"],
         ["move on to the {x}", "get the {x} done", "tackle the {x}"])),
    # NLI
    "nli_entail": _r(
        (["前提から導ける", "文脈に含意される", "文脈から言える", "正しいと言える"],
         ["前提に沿っている", "文脈から確実に言える", "前提から必ず成り立つ"],
         ["提示された内容から結論できる", "書かれた事実から言える", "前提を踏まえて真と判断できる"]),
        (["follows from the context", "is entailed", "is supported by the context", "is true given the context"],
         ["can be inferred from the context", "is implied by the text", "must hold given the text"],
         ["is a valid conclusion from what is stated", "is backed up by the passage", "holds true based on the given facts"])),
    "nli_contra": _r(
        (["文脈と矛盾する", "前提と食い違う", "文脈に反する", "誤りである"],
         ["前提と合わない", "文脈と両立しない", "前提に反している"],
         ["提示された内容と衝突する", "書かれた事実と相容れない", "前提に照らして偽と判断できる"]),
        (["contradicts the context", "conflicts with the context", "is contradicted", "is false given the context"],
         ["is inconsistent with the text", "goes against the context", "cannot be true given the text"],
         ["clashes with what is stated", "is refuted by the passage", "is incompatible with the given facts"])),
    "nli_neutral": _r(
        (["判断できない", "文脈からは不明", "どちらとも言えない", "情報が足りない"],
         ["確定できない", "前提だけでは分からない", "言及されていない"],
         ["与えられた内容では決められない", "書かれていないので不明", "根拠が示されていない"]),
        (["cannot be determined", "is unknown from the context", "is neither supported nor contradicted", "there is not enough information"],
         ["is undetermined", "is not stated in the context", "remains unclear"],
         ["is left open by the passage", "cannot be decided from the given facts", "lacks supporting evidence either way"])),
    # sentiment / priority
    "sent_pos": _r(
        (["肯定的", "ポジティブ", "好意的な内容", "満足している"],
         ["前向き", "好評", "好意的"],
         ["喜んでいる様子", "高く評価している", "ご機嫌な反応"]),
        (["positive", "favorable", "satisfied", "a positive sentiment"],
         ["upbeat", "approving", "pleased"],
         ["expresses delight", "speaks highly of it", "sounds happy about it"])),
    "sent_neg": _r(
        (["否定的", "ネガティブ", "不満がある", "批判的な内容"],
         ["後ろ向き", "不評", "不満げ"],
         ["怒っている様子", "低く評価している", "がっかりした反応"]),
        (["negative", "unfavorable", "dissatisfied", "a negative sentiment"],
         ["downbeat", "disapproving", "displeased"],
         ["expresses frustration", "speaks poorly of it", "sounds disappointed"])),
    "sent_neu": _r(
        (["中立", "どちらでもない", "事実を述べているだけ", "感情は読み取れない"],
         ["ニュートラル", "感情が含まれない", "淡々としている"],
         ["評価を含まない記述", "感想なしの報告", "ただの情報共有"]),
        (["neutral", "neither positive nor negative", "just stating facts", "no emotion detected"],
         ["objective", "emotionless", "matter-of-fact"],
         ["a plain description without opinion", "a report with no feeling", "mere information sharing"])),
    "prio_high": _r(
        (["緊急", "最優先で対応", "今すぐ対応が必要", "高優先度"],
         ["至急", "すぐに処理すべき", "優先度が高い"],
         ["即座の対処が求められる", "他を止めてでも扱う", "時間的余裕がない"]),
        (["urgent", "handle with top priority", "needs action right now", "high priority"],
         ["critical", "should be handled immediately", "high on the priority list"],
         ["demands immediate attention", "drop everything for this", "there is no time to spare"])),
    "prio_norm": _r(
        (["通常", "普通の優先度", "通常どおり対応", "中優先度"],
         ["標準", "いつもの優先度", "並の扱い"],
         ["他の作業と同じ順番で処理する", "期日内に普通に対応する", "特別扱いは不要"]),
        (["normal", "regular priority", "handle as usual", "medium priority"],
         ["standard", "routine priority", "ordinary handling"],
         ["process it in the usual order", "deal with it within the normal timeframe", "no special treatment needed"])),
    "prio_low": _r(
        (["低優先度", "後回しでよい", "急ぎではない", "いつでもよい"],
         ["優先度が低い", "余裕があるときで十分", "緊急性なし"],
         ["手が空いたときに見れば足りる", "重要度は低い", "待たせても問題ない"]),
        (["low priority", "can wait", "not urgent", "whenever is fine"],
         ["minor priority", "fine to do when free", "no urgency"],
         ["look at it when there is spare time", "of little importance", "it is fine to leave it waiting"])),
    # intent
    "intent_calendar": _r(
        (["予定の管理", "カレンダー操作", "スケジュール登録"], ["予定の確認や変更", "日程の調整"], ["予定表への入力・照会", "日付に関する予定の扱い"]),
        (["calendar management", "scheduling an event", "calendar lookup"], ["handling the schedule", "calendar operation"], ["dealing with dates on the agenda", "events and appointments handling"])),
    "intent_reminder": _r(
        (["リマインダー設定", "通知の予約", "あとで知らせる設定"], ["リマインド登録", "忘れ防止の通知"], ["指定時刻にアラートを鳴らす", "時間が来たら呼びかける設定"]),
        (["set a reminder", "schedule a notification", "remind me later"], ["create an alert", "timed nudge"], ["ring an alarm at a set time", "prompt me when the time comes"])),
    "intent_memo": _r(
        (["メモの保存", "ノートへの記録", "内容を覚えさせる"], ["メモ書き", "記録の追加"], ["情報を手帳に残す", "書き留め依頼"]),
        (["save a note", "write to the notebook", "store this information"], ["jot something down", "add a record"], ["put the info in the notepad", "a request to write this down"])),
    "intent_weather": _r(
        (["天気の問い合わせ", "気象情報の取得", "天気予報の確認"], ["天候の質問", "気温や降水の確認"], ["空模様についての質問", "外の様子の問い合わせ"]),
        (["weather inquiry", "get the forecast", "check the weather"], ["weather question", "temperature and rain check"], ["asking about the sky outside", "meteorological query"])),
    "intent_chat": _r(
        (["雑談", "世間話", "ただの会話"], ["おしゃべり", "気軽な会話"], ["特に用件のない会話", "気分を話しているだけ"]),
        (["small talk", "casual chat", "just conversation"], ["chitchat", "friendly banter"], ["conversation with no particular goal", "just sharing a mood"])),
    "intent_files": _r(
        (["ファイル検索", "資料を探す", "文書を開く"], ["ファイルの場所確認", "書類の検索"], ["保存済み文書の呼び出し", "手元のデータを探す依頼"]),
        (["file search", "find a document", "open a file"], ["locate a file", "document lookup"], ["pull up a saved document", "request to hunt for local data"])),
    "intent_settings": _r(
        (["設定変更", "端末の設定操作", "表示や音量の調整"], ["設定の切り替え", "環境設定の操作"], ["デバイスの挙動を変える依頼", "システムの調整"]),
        (["change settings", "device configuration", "adjust display or volume"], ["toggle a setting", "preferences operation"], ["a request to alter device behaviour", "system tweak"])),
    "intent_translate": _r(
        (["翻訳", "言葉の訳出", "他言語への変換"], ["訳してもらう", "言語の変換"], ["別の言語に直す依頼", "語句の意味を訳す"]),
        (["translation", "translate the text", "convert to another language"], ["get it translated", "language conversion"], ["a request to render it in another language", "interpret the phrase"])),
}

# ツール種別: 名前も train/unseen で別（候補文字列プール分離のため）
TOOLS: dict[str, dict] = {
    "calc": {"ja": ("計算機", "数式エンジン"), "en": ("calculator", "math engine")},
    "calendar": {"ja": ("カレンダー", "スケジュール帳"), "en": ("calendar", "schedule book")},
    "weather": {"ja": ("天気API", "気象情報サービス"), "en": ("weather API", "forecast service")},
    "files": {"ja": ("ファイルリーダー", "ドキュメントビューア"), "en": ("file reader", "document viewer")},
    "translate": {"ja": ("翻訳ツール", "言語変換サービス"), "en": ("translator", "language converter")},
    "websearch": {"ja": ("検索ツール", "ウェブクローラ"), "en": ("search tool", "web crawler")},
    "mail": {"ja": ("メール取得ツール", "受信箱アクセサ"), "en": ("mail fetcher", "inbox accessor")},
    "tasks": {"ja": ("タスク管理ツール", "やることリスト"), "en": ("task manager", "to-do list")},
}

# ランキングの領域: 名詞は train/unseen で別。属性は 3 段階（0=良い,1=普通,2=悪い）の定性表現
# （候補を Lc=16 token に収めるため数値は使わない）。level 文言は train/para の 2 種。
def _lv(price, time, rating):
    return {"price": price, "time": time, "rating": rating}


RANK_DOMAINS = {
    "lodging": {
        "ja": {"train": ["ビジネスホテル", "駅前の旅館", "ゲストハウス", "シティホテル", "民宿"],
               "unseen": ["カプセルホテル", "コテージ", "リゾートホテル", "ホステル"]},
        "en": {"train": ["business hotel", "guesthouse", "city hotel", "inn", "bed and breakfast"],
               "unseen": ["capsule hotel", "cabin", "resort hotel", "hostel"]},
        "lv": {
            "ja": {"train": _lv(["安い", "価格並", "高い"], ["駅近", "駅まで普通", "駅から遠い"], ["高評価", "評価並", "低評価"]),
                   "para": _lv(["低価格", "中価格", "高価格"], ["駅に近い", "駅まで中距離", "駅から離れる"], ["評判が良い", "評判は普通", "評判が悪い"])},
            "en": {"train": _lv(["cheap", "fair price", "expensive"], ["near station", "fair walk", "far from station"], ["top-rated", "mid rating", "low-rated"]),
                   "para": _lv(["low cost", "moderately priced", "pricey"], ["close to the station", "medium distance", "distant"], ["well reviewed", "so-so reviews", "badly reviewed"])},
        },
        "scene_ja": "宿を探している", "scene_en": "is looking for a place to stay",
    },
    "food": {
        "ja": {"train": ["定食屋", "ラーメン店", "カフェ", "蕎麦屋", "ファミレス"],
               "unseen": ["寿司店", "カレー専門店", "パン屋", "立ち食いうどん"]},
        "en": {"train": ["diner", "ramen shop", "cafe", "soba place", "family restaurant"],
               "unseen": ["sushi bar", "curry house", "bakery", "food stall"]},
        "lv": {
            "ja": {"train": _lv(["安い", "価格並", "高い"], ["待たない", "待ち普通", "待ち長い"], ["高評価", "評価並", "低評価"]),
                   "para": _lv(["低価格", "中価格", "高価格"], ["すぐ入れる", "待ちは中程度", "混んで待つ"], ["評判が良い", "評判は普通", "評判が悪い"])},
            "en": {"train": _lv(["cheap", "fair price", "expensive"], ["short wait", "avg wait", "long wait"], ["top-rated", "mid rating", "low-rated"]),
                   "para": _lv(["low cost", "moderately priced", "pricey"], ["quick seating", "moderate wait", "slow seating"], ["well reviewed", "so-so reviews", "badly reviewed"])},
        },
        "scene_ja": "昼食の店を選んでいる", "scene_en": "is choosing a place for lunch",
    },
    "transport": {
        "ja": {"train": ["急行電車", "高速バス", "タクシー", "在来線", "レンタカー"],
               "unseen": ["新幹線", "フェリー", "ロープウェイ", "自転車シェア"]},
        "en": {"train": ["express train", "highway bus", "taxi", "local train", "rental car"],
               "unseen": ["bullet train", "ferry", "cable car", "bike share"]},
        "lv": {
            "ja": {"train": _lv(["安い", "運賃並", "高い"], ["所要短い", "所要普通", "所要長い"], ["高評価", "評価並", "低評価"]),
                   "para": _lv(["低運賃", "中運賃", "高運賃"], ["早く着く", "所要は中程度", "時間がかかる"], ["評判が良い", "評判は普通", "評判が悪い"])},
            "en": {"train": _lv(["cheap", "fair fare", "expensive"], ["fast", "mid speed", "slow"], ["top-rated", "mid rating", "low-rated"]),
                   "para": _lv(["low fare", "moderate fare", "pricey"], ["quick trip", "medium duration", "lengthy trip"], ["well reviewed", "so-so reviews", "badly reviewed"])},
        },
        "scene_ja": "移動手段を選んでいる", "scene_en": "is choosing how to get there",
    },
}

# スロット語彙（コンテキスト側。候補プールとは無関係）
BK = {
    "ja": {
        "task": ["ビルド", "デプロイ", "データ同期", "バックアップ", "インデックス再構築", "メール送信", "ファイルのアップロード", "モデルの更新", "ログ集計"],
        "err": ["タイムアウト", "接続拒否(ECONNREFUSED)", "権限エラー(403)", "メモリ不足", "ディスク容量不足", "HTTP 503", "DNS解決の失敗"],
        "name": ["田中", "佐藤", "鈴木", "高橋", "伊藤", "Mika", "Ken", "Sora", "山本", "Rin"],
        "day": ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "明日", "あさって", "来週月曜"],
        "day_past": ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜"],
        "ev": ["会議", "歯医者", "打ち合わせ", "面談", "ランチ", "オンライン講座", "定例ミーティング"],
        "todo": ["薬を飲む", "洗濯物を取り込む", "請求書を送る", "母に電話する", "ゴミを出す", "資料を印刷する"],
        "city": ["東京", "大阪", "札幌", "福岡", "名古屋", "仙台", "京都"],
        "file": ["議事録.docx", "請求書_10月.pdf", "report_final.xlsx", "旅行計画.txt", "契約書v3.pdf"],
        "ftype": ["スプレッドシート", "プレゼン資料", "PDF", "ログファイル"],
        "kw": ["予算", "契約", "スケジュール", "障害報告", "見積もり"],
        "hobby": ["料理", "ランニング", "読書", "写真", "ボードゲーム", "ギター"],
        "fact": ["予備の鍵は玄関の下駄箱の中にある", "Wi-Fiのパスワードは変更した", "来月から駐車場が変わる", "犬の予防接種は春にする", "会議室Bは14時まで使えない"],
        "topic": ["新型GPUの発売", "為替相場", "来週のイベント情報", "最新のセキュリティ脆弱性", "ニュース"],
        "word": ["ご査収", "冗長", "ボトルネック", "セレンディピティ", "べき乗"],
        "company": ["A社", "ノースウィンド社", "サンライズ電機", "ミズホ商事"],
        "allergy": ["そば", "卵", "えび", "ピーナッツ"],
        "folder": ["アーカイブ", "仕事", "ダウンロード", "共有フォルダ"],
        "msg": ["遅れます", "資料を確認しました", "了解です", "明日の件、了承です"],
        "what": ["気温", "為替レート", "電車の運行状況", "株価"],
        "phrase_en": ["good morning", "thank you very much", "see you tomorrow", "where is the station"],
        "phrase_ja": ["お先に失礼します", "よろしくお願いします", "ありがとうございました"],
        "greet": ["こんにちは", "やあ、元気？", "おはよう", "ただいま"],
        "os": ["Linux", "macOS", "Windows"],
    },
    "en": {
        "task": ["build", "deployment", "data sync", "backup", "index rebuild", "email send job", "file upload", "model update", "log aggregation"],
        "err": ["timeout", "connection refused (ECONNREFUSED)", "permission error (403)", "out-of-memory error", "no space left on device", "HTTP 503", "DNS resolution failure"],
        "name": ["Alice", "Tom", "Mika", "Ken", "Priya", "Lena", "Omar", "Sora", "Daniel", "Rin"],
        "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "tomorrow", "the day after tomorrow", "next Monday"],
        "day_past": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        "ev": ["meeting", "dentist appointment", "sync-up", "interview", "lunch", "online class", "weekly stand-up"],
        "todo": ["take the medicine", "bring in the laundry", "send the invoice", "call mom", "take out the trash", "print the documents"],
        "city": ["Tokyo", "Osaka", "Sapporo", "Fukuoka", "Seattle", "Berlin", "Lisbon"],
        "file": ["minutes.docx", "invoice_october.pdf", "report_final.xlsx", "trip_plan.txt", "contract_v3.pdf"],
        "ftype": ["spreadsheet", "slide deck", "PDF", "log file"],
        "kw": ["budget", "contract", "schedule", "incident report", "quote"],
        "hobby": ["cooking", "running", "reading", "photography", "board games", "guitar"],
        "fact": ["the spare key is in the shoe cabinet by the door", "the Wi-Fi password was changed", "the parking spot changes next month", "the dog gets vaccinated in spring", "meeting room B is unavailable until 2 pm"],
        "topic": ["the new GPU release", "exchange rates", "next week's event information", "the latest security vulnerabilities", "the news"],
        "word": ["serendipity", "bottleneck", "redundant", "idempotent", "exponentiation"],
        "company": ["Company A", "Northwind", "Sunrise Electric", "Mizuho Trading"],
        "allergy": ["buckwheat", "eggs", "shrimp", "peanuts"],
        "folder": ["Archive", "Work", "Downloads", "Shared"],
        "msg": ["I'll be late", "I checked the documents", "Got it", "Fine with tomorrow's plan"],
        "what": ["temperature", "exchange rate", "train service status", "stock price"],
        "phrase_en": ["good morning", "thank you very much", "see you tomorrow", "where is the station"],
        "phrase_ja": ["お先に失礼します", "よろしくお願いします", "ありがとうございました"],
        "greet": ["hey, how's it going", "hello there", "good morning", "I'm back"],
        "os": ["Linux", "macOS", "Windows"],
    },
}

COLORS = {"ja": ["赤", "青", "緑", "黒", "白", "黄色"], "en": ["red", "blue", "green", "black", "white", "yellow"]}
OBJECTS = {
    "ja": ["かばん", "傘", "本", "靴", "ノートパソコン", "マグカップ"],
    "en": [("bag", "bags"), ("umbrella", "umbrellas"), ("book", "books"), ("shoe", "shoes"), ("laptop", "laptops"), ("mug", "mugs")],
}
PLACES = {
    "ja": ["駅前の店", "市場", "オンラインストア", "空港の売店", "近所のスーパー", "商店街", "図書館", "公園", "美術館"],
    "en": ["the station shop", "the market", "an online store", "the airport shop", "the local supermarket", "the shopping street", "the library", "the park", "the museum"],
}
WORKS = {
    "ja": ["報告書", "設計書", "見積書", "試験計画", "発表資料"],
    "en": ["the report", "the design doc", "the quote", "the test plan", "the presentation"],
}

FILLERS = {
    "ja": [
        "システムは正常に稼働している。", "現在の接続状態は良好。", "ユーザーの端末は{os}で動作している。",
        "直近の会話は{n}ターンある。", "現在時刻は{h}時{mm}分。", "CPU温度は正常範囲内にある。",
        "画面は{theme}モードで表示されている。", "ネットワーク遅延は{ms}msほど。", "今日の予定は{n}件登録されている。",
        "ログの保存先は定期的に整理されている。", "センサーの値は安定している。",
        "ユーザーは{wd}にも同じ端末を使っていた。", "ファンの回転数は通常どおり。", "今日のログインは{n}回目。", "音量は{pct}%に設定されている。", "ストレージの空きは十分にある。", "画面の明るさは{pct}%。",
    ],
    "en": [
        "The system is running normally.", "The connection status is good.", "The user's device is running {os}.",
        "The recent conversation has {n} turns.", "The current time is {h}:{mm}.", "The CPU temperature is within the normal range.",
        "The screen is in {theme} mode.", "Network latency is around {ms} ms.", "{n} events are registered for today.",
        "Log storage is tidied up on a regular basis.", "Sensor readings are stable.",
        "The user also used this device on {wd}.", "Fan speed is normal.", "This is login number {n} today.", "Volume is set to {pct}%.", "There is plenty of free storage.", "Screen brightness is {pct}%.",
    ],
}
IRRELEVANT = {
    "ja": [
        "窓の外では雨が降りはじめた。", "近所のパン屋は日曜が定休日だ。", "猫がソファの上で寝ている。",
        "隣の席の人がコーヒーをこぼした。", "駅前の桜はもう散ってしまった。", "庭のトマトが赤くなってきた。",
        "昨日の夕焼けはとてもきれいだった。", "冷蔵庫の中に牛乳が一本残っている。",
    ],
    "en": [
        "It started to rain outside the window.", "The bakery down the street is closed on Sundays.", "The cat is asleep on the sofa.",
        "Someone at the next desk spilled their coffee.", "The cherry blossoms by the station have already fallen.", "The tomatoes in the garden are turning red.",
        "Yesterday's sunset was beautiful.", "There is one carton of milk left in the fridge.",
    ],
}
PREAMBLES = {
    "ja": ["__AGENT__ は常駐エージェントとして、ユーザーの端末上で動いている。", "[__AGENT__ 観測ログ]", "常駐エージェント __AGENT__ の判断エンジンが入力を受け取った。"],
    "en": ["__AGENT__ is a resident agent running on the user's device.", "[__AGENT__ observation log]", "The decision engine of the resident agent __AGENT__ received this input."],
}

QUESTIONS = {
    "state": {
        "ja": ["次の行動として最も適切なのはどれか。", "この状況で、__AGENT__ はどうすべきか。", "最善の判断を選べ。", "どの行動を取るべきか。"],
        "en": ["Which action is the most appropriate next?", "What should __AGENT__ do in this situation?", "Choose the best decision.", "Which action should be taken?"],
    },
    "gate": {
        "ja": ["常駐エージェントとして今どう判断するか。", "この場面で__AGENT__が取るべき対応はどれか。", "ゲート判断を選べ。"],
        "en": ["As a resident agent, how should __AGENT__ decide right now?", "Which response should __AGENT__ take in this scene?", "Pick the gate decision."],
    },
    "intent": {
        "ja": ["この依頼はどの機能で処理すべきか。", "ユーザーの意図に最も近いものを選べ。", "適切なルーティング先はどれか。"],
        "en": ["Which handler should process this request?", "Pick the closest match to the user's intent.", "Where should this be routed?"],
    },
    "rank": {
        "ja": ["ユーザーの希望に最も合うのはどれか。", "最も適した選択肢を選べ。", "この条件で一番良いのはどれか。"],
        "en": ["Which option best fits the user's preference?", "Choose the most suitable option.", "Which one is best under this condition?"],
    },
    "sent": {
        "ja": ["この発言の感情はどれか。", "書き手の評価に最も近いのはどれか。"],
        "en": ["What is the sentiment of this message?", "Which best describes the writer's attitude?"],
    },
    "prio": {
        "ja": ["この連絡の優先度はどれか。", "どの優先度で扱うべきか。"],
        "en": ["What priority does this message have?", "At which priority should this be handled?"],
    },
    "nli": {
        "ja": ["次の主張は前提から言えるか：「{h}」", "「{h}」という主張と文脈の関係はどれか。"],
        "en": ["Does the following statement hold given the context: \"{h}\"", "What is the relation between the context and the claim \"{h}\"?"],
    },
    "vague": {
        "ja": ["どうする？", "どれがいい？", "どう思う？", "それでいい？", "どう進める？"],
        "en": ["What now?", "Which one?", "What do you think?", "Is that fine?", "How should we proceed?"],
    },
}

# ctx_paraphrase 用の同義語辞書（先頭一致した全てを置換）
SYN = {
    "ja": [
        ("ユーザー:", "ユーザーの発言:"), ("失敗", "エラー終了"), ("正常", "問題なく"), ("現在", "いま"), ("確認", "チェック"),
        ("記録", "保存"), ("予定", "スケジュール"), ("検索", "探索"), ("通知", "お知らせ"), ("元に戻せる", "復元できる"),
        ("元に戻せない", "復元できない"), ("不明", "わからない"), ("急ぎではない", "急がない"), ("依頼", "リクエスト"),
        ("ある。", "あります。"), ("いる。", "います。"), ("した。", "しました。"), ("大満足", "とても満足"), ("最悪", "ひどい"),
    ],
    "en": [
        ("User:", "The user says:"), ("failed", "did not succeed"), ("currently", "right now"), ("check", "verify"),
        ("notification", "alert"), ("cannot be undone", "is irreversible"), ("normally", "without problems"),
        ("request", "ask"), ("unknown", "not known"), ("finished", "completed"), ("bought", "purchased"),
        ("sent", "dispatched"), ("wants", "would like"), ("great", "excellent"), ("terrible", "awful"),
    ],
}
SYN_PREFIX = {"ja": "補足として、次の状況メモを参照。", "en": "For reference, here are the situation notes."}

# ---------------------------------------------------------------------------
# 2. データ構造
# ---------------------------------------------------------------------------


@dataclass
class CandSpec:
    role: str
    slots: dict = field(default_factory=dict)


@dataclass
class Scn:
    category: str
    lang: str
    seed: int
    sents: list  # list[(text, kind)] kind in {"core", "filler"}
    sep: str
    question: str
    q_kind: str
    cands: list  # list[CandSpec]（ベース順序）
    gold: int | None



# 常駐エージェント名（テンプレート中の __AGENT__）。既定は中立名。既存 replay DB と同一の生成を再現する場合は
# 生成時と同じ名前を --agent-name / 環境変数 SYNTH_AGENT_NAME で与える（data/ 側の記録を参照）。
DEFAULT_AGENT_NAME = "Navi"
_PREAMBLES_T = PREAMBLES
_QUESTIONS_T = QUESTIONS


def _sub_agent(obj, name: str):
    if isinstance(obj, str):
        return obj.replace("__AGENT__", name)
    if isinstance(obj, dict):
        return {k: _sub_agent(v, name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sub_agent(v, name) for v in obj]
    return obj


def set_agent_name(name: str) -> None:
    global PREAMBLES, QUESTIONS, AGENT_NAME
    AGENT_NAME = name
    PREAMBLES = _sub_agent(_PREAMBLES_T, name)
    QUESTIONS = _sub_agent(_QUESTIONS_T, name)


set_agent_name(os.environ.get("SYNTH_AGENT_NAME", DEFAULT_AGENT_NAME))

def _filler_slots(rng: random.Random, L: str) -> dict:
    s = _num_slots(rng)
    s["os"] = _pick(rng, BK[L]["os"])
    s["theme"] = _pick(rng, ["ダーク", "ライト"] if L == "ja" else ["dark", "light"])
    s["wd"] = _pick(rng, BK[L]["day_past"])
    return s


def _pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def _slot(rng: random.Random, L: str, key: str) -> str:
    return _pick(rng, BK[L][key])


def _num_slots(rng: random.Random) -> dict:
    return {
        "n": rng.randint(2, 9), "h": rng.randint(7, 21), "mm": f"{rng.randrange(0, 60):02d}", "m": rng.randint(5, 55),
        "pct": rng.randint(5, 95), "ms": rng.randint(8, 240),
    }


# ---------------------------------------------------------------------------
# 3. シナリオ生成
# ---------------------------------------------------------------------------
# 各 axis 関数: fn(rng, L, g) -> list[str]（core 文）。g は gold 側の役割 index（roles の順）。

AXIS_ROLES: dict[str, list[str]] = {
    "continue_stop": ["continue", "stop"],
    "answer_search": ["answer", "search"],
    "ask_act": ["ask", "act"],
    "remember_discard": ["remember", "discard"],
    "retrieve_skip": ["retrieve", "skip"],
    "safe_defer": ["safe_action", "defer"],
    "model_route": ["local", "larger"],
    "gate": ["speak", "silent"],
}


def _u(L: str, text: str) -> str:
    return f"ユーザー:「{text}」" if L == "ja" else f"User: \"{text}\""


def ax_continue_stop(rng, L, g):
    b = BK[L]
    task, err, name = _pick(rng, b["task"]), _pick(rng, b["err"]), _pick(rng, b["name"])
    n, m, pct, bat = rng.randint(3, 8), rng.randint(3, 40), rng.randint(60, 95), rng.randint(3, 15)
    which = rng.randrange(4)
    ja = L == "ja"
    if which == 0:
        if g == 1:
            return ([f"{task}が{n}回連続で同じ{err}により失敗している。", "原因は未解決のままで、再試行しても状況は変わらない。"] if ja else
                    [f"The {task} has failed {n} times in a row with the same {err}.", "The cause is still unresolved and retrying has not changed anything."])
        return ([f"{task}は{pct}%まで進んでいる。", f"直近のログにエラーはなく、残りは約{m}分。"] if ja else
                [f"The {task} is {pct}% complete.", f"No errors in the recent log; about {m} minutes remain."])
    if which == 1:
        run = f"{task}は現在実行中。" if ja else f"The {task} is currently running."
        if g == 1:
            return [run, _u(L, f"待って、{task}は止めて" if ja else f"wait, stop the {task}")]
        return [run, _u(L, f"そのまま{task}を続けて" if ja else f"keep the {task} going")]
    if which == 2:
        if g == 1:
            return ([f"ディスクの空きが{pct % 4 + 1}%しかなく、書き込みエラーが出始めている。", f"{task}を続けるとデータが壊れる恐れがある。"] if ja else
                    [f"Only {pct % 4 + 1}% of the disk is free and write errors have started.", f"Continuing the {task} may corrupt data."])
        return ([f"ディスクの空きは{pct}%あり十分。", f"{task}は順調に進んでいる。"] if ja else
                [f"{pct}% of the disk is free, which is plenty.", f"The {task} is going smoothly."])
    if g == 1:
        return ([f"バッテリー残量が{bat}%で、充電器はつながっていない。", f"{task}はあと{m}分かかる。"] if ja else
                [f"Battery is at {bat}% and the charger is not connected.", f"The {task} needs {m} more minutes."])
    return ([f"電源に接続されている。{task}はあと{m}分で終わる。", f"{name}さんも結果を待っている。"] if ja else
            [f"The device is plugged in. The {task} will finish in {m} minutes.", f"{name} is also waiting for the result."])


def ax_answer_search(rng, L, g):
    b = BK[L]
    ja = L == "ja"
    name, topic, city, what = _pick(rng, b["name"]), _pick(rng, b["topic"]), _pick(rng, b["city"]), _pick(rng, b["what"])
    word, company = _pick(rng, b["word"]), _pick(rng, b["company"])
    a, c = rng.randint(11, 98), rng.randint(11, 98)
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    year = rng.randint(2019, 2023)
    which = rng.randrange(3)
    if which == 0:
        if g == 0:
            return ([_u(L, f"{name}の誕生日はいつ？"), f"メモリには「{name}の誕生日は{month}月{day}日」と記録されている。"] if ja else
                    [_u(L, f"when is {name}'s birthday?"), f"Memory contains the entry \"{name}'s birthday is {month}/{day}\"."])
        return ([_u(L, f"{topic}の最新情報を教えて"), f"手元の情報は{year}年のもので古い可能性がある。"] if ja else
                [_u(L, f"tell me the latest about {topic}"), f"The information on hand dates from {year} and may be outdated."])
    if which == 1:
        if g == 0:
            return [_u(L, f"{a}+{c}はいくつ？" if ja else f"what is {a}+{c}?")]
        return ([_u(L, f"今の{city}の{what}は？"), "リアルタイムの情報源には接続できる。"] if ja else
                [_u(L, f"what is the current {what} in {city}?"), "A real-time information source is reachable."])
    if g == 0:
        return ([f"ユーザーは「{word}」の意味を尋ねている。", "これは一般的な語で、既知の知識で確実に説明できる。"] if ja else
                [f"The user is asking what \"{word}\" means.", "It is a common term that can be explained reliably from known knowledge."])
    return ([f"ユーザーは{company}の先月の決算内容を尋ねている。", "モデルの知識にはその情報がない。"] if ja else
            [f"The user is asking about {company}'s earnings last month.", "The model's knowledge does not contain that information."])


def ax_ask_act(rng, L, g):
    b = BK[L]
    ja = L == "ja"
    ftype, file, folder, name = _pick(rng, b["ftype"]), _pick(rng, b["file"]), _pick(rng, b["folder"]), _pick(rng, b["name"])
    msg, ev, day = _pick(rng, b["msg"]), _pick(rng, b["ev"]), _pick(rng, b["day"])
    n, h = rng.randint(3, 40), rng.randint(8, 19)
    which = rng.randrange(3)
    if which == 0:
        if g == 0:
            return ([_u(L, f"古い{ftype}を消しておいて"), f"『古い』の基準は不明で、該当しうるものが{n}件ある。削除は元に戻せない。"] if ja else
                    [_u(L, f"delete the old {ftype}s"), f"\"Old\" is not defined and {n} files could match. Deletion cannot be undone."])
        return ([_u(L, f"{file}を{folder}に移動して"), "対象は一意に特定でき、移動は元に戻せる。"] if ja else
                [_u(L, f"move {file} to {folder}"), "The target is uniquely identified and the move can be undone."])
    if which == 1:
        if g == 0:
            return ([_u(L, f"{name}にメールを送って"), f"連絡先に{name}が{n % 4 + 2}人いて、どの人か分からない。"] if ja else
                    [_u(L, f"send an email to {name}"), f"There are {n % 4 + 2} contacts named {name} and it is unclear which one is meant."])
        return ([_u(L, f"{name}に『{msg}』とメールして"), f"連絡先に{name}は1人だけ。"] if ja else
                [_u(L, f"email {name} that \"{msg}\""), f"There is only one contact named {name}."])
    if g == 0:
        return ([_u(L, "予定を入れて"), "日時も内容も指定されていない。"] if ja else
                [_u(L, "put it on my calendar"), "Neither the date nor the content was specified."])
    return ([_u(L, f"{day}の{h}時に{ev}を入れて"), "その時間は空いている。"] if ja else
            [_u(L, f"add a {ev} for {day} at {h}"), "That time slot is free."])


def ax_remember_discard(rng, L, g):
    b = BK[L]
    ja = L == "ja"
    allergy, city, word = _pick(rng, b["allergy"]), _pick(rng, b["city"]), _pick(rng, b["word"])
    ev, h = _pick(rng, b["ev"]), rng.randint(8, 19)
    day = _pick(rng, b["day_past"])
    code, m = rng.randint(100000, 999999), rng.randint(3, 10)
    which = rng.randrange(3)
    if which == 0:
        if g == 0:
            return [_u(L, f"私は{allergy}アレルギーなので、料理の提案では気をつけて" if ja else f"I'm allergic to {allergy}, so be careful with food suggestions")]
        return ([_u(L, f"確認コードは{code}だよ"), f"このコードは一度きりで{m}分後に失効する。"] if ja else
                [_u(L, f"the verification code is {code}"), f"The code is single-use and expires in {m} minutes."])
    if which == 1:
        if g == 0:
            return [_u(L, f"来月から{city}に引っ越す予定なんだ" if ja else f"I'm planning to move to {city} next month")]
        return [_u(L, f"あ、今の{word}、打ち間違いだから無視して" if ja else f"oh, ignore that '{word}', it was a typo")]
    if g == 0:
        return ([f"ユーザーは毎週{day}の{h}時に{ev}があると話した。", "これは繰り返し発生する予定のようだ。"] if ja else
                [f"The user said there is a {ev} every {day} at {h}.", "It appears to be a recurring commitment."])
    return ([_u(L, "えーと、何だっけ…まあいいや" if ja else "um, what was it... never mind"), "特に依頼や情報は含まれていない。"] if ja else
            [_u(L, "um, what was it... never mind"), "It contains no request or information."])


def ax_retrieve_skip(rng, L, g):
    b = BK[L]
    ja = L == "ja"
    topic, a, c = _pick(rng, b["topic"]), rng.randint(11, 60), rng.randint(11, 60)
    which = rng.randrange(2)
    if which == 0:
        if g == 0:
            return ([_u(L, f"先週話した{topic}の件、どうなった？"), "過去の会話の要約は長期メモリに保存されている。"] if ja else
                    [_u(L, f"what happened with {topic} we discussed last week?"), "Summaries of past conversations are stored in long-term memory."])
        return [_u(L, "ありがとう、おやすみ" if ja else "thanks, good night")]
    if g == 0:
        return ([_u(L, "いつものお店を予約して"), "「いつもの」が何かは今の会話にはない。メモリには好みが記録されている。"] if ja else
                [_u(L, "book my usual place"), "What \"usual\" means is not in the current conversation. Memory holds the user's preferences."])
    return ([_u(L, f"{a}+{c}は？"), "今の会話だけで答えられる。"] if ja else
            [_u(L, f"what is {a}+{c}?"), "It can be answered from the current conversation alone."])


def ax_safe_defer(rng, L, g):
    b = BK[L]
    ja = L == "ja"
    task = _pick(rng, b["task"])
    n = rng.randint(2, 30)
    which = rng.randrange(3)
    if which == 0:
        if g == 0:
            return ([f"{task}の設定を変更する必要がある。バックアップは取得済みで、変更は{n}行の差分のみ。", "元に戻せる。"] if ja else
                    [f"The {task} settings need to change. A backup exists and the change is a diff of only {n} lines.", "It can be reverted."])
        return ([f"本番環境への{task}を依頼されたが、テストが{n}件失敗中で担当者は不在。", "急ぎではない。"] if ja else
                [f"A production {task} was requested, but {n} tests are failing and the owner is away.", "It is not urgent."])
    if which == 1:
        if g == 0:
            return ([f"不要なキャッシュ{n * 10}MBを削除したい。", "ごみ箱経由で復元できる。"] if ja else
                    [f"{n * 10} MB of unneeded cache should be removed.", "It can be restored from the trash."])
        return ([f"共有データベースのスキーマ変更を求められたが、他チームが使用中で影響範囲が不明。", "期限は来週。"] if ja else
                ["A schema change on the shared database was requested, but another team is using it and the impact is unknown.", "The deadline is next week."])
    if g == 0:
        return ([f"テスト環境で{task}の動作を確認する。本番には影響しない。"] if ja else
                [f"The {task} will be verified in the test environment. Production is unaffected."])
    return ([f"金曜の夜に{task}の本番リリースを行う案が出ている。週末は対応できる人がいない。"] if ja else
            [f"A proposal is to run the production {task} on Friday night. No one can respond over the weekend."])


def ax_model_route(rng, L, g):
    b = BK[L]
    ja = L == "ja"
    word, ev = _pick(rng, b["word"]), _pick(rng, b["ev"])
    n, h = rng.randint(12, 80), rng.randint(8, 19)
    which = rng.randrange(3)
    if which == 0:
        if g == 0:
            return ([_u(L, f"「{word}」を言い換えて"), "短い一語の単純な作業。"] if ja else
                    [_u(L, f"paraphrase \"{word}\""), "A simple single-word task."])
        return ([_u(L, f"この{n}ページの契約書の矛盾点をすべて洗い出して"), "高度な読解と長い推論が必要。"] if ja else
                [_u(L, f"find every inconsistency in this {n}-page contract"), "It requires deep reading and long reasoning."])
    if which == 1:
        if g == 0:
            return [_u(L, "今何時？" if ja else "what time is it?")]
        return ([_u(L, "この数学の証明の誤りを指摘して"), "複数段階の論理的検証が必要。"] if ja else
                [_u(L, "point out the flaw in this mathematical proof"), "Multi-step logical verification is needed."])
    if g == 0:
        return ([_u(L, f"{ev}の予定を{h}時に登録して"), "定型的な処理で、判断は不要。"] if ja else
                [_u(L, f"register the {ev} for {h}"), "A routine operation that needs no judgment."])
    return ([f"ユーザーは複数の資料を横断して比較分析し、結論をまとめるよう求めている。", "資料は計{n}ページある。"] if ja else
            ["The user wants a comparative analysis across several documents with a written conclusion.", f"The documents total {n} pages."])


def ax_gate(rng, L, g):
    b = BK[L]
    ja = L == "ja"
    ev, task, what = _pick(rng, b["ev"]), _pick(rng, b["task"]), _pick(rng, b["what"])
    m, h = rng.randint(3, 20), rng.randint(2, 9)
    which = rng.randrange(3)
    if which == 0:
        if g == 0:
            return ([f"{ev}の開始まであと{m}分だが、ユーザーはまだ別の作業をしていて気づいていない。"] if ja else
                    [f"The {ev} starts in {m} minutes but the user is still working on something else and has not noticed."])
        return ([f"ユーザーは集中モードで{ev}中。", f"届いたのは{what}についての軽微な通知のみ。"] if ja else
                [f"The user is in focus mode during a {ev}.", f"The only incoming item is a minor notification about {what}."])
    if which == 1:
        if g == 0:
            return ([f"{task}がエラーで停止した。", "ユーザーの今日の作業に影響する。"] if ja else
                    [f"The {task} stopped with an error.", "It will affect the user's work today."])
        return ([f"{task}が正常に完了した。", "ユーザーは今、電話中で、結果は急ぎではない。"] if ja else
                [f"The {task} completed successfully.", "The user is on a phone call and the result is not urgent."])
    if g == 0:
        return ([_u(L, f"{m}分後に教えて" if ja else f"tell me in {m} minutes"), f"{m}分が経過した。"] if ja else
                [_u(L, f"tell me in {m} minutes"), f"{m} minutes have passed."])
    return ([f"ユーザーは{h}時間以上操作しておらず、就寝中の可能性が高い。", "通知内容は広告メール。"] if ja else
            [f"The user has not touched the device for over {h} hours and is probably asleep.", "The incoming item is a promotional email."])


AXIS_FNS: dict[str, Callable] = {
    "continue_stop": ax_continue_stop, "answer_search": ax_answer_search, "ask_act": ax_ask_act,
    "remember_discard": ax_remember_discard, "retrieve_skip": ax_retrieve_skip, "safe_defer": ax_safe_defer,
    "model_route": ax_model_route, "gate": ax_gate,
}

# ツール利用シナリオ
TOOL_NEEDS = ["calc", "calendar", "weather", "files", "translate", "websearch"]


def _tool_utterance(rng, L, kind):
    b = BK[L]
    ja = L == "ja"
    a, c, p = rng.randint(12, 99), rng.randint(12, 99), rng.choice([10, 15, 20, 30])
    if kind == "calc":
        return _pick(rng, [f"{a}×{c}はいくつ？", f"{a * 100}円の{p}%引きはいくら？"] if ja else [f"what is {a} times {c}?", f"how much is {a * 10} dollars after {p}% off?"])
    if kind == "calendar":
        return _pick(rng, [f"{_slot(rng, L, 'day')}の予定は？", "今週の空き時間を教えて"] if ja else [f"what's on my schedule {_slot(rng, L, 'day')}?", "when am I free this week?"])
    if kind == "weather":
        return _pick(rng, [f"{_slot(rng, L, 'city')}の天気を教えて", f"{_slot(rng, L, 'city')}は明日傘が必要？"] if ja else [f"tell me the weather in {_slot(rng, L, 'city')}", f"will I need an umbrella in {_slot(rng, L, 'city')} tomorrow?"])
    if kind == "files":
        f = _slot(rng, L, "file")
        return f"{f}を開いて内容を要約して" if ja else f"open {f} and summarize it"
    if kind == "translate":
        return f"「{_slot(rng, L, 'phrase_en')}」を日本語に訳して" if ja else f"translate \"{_slot(rng, L, 'phrase_ja')}\" into English"
    if kind == "websearch":
        return f"{_slot(rng, L, 'topic')}の最新ニュースを調べて" if ja else f"look up the latest news on {_slot(rng, L, 'topic')}"
    return _pick(rng, ["ありがとう、助かった", "今日は楽しかったな"] if ja else ["thanks, that helped", "I had fun today"])  # none


def scn_tool(rng, L):
    """tool-A/tool-B/no-tool: 候補は ツール 2〜4 個 + ツールなし（3〜5 候補）。"""
    need = _pick(rng, TOOL_NEEDS + ["none"] * 2)
    n_tools = rng.randint(2, 4)
    kinds = [need] if need != "none" else []
    others = [k for k in TOOL_NEEDS if k not in kinds]
    rng.shuffle(others)
    kinds += others[: n_tools - len(kinds)]
    sents = [_u(L, _tool_utterance(rng, L, need))]
    cands = [CandSpec("use_tool", {"kind": k}) for k in kinds] + [CandSpec("no_tool")]
    gold_spec = 0 if need != "none" else len(cands) - 1
    return sents, cands, gold_spec


def _shuffle_cands(rng, cands, gold):
    order = list(range(len(cands)))
    rng.shuffle(order)
    new = [cands[i] for i in order]
    ng = None if gold is None else order.index(gold)
    return new, ng


def _core_state_action(rng, L):
    """(sents, cands, gold, q_kind)。gold は明確な 2 値 axis では 65% で設定、残りは None（曖昧な事例として）。"""
    r = rng.random()
    if r < 0.18:
        sents, cands, gold = scn_tool(rng, L)
        return sents, cands, gold, "state"
    axis = _pick(rng, list(AXIS_ROLES))
    roles = AXIS_ROLES[axis]
    g = rng.randrange(2)
    sents = AXIS_FNS[axis](rng, L, g)
    cands = [CandSpec(roles[0]), CandSpec(roles[1])]
    gold = g
    # 3〜5 候補へ拡張（余分な候補は明らかに状況に合わない役割ではなく「別の妥当な行動」なので gold は None にする）
    if rng.random() < 0.30:
        extra_pool = [r_ for ax, rs in AXIS_ROLES.items() if ax != axis for r_ in rs if r_ not in ("speak", "silent")]
        rng.shuffle(extra_pool)
        for r_ in extra_pool[: rng.randint(1, 3)]:
            cands.append(CandSpec(r_))
        gold = None
    elif rng.random() < 0.12:
        gold = None  # 状況は十分に明確でも gold を付けない例も混ぜる
    return sents, cands, gold, "state"


def _core_agent_gate(rng, L):
    axis = _pick(rng, ["gate", "gate", "gate", "model_route", "model_route", "safe_defer", "ask_act", "continue_stop"])
    roles = AXIS_ROLES[axis]
    g = rng.randrange(2)
    sents = AXIS_FNS[axis](rng, L, g)
    cands = [CandSpec(roles[0]), CandSpec(roles[1])]
    gold = g if rng.random() < 0.7 else None
    if rng.random() < 0.25:
        pool = ["speak", "silent", "larger", "local", "defer", "ask", "act"]
        rng.shuffle(pool)
        have = {c.role for c in cands}
        for r_ in [p for p in pool if p not in have][: rng.randint(1, 2)]:
            cands.append(CandSpec(r_))
        gold = None
    return sents, cands, gold, "gate"


# NLI ---------------------------------------------------------------------

def _nli_frames_ja(s):
    w, d, p, p2, n, n2, o, c, c2, ot, ot2, wk = (s[k] for k in ("who", "day", "place", "place2", "n", "n2", "obj", "color", "color2", "other", "other2", "work"))
    return {
        "buy": dict(
            prem=f"{w}は{d}、{p}で{c}の{o}を{n}個買った。",
            ent=[f"{w}は{p}で{o}を買った。", f"{w}は{d}に{c}の{o}を購入した。"] + ([f"{w}は{o}を2個以上買った。"] if n >= 2 else []),
            con=[f"{w}は{o}を買っていない。", f"{w}は{p2}で{o}を買った。", f"{w}が買った{o}は{c2}だった。", f"{w}は{o}を{n2}個買った。"],
            neu=[f"{w}は{o}を現金で支払った。", f"{w}は{o}を友人への贈り物にした。", f"{w}は{o}を割引価格で買った。"]),
        "visit": dict(
            prem=f"{w}は{d}に{p}へ行き、{n}時間過ごした。",
            ent=[f"{w}は{p}にいたことがある。", f"{w}は{d}に出かけた。"] + ([f"{w}は{p}に1時間以上いた。"] if n >= 2 else []),
            con=[f"{w}は{d}に{p}へ行かなかった。", f"{w}は{p}で{n2}時間過ごした。", f"{w}は{p2}にだけ行った。"],
            neu=[f"{w}は{p}で友人に会った。", f"{w}は{p}まで電車で行った。", f"{w}は{p}で写真を撮った。"]),
        "send": dict(
            prem=f"{w}は{d}に{ot}へ{n}通のメールを送った。",
            ent=[f"{w}は{ot}に連絡した。", f"{w}はメールを送った。"] + ([f"{w}は{ot}へ複数のメールを送った。"] if n >= 2 else []),
            con=[f"{w}は{ot}に何も送っていない。", f"{w}は{ot}へ{n2}通のメールを送った。", f"{w}は{ot2}にだけメールを送った。"],
            neu=[f"{w}のメールには添付ファイルがあった。", f"{ot}は{w}のメールに返信した。", f"{w}は{ot}にお礼を伝えた。"]),
        "finish": dict(
            prem=f"{w}は{d}までに{wk}を終え、{n}件の修正を反映した。",
            ent=[f"{w}は{wk}を終えた。", f"{w}は修正を反映した。"],
            con=[f"{w}は{wk}を終えていない。", f"{w}が反映した修正は{n2}件だった。"],
            neu=[f"{w}は{wk}を{ot}にレビューしてもらった。", f"{w}は{wk}の完了を報告した。", f"{w}は{wk}に予定より時間がかかった。"]),
    }


def _nli_frames_en(s):
    w, d, p, p2, n, n2, c, c2, ot, ot2, wk = (s[k] for k in ("who", "day", "place", "place2", "n", "n2", "color", "color2", "other", "other2", "work"))
    sg, pl = s["obj"]
    art = "an" if sg[0] in "aeiou" else "a"
    on = f"{art} {sg}" if n == 1 else pl          # 一般的な言及（冠詞つき単数 or 複数）
    on2 = f"{art} {sg}" if n2 == 1 else pl
    pc = f"{art} {c} {sg}" if n == 1 else f"{n} {c} {pl}"   # 前提中の「色つき数量句」
    pc_art = "an" if c[0] in "aeiou" else "a"
    if n == 1:
        pc = f"{pc_art} {c} {sg}"
    return {
        "buy": dict(
            prem=f"{w} bought {pc} at {p} on {d}.",
            ent=[f"{w} bought {on} at {p}.", f"{w} purchased {pc} on {d}."] + ([f"{w} bought more than one {sg}."] if n >= 2 else []),
            con=[f"{w} did not buy any {pl}.", f"{w} bought {on} at {p2}.", f"The {sg if n == 1 else pl} {w} bought {'was' if n == 1 else 'were'} {c2}.", f"{w} bought {n2 if n2 > 1 else 'one'} {sg if n2 == 1 else pl}."],
            neu=[f"{w} paid in cash for the {pl}.", f"{w} bought the {pl} as a gift for a friend.", f"{w} got the {pl} at a discount."]),
        "visit": dict(
            prem=f"{w} went to {p} on {d} and spent {n} {'hour' if n == 1 else 'hours'} there.",
            ent=[f"{w} has been to {p}.", f"{w} went out on {d}."] + ([f"{w} spent more than an hour at {p}."] if n >= 2 else []),
            con=[f"{w} did not go to {p} on {d}.", f"{w} spent {n2} {'hour' if n2 == 1 else 'hours'} at {p}.", f"{w} only went to {p2}."],
            neu=[f"{w} met a friend at {p}.", f"{w} took the train to {p}.", f"{w} took photos at {p}."]),
        "send": dict(
            prem=f"{w} sent {n} emails to {ot} on {d}.",
            ent=[f"{w} contacted {ot}.", f"{w} sent emails."] + ([f"{w} sent multiple emails to {ot}."] if n >= 2 else []),
            con=[f"{w} sent nothing to {ot}.", f"{w} sent {n2} emails to {ot}.", f"{w} only emailed {ot2}."],
            neu=[f"One of {w}'s emails had an attachment.", f"{ot} replied to {w}'s email.", f"{w} thanked {ot}."]),
        "finish": dict(
            prem=f"{w} finished {wk} by {d} and applied {n} {'fix' if n == 1 else 'fixes'}.",
            ent=[f"{w} finished {wk}.", f"{w} applied fixes."],
            con=[f"{w} did not finish {wk}.", f"{w} applied {n2} {'fix' if n2 == 1 else 'fixes'}."],
            neu=[f"{w} had {wk} reviewed by {ot}.", f"{w} reported that {wk} was done.", f"{w} took longer than planned on {wk}."]),
    }


def scn_nli(rng, L):
    b = BK[L]
    n = rng.randint(1, 5)
    n2 = rng.choice([x for x in range(1, 7) if x != n])
    colors = rng.sample(COLORS[L], 2)
    places = rng.sample(PLACES[L], 2)
    names = rng.sample(b["name"], 3)
    objs = OBJECTS[L]
    s = {
        "who": names[0], "other": names[1], "other2": names[2], "day": _pick(rng, b["day_past"]), "place": places[0], "place2": places[1],
        "n": n, "n2": n2, "color": colors[0], "color2": colors[1], "obj": _pick(rng, objs), "work": _pick(rng, WORKS[L]),
    }
    frames = (_nli_frames_ja if L == "ja" else _nli_frames_en)(s)
    fr = frames[_pick(rng, list(frames))]
    label = rng.choices(["ent", "con", "neu"], weights=[1, 1, 1])[0]
    two_way = rng.random() < 0.2
    if two_way and label == "neu":
        label = _pick(rng, ["ent", "con"])
    hyp = _pick(rng, fr[label])
    roles = {"ent": "nli_entail", "con": "nli_contra", "neu": "nli_neutral"}
    cand_roles = ["nli_entail", "nli_contra"] + ([] if two_way else ["nli_neutral"])
    cands = [CandSpec(r) for r in cand_roles]
    gold = cand_roles.index(roles[label])
    sents = [fr["prem"]]
    if rng.random() < 0.4:  # 無関係だが整合する背景文
        sents.append(_pick(rng, FILLERS[L]).format(**_filler_slots(rng, L)))
    q = _pick(rng, QUESTIONS["nli"][L]).format(h=hyp)
    return sents, q, cands, gold


# intent ------------------------------------------------------------------

INTENT_UTT = {
    "ja": {
        "intent_calendar": ["来週の{day_past}の{h}時に{ev}の予定を入れて", "{day}の予定を教えて", "{ev}を{day}の{h}時に変更して", "今週空いている時間はいつ？"],
        "intent_reminder": ["{h}時に{todo}って教えて", "{m}分後に{todo}のリマインドして", "{day}の朝に{todo}ことを忘れないよう通知して"],
        "intent_memo": ["{fact}ってメモしておいて", "これを覚えておいて：{fact}", "さっきの内容をノートに保存して"],
        "intent_weather": ["{day}の{city}の天気は？", "傘は必要かな、{day}は", "{city}は今何度？"],
        "intent_chat": ["今日はちょっと疲れたよ", "おすすめの{hobby}ある？雑談したい", "{greet}", "最近{hobby}にはまってるんだ"],
        "intent_files": ["{file}ってファイルどこにある？", "{kw}を含む資料を探して", "先週編集した{ftype}を開いて"],
        "intent_settings": ["画面の明るさを{pct}%にして", "通知をオフにして", "音量を下げて", "ダークモードに切り替えて"],
        "intent_translate": ["「{phrase_en}」を日本語に訳して", "この文を英語にして：{phrase_ja}", "{phrase_en}ってどういう意味？"],
    },
    "en": {
        "intent_calendar": ["put a {ev} on my calendar for {day} at {h}", "what's on my schedule {day}?", "move the {ev} to {day} at {h}", "when am I free this week?"],
        "intent_reminder": ["remind me to {todo} at {h}", "ping me in {m} minutes to {todo}", "notify me {day} morning so I don't forget to {todo}"],
        "intent_memo": ["note down that {fact}", "save this to my notes: {fact}", "write that last bit in my notebook"],
        "intent_weather": ["what's the weather in {city} {day}?", "do I need an umbrella {day}?", "how warm is it in {city} right now?"],
        "intent_chat": ["I'm a bit tired today", "got any {hobby} ideas? just chatting", "{greet}", "I've been into {hobby} lately"],
        "intent_files": ["where is the file {file}?", "find documents that mention {kw}", "open the {ftype} I edited last week"],
        "intent_settings": ["set the screen brightness to {pct}%", "turn notifications off", "lower the volume", "switch to dark mode"],
        "intent_translate": ["translate \"{phrase_ja}\" into English", "how do you say \"{phrase_en}\" in Japanese?", "what does \"{phrase_en}\" mean in Japanese?"],
    },
}
INTENTS = list(INTENT_UTT["ja"])


def scn_intent(rng, L):
    b = BK[L]
    gold_intent = _pick(rng, INTENTS)
    tmpl = _pick(rng, INTENT_UTT[L][gold_intent])
    slots = {k: _pick(rng, v) for k, v in b.items()}
    slots.update(_num_slots(rng))
    utt = tmpl.format(**slots)
    k = rng.choices(list(K_DIST), weights=list(K_DIST.values()))[0]
    others = [i for i in INTENTS if i != gold_intent]
    rng.shuffle(others)
    roles = [gold_intent] + others[: k - 1]
    cands = [CandSpec(r) for r in roles]
    sents = [_u(L, utt)]
    if rng.random() < 0.3:
        sents.insert(0, _pick(rng, FILLERS[L]).format(**_filler_slots(rng, L)))
    return sents, _pick(rng, QUESTIONS["intent"][L]), cands, 0


# ranking -----------------------------------------------------------------

CRIT = {
    "ja": {"price": ["とにかく安いものを探している。", "予算をできるだけ抑えたいと言っている。"],
           "time": ["時間をかけたくないと言っている。", "できるだけ短時間で済ませたい。"],
           "rating": ["品質を最優先したいと言っている。", "評価の高いものを選びたい。"],
           "mixed": ["安さも早さも大事だと言っている。", "料金と時間のバランスを重視している。"]},
    "en": {"price": ["The user wants the cheapest option.", "The user is trying to keep costs as low as possible."],
           "time": ["The user does not want to spend much time.", "The user wants it done as quickly as possible."],
           "rating": ["The user puts quality first.", "The user wants to pick the highest-rated option."],
           "mixed": ["The user cares about both price and speed.", "The user wants a balance between cost and time."]},
}


def scn_ranking(rng, L, mixed=False):
    dom_key = _pick(rng, list(RANK_DOMAINS))
    dom = RANK_DOMAINS[dom_key]
    k = rng.choices(list(K_DIST), weights=list(K_DIST.values()))[0]
    if mixed:
        k = max(k, 3)
    attrs = ["price", "time", "rating"]
    lv = [{a: rng.randrange(3) for a in attrs} for _ in range(k)]
    if mixed:
        crit, gold_idx = "mixed", None
        a, b = rng.sample(range(k), 2)
        lv[a].update(price=0, time=2)
        lv[b].update(price=2, time=0)
        for i in range(k):
            if i not in (a, b):  # 価格・時間のどちらも最良にならない候補
                lv[i].update(price=rng.choice([1, 2]), time=rng.choice([1, 2]))
        for i in range(k):
            lv[i]["rating"] = rng.randrange(3)
    else:
        crit = _pick(rng, attrs)
        gold_idx = rng.randrange(k)
        for i in range(k):
            lv[i][crit] = 0 if i == gold_idx else rng.choice([1, 2])
    nouns = rng.sample(range(len(dom["ja"]["train"])), k)
    cands = [CandSpec("rank_opt", {"dom": dom_key, "ni": nouns[i], **lv[i]}) for i in range(k)]
    scene = dom["scene_ja"] if L == "ja" else dom["scene_en"]
    city = _slot(rng, L, "city")
    s0 = f"ユーザーは{city}で{scene}。" if L == "ja" else f"The user {scene} in {city}."
    c0 = _pick(rng, CRIT[L][crit])
    sents = [s0, ("ユーザーは" + c0) if L == "ja" else c0]
    return sents, _pick(rng, QUESTIONS["rank"][L]), cands, gold_idx


# sentiment / priority ------------------------------------------------------

SENT = {
    "ja": {
        "subj": ["このアプリ", "新しいキーボード", "昨日のサポート対応", "カフェの接客", "配送", "新バージョンのUI", "このホテル"],
        "pos": ["とても使いやすくて大満足です", "期待以上で本当に助かった", "最高でした、また使いたい", "感動するほど速かった"],
        "neg": ["最悪だった、二度と使わない", "全然動かなくて腹が立つ", "対応が遅すぎてがっかり", "期待外れで返金してほしい"],
        "neu": ["について、{day}に{n}件の問い合わせがあった。", "の説明資料は{day}に更新された。", "に関する打ち合わせは{day}に行われた。"],
        "high": ["{subj}が停止しており、至急の対応をお願いします", "{ev}の{m}分前です、今すぐ確認してください", "本番障害が発生中です。最優先で対応してください"],
        "norm": ["{subj}について来週までに確認をお願いします", "{subj}の件、通常の手順で対応をお願いします", "今週中に{subj}の状況を共有してください"],
        "low": ["お時間のあるときに{subj}をご確認ください。急ぎではありません", "{subj}の件は、手が空いたらで構いません", "参考までに{subj}の情報を共有します。対応は不要です"],
    },
    "en": {
        "subj": ["this app", "the new keyboard", "yesterday's support service", "the cafe's service", "the delivery", "the new UI version", "this hotel"],
        "pos": ["It's so easy to use, I'm really happy with it", "It exceeded my expectations and helped a lot", "It was the best, I want to use it again", "It was impressively fast"],
        "neg": ["It was terrible, I'll never use it again", "It doesn't work at all and it's infuriating", "The response was far too slow, what a letdown", "It fell short and I want a refund"],
        "neu": ["There were {n} inquiries about {subj} on {day}.", "The documentation for {subj} was updated on {day}.", "A meeting about {subj} was held on {day}."],
        "high": ["{subj} is down, please handle it urgently", "It's {m} minutes before the {ev}, check right now", "A production incident is in progress. Top priority please"],
        "norm": ["Please check on {subj} by next week", "Regarding {subj}, please handle it through the normal process", "Please share the status of {subj} within this week"],
        "low": ["Please look at {subj} when you have time. It is not urgent", "No rush on {subj}, whenever you are free", "Sharing the info on {subj} for reference. No action is needed"],
    },
}


def scn_sentiment(rng, L):
    b = BK[L]
    s = SENT[L]
    ja = L == "ja"
    slots = {"day": _pick(rng, b["day_past"]), "p": rng.randint(5, 80) * (100 if ja else 1), "n": rng.randint(2, 9), "m": rng.randint(5, 30),
             "ev": _pick(rng, b["ev"]), "subj": _pick(rng, s["subj"])}
    if rng.random() < 0.72:
        kind = _pick(rng, ["pos", "neg", "neu"])
        subj = slots["subj"]
        if kind == "neu":
            body = (subj + _pick(rng, s["neu"]).format(**slots)) if ja else _pick(rng, s["neu"]).format(**slots)
        elif ja:
            body = f"{subj}、{_pick(rng, s[kind])}。"
        else:
            body = f"About {subj}: {_pick(rng, s[kind])}." if rng.random() < 0.5 else f"{_pick(rng, s[kind])}."
        roles = ["sent_pos", "sent_neg", "sent_neu"]
        gold_role = {"pos": "sent_pos", "neg": "sent_neg", "neu": "sent_neu"}[kind]
        q = _pick(rng, QUESTIONS["sent"][L])
        sents = [_u(L, body.rstrip("。.")) if rng.random() < 0.4 else body]
    else:
        kind = _pick(rng, ["high", "norm", "low"])
        body = _pick(rng, s[kind]).format(**slots) + ("。" if ja else ".")
        roles = ["prio_high", "prio_norm", "prio_low"]
        gold_role = {"high": "prio_high", "norm": "prio_norm", "low": "prio_low"}[kind]
        q = _pick(rng, QUESTIONS["prio"][L])
        sents = [("メッセージ:" if ja else "Message: ") + body]
    if rng.random() < 0.15:
        roles = [r for r in roles if r == gold_role or rng.random() < 0.6]
        if len(roles) < 2:
            roles = [gold_role, roles[0] if roles[0] != gold_role else ("sent_neu" if gold_role.startswith("sent") else "prio_norm")]
    cands = [CandSpec(r) for r in roles]
    return sents, q, cands, roles.index(gold_role)


# ambiguous ---------------------------------------------------------------

VAGUE_UTT = {
    "ja": ["それ、やっといて", "さっきのやつ、お願い", "あれどうなった？", "例の件、進めて", "それでよろしく"],
    "en": ["do that for me", "take care of that thing from before", "what happened to that?", "go ahead with the usual", "just do it like we said"],
}


def scn_ambiguous(rng, L):
    b = BK[L]
    ja = L == "ja"
    r = rng.random()
    if r < 0.25:  # 指示語が曖昧
        t1, t2 = rng.sample(b["task"], 2)
        sents = ([f"直近の作業は2つ：{t1}と{t2}。"] if ja else [f"There are two recent tasks: the {t1} and the {t2}."])
        sents.append(_u(L, _pick(rng, VAGUE_UTT[L])))
        cands = [CandSpec("ask"), CandSpec("do", {"x": t1}), CandSpec("do", {"x": t2})]
        if rng.random() < 0.5:
            cands.append(CandSpec("defer"))
        return sents, _pick(rng, QUESTIONS["state"][L]), cands, "state"
    if r < 0.60:  # 証拠の衝突
        axis = _pick(rng, list(AXIS_ROLES))
        roles = AXIS_ROLES[axis]
        fn = AXIS_FNS[axis]
        s0, s1 = fn(rng, L, 0), fn(rng, L, 1)
        sents = s0 + s1 if rng.random() < 0.5 else s1 + s0
        return sents, _pick(rng, QUESTIONS["state"][L]), [CandSpec(roles[0]), CandSpec(roles[1])], "state"
    if r < 0.70:  # 何も観測がない
        sents = ["直近の入力はなく、状態にも変化はない。", f"最後の操作は{rng.randint(5, 90)}分前だった。"] if ja else ["There has been no recent input and no state change.", f"The last user action was {rng.randint(5, 90)} minutes ago."]
        pool = ["answer", "search", "ask", "defer", "continue", "retrieve", "silent", "speak", "local"]
        rng.shuffle(pool)
        cands = [CandSpec(p) for p in pool[: rng.randint(3, 5)]]
        return sents, _pick(rng, QUESTIONS["gate"][L]), cands, "gate"
    if r < 0.85:  # 賛否が混在する感情
        s = SENT[L]
        subj = _pick(rng, s["subj"])
        pos, neg = _pick(rng, s["pos"]), _pick(rng, s["neg"])
        body = f"{subj}、{pos}。でも{neg}。" if ja else f"{pos}, but {neg[0].lower()}{neg[1:]}."
        cands = [CandSpec("sent_pos"), CandSpec("sent_neg"), CandSpec("sent_neu")]
        return [_u(L, body.rstrip("。."))], _pick(rng, QUESTIONS["sent"][L]), cands, "sent"
    sents, q, cands, _g = scn_ranking(rng, L, mixed=True)
    return sents, q, cands, "rank"


# ---------------------------------------------------------------------------
# context 組み立て
# ---------------------------------------------------------------------------


def _text_len(L: str, text: str) -> int:
    return len(text) if L == "ja" else len(text.split())


def _make_context_sents(rng, L, core, category, hi_scale: float = 1.0, core_last: bool = False):
    """core 文に filler を足して目標長に近づける。(text, kind) のリストを返す。"""
    lo, hi = CTX_LEN_RANGE[L]
    hi = max(lo + 10, int(hi * hi_scale))
    target = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    slots = _filler_slots(rng, L)
    pre = []
    if category in ("state_action", "agent_gate", "ambiguous") and rng.random() < 0.4:
        pre.append(_pick(rng, PREAMBLES[L]))
    cur = sum(_text_len(L, t) for t in core + pre)
    fillers: list[str] = []
    pool = list(FILLERS[L])
    rng.shuffle(pool)
    while cur < target and pool and len(fillers) < 12:
        t = pool.pop().format(**slots)
        fillers.append(t)
        cur += _text_len(L, t)
    layout = 0.0 if core_last else rng.random()   # core_last: teacher の先頭切り詰めで core が落ちないよう末尾に置く
    core_s = [(t, "core") for t in core]
    fil_s = [(t, "filler") for t in fillers]
    pre_s = [(t, "filler") for t in pre]
    if layout < 0.55:
        sents = pre_s + fil_s + core_s
    elif layout < 0.80:
        sents = pre_s + core_s + fil_s
    else:
        sents = pre_s + core_s
        for f in fil_s:
            sents.insert(rng.randint(0, len(sents)), f)
    return sents


CATEGORY_FN = {
    "nli": "nli", "intent": "intent", "state_action": "state_action", "ranking": "ranking",
    "sentiment": "sentiment", "ambiguous": "ambiguous", "agent_gate": "agent_gate",
}


def make_scenario(category: str, lang: str, seed: int) -> Scn:
    rng = random.Random(seed)
    L = lang
    if category == "nli":
        core, q, cands, gold = scn_nli(rng, L)
        q_kind = "nli"
    elif category == "intent":
        core, q, cands, gold = scn_intent(rng, L)
        q_kind = "intent"
    elif category == "ranking":
        core, q, cands, gold = scn_ranking(rng, L)
        q_kind = "rank"
    elif category == "sentiment":
        core, q, cands, gold = scn_sentiment(rng, L)
        q_kind = "sent"
    elif category == "state_action":
        core, cands, gold, q_kind = _core_state_action(rng, L)
        q = _pick(rng, QUESTIONS[q_kind][L])
    elif category == "agent_gate":
        core, cands, gold, q_kind = _core_agent_gate(rng, L)
        q = _pick(rng, QUESTIONS[q_kind][L])
    elif category == "ambiguous":
        core, q, cands, q_kind = scn_ambiguous(rng, L)
        gold = None
        if q_kind != "state":
            pass
    else:
        raise ValueError(category)
    cands, gold = _shuffle_cands(rng, cands, gold)
    # ranking 系は候補文字列（数字を含む）が長く token を食うので context 上限を絞る
    sents = _make_context_sents(rng, L, core, category, hi_scale=0.45 if q_kind == "rank" else 1.0, core_last=q_kind == "rank")
    sep = "\n" if rng.random() < 0.2 else ("" if L == "ja" else " ")
    return Scn(category, L, seed, sents, sep, q, q_kind, cands, gold)


# ---------------------------------------------------------------------------
# 4. 描画
# ---------------------------------------------------------------------------


def render_cand(spec: CandSpec, lang: str, tier: str, rng: random.Random) -> str:
    if spec.role == "rank_opt":
        s = spec.slots
        dom = RANK_DOMAINS[s["dom"]]
        pool = dom[lang]["unseen" if tier == "unseen" else "train"]
        noun = pool[s["ni"] % len(pool)]
        lv = dom["lv"][lang]["para" if tier == "para" else "train"]
        parts = [lv[a][s[a]] for a in ("price", "time", "rating")]
        return f"{noun}：{'・'.join(parts)}" if lang == "ja" else f"{noun}: {', '.join(parts)}"
    role = ROLES[spec.role][lang][tier]
    text = _pick(rng, role)
    if spec.role == "use_tool":
        names = TOOLS[spec.slots["kind"]][lang]
        return text.format(tool=names[1] if tier == "unseen" else names[0])
    if spec.role == "do":
        return text.format(x=spec.slots["x"])
    return text


def render_candidates(scn: Scn, tier: str = "train", order: list[int] | None = None) -> list[str]:
    out = []
    for i, spec in enumerate(scn.cands):
        rng = random.Random(f"{scn.seed}:{scn.lang}:cand:{i}:{tier}")
        out.append(render_cand(spec, scn.lang, tier, rng))
    # 同一 item 内の重複を避ける（同 role の再描画）
    seen = set()
    for i, t in enumerate(out):
        if t in seen:
            for j in range(1, 20):
                rng = random.Random(f"{scn.seed}:{scn.lang}:cand:{i}:{tier}:{j}")
                t2 = render_cand(scn.cands[i], scn.lang, tier, rng)
                if t2 not in seen:
                    out[i] = t2
                    t = t2
                    break
        seen.add(t)
    if order is not None:
        out = [out[i] for i in order]
    return out


def _apply_syn(L: str, text: str) -> str:
    for a, b in SYN[L]:
        text = text.replace(a, b)
    return text


def render_context(scn: Scn, mode: str = "base", rng: random.Random | None = None) -> str:
    sents = list(scn.sents)
    L = scn.lang
    if mode == "ctx_paraphrase":
        new = [(_apply_syn(L, t) if k == "core" else t, k) for t, k in sents]
        if [t for t, _ in new] == [t for t, _ in sents]:
            new = [(SYN_PREFIX[L], "filler")] + new
        # filler の順序を入れ替える（core の相対順序は保つ）
        idx = [i for i, (_, k) in enumerate(new) if k == "filler"]
        if len(idx) >= 2 and rng is not None:
            fs = [new[i] for i in idx]
            rng.shuffle(fs)
            for i, f in zip(idx, fs):
                new[i] = f
        sents = new
    elif mode == "irrelevant_ctx":
        assert rng is not None
        pool = list(IRRELEVANT[L])
        rng.shuffle(pool)
        for t in pool[: rng.randint(1, 3)]:
            sents.insert(rng.randint(0, len(sents)), (t, "filler"))
    return scn.sep.join(t for t, _ in sents)


def render(scn: Scn, variant: str | None = None, variant_seed: int = 0) -> dict:
    """Scn を {context, question, candidates, gold} にする。variant で robust 描画。"""
    rng = random.Random(f"{scn.seed}:{variant}:{variant_seed}")
    tier = "train"
    order = None
    ctx_mode = "base"
    question = scn.question
    if variant == "perm":
        k = len(scn.cands)
        order = list(range(k))
        for _ in range(20):
            rng.shuffle(order)
            if order != list(range(k)):
                break
        else:
            order = order[1:] + order[:1]
    elif variant == "cand_paraphrase":
        tier = "para"
    elif variant == "unseen_cand":
        tier = "unseen"
    elif variant == "ctx_paraphrase":
        ctx_mode = "ctx_paraphrase"
    elif variant == "irrelevant_ctx":
        ctx_mode = "irrelevant_ctx"
    elif variant == "ambiguous":
        question = _pick(rng, QUESTIONS["vague"][scn.lang])
    elif variant is not None:
        raise ValueError(variant)
    cands = render_candidates(scn, tier, order)
    gold = scn.gold
    if order is not None and gold is not None:
        gold = order.index(gold)
    return {"context": render_context(scn, ctx_mode, rng), "question": question, "candidates": cands, "gold": gold}


# ---------------------------------------------------------------------------
# 5. データセット組み立て
# ---------------------------------------------------------------------------

SPLIT_SEED_BASE = {"train": 0, "val": 10**9, "test": 2 * 10**9}


def category_plan(n: int, rng: random.Random) -> list[str]:
    """比率どおり（端数は最大剰余法）のカテゴリ列を shuffle して返す。"""
    tot = sum(CATEGORY_RATIOS.values())
    quotas = {c: n * v / tot for c, v in CATEGORY_RATIOS.items()}
    base = {c: int(q) for c, q in quotas.items()}
    rest = n - sum(base.values())
    for c in sorted(quotas, key=lambda c: quotas[c] - base[c], reverse=True)[:rest]:
        base[c] += 1
    plan = [c for c, k in base.items() for _ in range(k)]
    rng.shuffle(plan)
    return plan


def _key(r: dict) -> tuple:
    return (r["context"], r["question"], json.dumps(r["candidates"], ensure_ascii=False))


def generate_split(split: str, n: int, master_seed: int = 20261006, start_index: int = 0,
                   seen: set | None = None, keep_scn: bool = False) -> list[dict]:
    """split の通常 item を n 件生成（行 dict のリスト）。start_index で追加生成（train の拡張）に対応。

    gen_seed は SPLIT_SEED_BASE[split] + master_seed*7919 + index（index が決まれば決定的）。
    seen（既存 (context, question, candidates) の集合）と重複した場合は index を進めて引き直す。
    """
    seen = set() if seen is None else seen
    rows: list[dict] = []
    plan_rng = random.Random(f"plan:{master_seed}:{split}:{start_index}")
    plan = category_plan(n, plan_rng)
    lang_rng = random.Random(f"lang:{master_seed}:{split}:{start_index}")
    langs = ["ja" if lang_rng.random() < LANG_RATIOS["ja"] / 100 else "en" for _ in range(n)]
    idx = start_index
    for cat, lang in zip(plan, langs):
        for attempt in range(200):
            seed = SPLIT_SEED_BASE[split] + master_seed * 7919 + idx * 31 + attempt
            scn = make_scenario(cat, lang, seed)
            r = render(scn)
            r.update(split=split, category=cat, lang=lang, variant_of=None, variant=None, gen_seed=seed)
            k = _key(r)
            if k not in seen:
                seen.add(k)
                if keep_scn:
                    r["_scn"] = scn
                rows.append(r)
                break
        else:
            raise RuntimeError("could not generate a unique item")
        idx += 1
    return rows


def make_variants(base_rows: list[dict], first_id_of: Callable[[dict], int], variants=VARIANT_KINDS) -> list[dict]:
    """base_rows（_scn 付き、item_id 済み）から robust variant 行を作る。"""
    out = []
    for b in base_rows:
        scn: Scn = b["_scn"]
        for v in variants:
            r = render(scn, v, variant_seed=b["item_id"])
            r.update(split="robust", category=b["category"], lang=b["lang"], variant_of=b["item_id"], variant=v, gen_seed=b["gen_seed"])
            out.append(r)
    return out


def build_dataset(n_train: int, n_val: int, n_test: int, n_robust_base: int = 300, master_seed: int = 20261006,
                  first_id: int = 1, seen: set | None = None, train_start: int = 0) -> list[dict]:
    """train/val/test/robust の全 item 行（item_id 付き）を返す。item_id は train→val→test→robust の順に連番。"""
    seen = set() if seen is None else seen
    train = generate_split("train", n_train, master_seed, train_start, seen)
    val = generate_split("val", n_val, master_seed, 0, seen)
    test = generate_split("test", n_test, master_seed, 0, seen, keep_scn=True)
    nid = first_id
    for r in train + val + test:
        r["item_id"] = nid
        nid += 1
    robust: list[dict] = []
    if n_robust_base > 0 and test:
        pick = random.Random(f"robust:{master_seed}")
        base = pick.sample(test, min(n_robust_base, len(test)))
        base.sort(key=lambda r: r["item_id"])
        robust = make_variants(base, lambda r: r["item_id"])
        for r in robust:
            r["item_id"] = nid
            nid += 1
    for r in test:
        r.pop("_scn", None)
    return train + val + test + robust


def main(argv=None) -> int:
    from tb250distill import replay

    ap = argparse.ArgumentParser(description="raw item を生成して replay DB の items へ insert")
    ap.add_argument("--db", default="data/replay.sqlite")
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--robust-base", type=int, default=300, help="test から variant を作る元 item 数（×6 variant）")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--agent-name", default=None, help="テンプレート中の常駐エージェント名（既定: SYNTH_AGENT_NAME 環境変数か Navi）")
    ap.add_argument("--append-train", type=int, default=0,
                    help="既存 DB の train を N 件追加（50k/100k へ拡張）。val/test/robust は生成しない")
    args = ap.parse_args(argv)
    if args.agent_name:
        set_agent_name(args.agent_name)

    conn = replay.connect(args.db)
    existing = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    if args.append_train:
        seen = replay.existing_keys(conn)
        start = conn.execute("SELECT COUNT(*) FROM items WHERE split='train'").fetchone()[0]
        rows = generate_split("train", args.append_train, args.seed, start_index=start, seen=seen)
        nid = replay.max_item_id(conn) + 1
        for r in rows:
            r["item_id"] = nid
            nid += 1
        replay.insert_items(conn, rows)
        print(f"appended {len(rows)} train items (start_index={start}, ids {rows[0]['item_id']}..{rows[-1]['item_id']})")
        return 0
    if existing:
        print(f"items already exist ({existing}); refusing to regenerate. use --append-train to extend.", file=sys.stderr)
        return 2
    rows = build_dataset(args.train, args.val, args.test, args.robust_base, args.seed)
    replay.insert_items(conn, rows)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    by = {}
    for r in rows:
        by.setdefault(r["split"], {}).setdefault(r["category"], 0)
        by[r["split"]][r["category"]] += 1
    print(json.dumps(by, ensure_ascii=False, indent=1))
    h = hashlib.sha256(json.dumps([_key(r) for r in rows], ensure_ascii=False).encode()).hexdigest()[:16]
    print(f"inserted {len(rows)} items, content hash {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
