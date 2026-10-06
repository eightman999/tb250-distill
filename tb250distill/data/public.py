"""公開データセット → replay DB の items への変換（決定的・CLI あり）。

  python -m tb250distill.data.public build  --root data/external --db data/replay.sqlite [--dry-run] [--sources jcqa,jnli,...]
  python -m tb250distill.data.public backup --db data/replay.sqlite --to /mnt/hdd/replay-before-public-<日時>.sqlite

対象 source（items.source）:
  jcqa      JGLUE JCommonsenseQA   選択肢 5 つをそのまま候補。category=commonsense, lang=ja
  jnli      JGLUE JNLI             含意/矛盾/中立の言い回し候補（2〜3 候補）。category=nli, lang=ja
  massive   MASSIVE ja-JP/en-US     intent。gold + 同 scenario 等の hard negative の言い回し 3〜5 候補。category=intent
  wrime     WRIME ver.2            感情（8 感情 + なし）または極性。書き手/読者平均。category=sentiment, lang=ja
  when2call When2Call              ツール呼出し/聞き返し/不可/直接回答の実文を候補。category=agent_gate
  routellm  RouteLLM gpt4_dataset  「小さいローカルモデルで足りるか / 大きいモデルが要るか」の 2 候補。category=routing

候補文言: 分類系は「ラベル名の言い回し」を複数（train 用 / 評価用で別集合）持ち、item ごとにランダムに選ぶ
（val/test は評価用の言い回しだけを使う＝学習で見ていない文言での汎化を測る）。
全関数は決定的（SEED + source + 元 ID から乱数を作る）。IO（ローダ）と変換（純関数）を分けてあり、変換は小さな fixture で単体テストできる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SEED = 20261006
KANA_RE = re.compile(r"[぀-ヿ]")

DEFAULT_CAPS = {
    # source: {split: 上限}（None = 全件）。massive は train/val/test を ja:en = 2:1 で配分する。
    "jcqa": {"train": None, "val": 500, "test": 500},
    "jnli": {"train": 10000, "val": 500, "test": 500},
    "massive": {"train": 12000, "val": 500, "test": 500},
    "wrime": {"train": 8000, "val": 500, "test": 500},
    "when2call": {"train": 10000, "val": 500, "test": 500},
    "routellm": {"train": 8000, "val": 500, "test": 500},
}
SOURCE_ORDER = tuple(DEFAULT_CAPS)


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------

def stable_seed(source: str, key) -> int:
    return int(hashlib.sha1(f"{SEED}:{source}:{key}".encode()).hexdigest()[:8], 16) % (2**31)


def item_rng(source: str, key) -> random.Random:
    return random.Random(stable_seed(source, key))


def sanitize(text: str) -> str:
    """teacher の chat テンプレートを壊す特殊トークン文字列を無害化し、空白を整える。"""
    text = text.replace("<|", "< |").replace("|>", "| >")
    return re.sub(r"[ \t　]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def clip(text: str, n: int, tail: int = 0) -> str:
    """n 文字に収める。tail>0 なら先頭 n-tail-3 + '…' + 末尾 tail（長い依頼の頭と終わりを残す）。"""
    if len(text) <= n:
        return text
    if tail <= 0:
        return text[: n - 1] + "…"
    return text[: n - tail - 1] + "…" + text[-tail:]


def lang_of(*texts: str) -> str:
    return "ja" if any(KANA_RE.search(t) for t in texts) else "en"


def pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def sample_capped(rng: random.Random, rows: list, cap: int | None) -> list:
    if cap is None or len(rows) <= cap:
        return list(rows)
    return rng.sample(rows, cap)


def make_item(source: str, key, split: str, category: str, lang: str, context: str, question: str,
              candidates: list[str], gold: int | None, extra: dict) -> dict:
    return {
        "split": split, "category": category, "lang": lang, "context": context, "question": question,
        "candidates": candidates, "gold": gold, "variant_of": None, "variant": None,
        "gen_seed": stable_seed(source, key), "source": source, "extra": extra,
    }


def phrase_candidates(rng, labels: list[str], gold_label: str | None, phrases: dict, wrapper: str | None = None) -> tuple[list[str], int | None]:
    """ラベル列 labels（順序は後で shuffle）の言い回しを phrases[label] から 1 つずつ選び、(候補, gold index) を返す。"""
    order = list(labels)
    rng.shuffle(order)
    cands = []
    for lab in order:
        txt = pick(rng, phrases[lab])
        cands.append(wrapper.format(d=txt) if wrapper else txt)
    gold = order.index(gold_label) if gold_label is not None and gold_label in order else None
    return cands, gold


# ---------------------------------------------------------------------------
# 言い回し表（train 用 / 評価用は別集合）
# ---------------------------------------------------------------------------

# JCommonsenseQA / JNLI / WRIME / When2Call の質問文
Q_JCQA = ["最も適切な選択肢を選べ。", "次の質問の答えとして正しいものはどれか。", "常識に照らして最も自然な答えはどれか。", "正解を選んでください。"]
Q_JNLI = ["次の文は前提から言えるか：「{h}」", "「{h}」と前提の関係はどれか。", "「{h}」という文は前提に対してどういう関係か。", "前提を踏まえて、「{h}」はどう判断できるか。"]
Q_MASSIVE = {
    "ja": ["この発話はどの機能への依頼か。", "ユーザーは何を求めているか。", "ユーザーの意図に最も近いものを選べ。", "この発言の意図は何か。"],
    "en": ["What does the user want?", "Which request does this utterance express?", "Pick the closest match to the user's intent.", "What is the intent of this utterance?"],
}
Q_WRIME = {
    ("emotion", "writer"): ["この投稿を書いた人は、どんな気持ちだったか。", "書き手の感情として最も近いものはどれか。"],
    ("emotion", "reader"): ["この投稿を読んだ人は、どんな気持ちになりやすいか。", "読み手が最も強く感じる感情はどれか。"],
    ("polarity", "writer"): ["この投稿を書いた人の気分の向きはどれか。", "書き手の感情の極性はどれか。"],
    ("polarity", "reader"): ["この投稿を読んだ人の気分の向きはどれか。", "読み手が受ける印象の極性はどれか。"],
}
Q_W2C = ["What should the assistant do next?", "Which response is the best next step?", "How should the assistant respond to the user's request?", "Choose the most appropriate reply."]
Q_ROUTE = {
    "en": ["Can a small local model handle this request, or is a larger model needed?", "Which model should answer this request?", "Route this request: small local model or large model?"],
    "ja": ["この依頼は小さいローカルモデルで足りるか、大きいモデルが必要か。", "この依頼にはどのモデルで答えるべきか。", "この依頼を小型モデルと大型モデルのどちらへ回すか。"],
}

# JNLI: 候補は前提との関係。(train, eval)
JNLI_PHRASES = {
    "entailment": (
        ["前提から言える", "前提の内容に含まれる", "前提が正しければ必ず正しい", "前提と同じ事実を述べている", "前提から自然に導かれる"],
        ["前提を言い換えた内容になっている", "前提を認めるなら受け入れるしかない"]),
    "contradiction": (
        ["前提と矛盾する", "前提と食い違っている", "前提が正しければ誤りになる", "前提とは反対のことを述べている", "前提と両立しない"],
        ["前提の内容を否定している", "前提と同時には成り立たない"]),
    "neutral": (
        ["前提だけでは判断できない", "前提からは真偽が決まらない", "前提と無関係ではないが断定できない", "前提に書かれていない情報を含む", "前提からは言えるとも言えないとも決められない"],
        ["前提に情報が足りず決められない", "前提に照らして正否が不明だ"]),
}

# MASSIVE intent: (ja train×2, ja eval, en train×2, en eval)。ja/en とも「…する」「verb phrase」の形でラッパーに埋める。
_I = {
    "datetime_query": ("日付や時刻を調べる|今の時間を確認する|日時について問い合わせる", "check the date or time|look up what time it is|ask about the current date and time"),
    "iot_hue_lightchange": ("照明の色を変える|ライトのカラーを変更する|部屋のあかりの色合いを切り替える", "change the light color|switch the lamp to another color|adjust the hue of the room lights"),
    "transport_ticket": ("乗車券を予約する|チケットを手配する|切符の購入を頼む", "book a transit ticket|buy a train ticket|arrange a ticket for the trip"),
    "takeaway_query": ("テイクアウトの注文状況を確認する|持ち帰り注文の状態を調べる|出前の進み具合を尋ねる", "check the status of a takeaway order|look up my takeout order|ask where my food delivery is"),
    "qa_stock": ("株価を調べる|銘柄の値段を確認する|株式相場について尋ねる", "look up a stock price|check how a share is trading|ask about a company's stock quote"),
    "general_greet": ("あいさつをする|声をかけて挨拶する|こんにちはと呼びかける", "say hello|greet the assistant|open with a friendly greeting"),
    "recommendation_events": ("イベントを勧めてもらう|行事やイベントの情報を求める|近くの催し物を探してもらう", "get an event recommendation|ask for events to attend|find something happening nearby"),
    "music_dislikeness": ("曲が嫌いだと伝える|この音楽は好みではないと表明する|流れている曲に不満を言う", "say a song is disliked|tell the assistant I dislike this music|complain about the track playing"),
    "iot_wemo_off": ("スマートプラグをオフにする|コンセントの電源を切る|プラグの給電を止める", "turn the smart plug off|switch the outlet off|cut power to the plug"),
    "cooking_recipe": ("料理のレシピを尋ねる|作り方を教えてもらう|調理手順を調べる", "ask for a recipe|find out how to cook a dish|look up cooking instructions"),
    "qa_currency": ("通貨の換算レートを調べる|為替を計算する|お金を別の通貨に換算する", "convert between currencies|check an exchange rate|ask what an amount is worth in another currency"),
    "transport_traffic": ("道路の渋滞状況を確認する|交通情報を調べる|混雑具合を尋ねる", "check the traffic conditions|ask about road congestion|get a traffic update"),
    "general_quirky": ("雑談や変わった質問をする|とりとめのない話題を振る|ふざけた問いかけをする", "make small talk or an offbeat remark|ask a quirky question|chat about something random"),
    "weather_query": ("天気を尋ねる|天気予報を確認する|気象状況を調べる", "ask about the weather|check the forecast|find out the weather conditions"),
    "audio_volume_up": ("音量を上げる|音を大きくする|ボリュームを増やす", "turn the volume up|make it louder|raise the sound level"),
    "email_addcontact": ("連絡先を追加する|メールの宛先に新しい人を登録する|アドレス帳に相手を加える", "add a new email contact|save someone to the contacts|put an address into the address book"),
    "takeaway_order": ("料理を注文する|持ち帰りの食事を頼む|出前を発注する", "order takeaway food|place a takeout order|have a meal delivered"),
    "email_querycontact": ("連絡先の情報を調べる|相手のメールアドレスを尋ねる|アドレス帳から探す", "look up a contact's details|ask for someone's email address|search the address book"),
    "iot_hue_lightup": ("照明を明るくする|ライトの明るさを上げる|部屋をもっと明るくする", "brighten the lights|turn the light brightness up|make the room lighter"),
    "recommendation_locations": ("おすすめの場所を教えてもらう|行く場所の候補を尋ねる|近くのお店や名所を探してもらう", "ask for place recommendations|find a good location nearby|get suggestions of where to go"),
    "play_audiobook": ("オーディオブックを再生する|朗読を聴く|本の音声版を流す", "play an audiobook|listen to a book being read|start the audio version of a book"),
    "lists_createoradd": ("リストを作成したり項目を追加したりする|やることリストに書き足す|買い物リストに品物を加える", "create a list or add an item|put something on my to-do list|add an entry to a list"),
    "news_query": ("ニュースを尋ねる|最新の出来事を調べる|報道の内容を知りたいと伝える", "ask for the news|check the latest headlines|find out what is happening in the news"),
    "alarm_query": ("設定済みのアラームを確認する|目覚ましの予定を調べる|何時にアラームがあるか尋ねる", "check the alarms I have set|ask when the alarm rings|list my existing alarms"),
    "iot_wemo_on": ("スマートプラグをオンにする|コンセントの電源を入れる|プラグに給電する", "turn the smart plug on|switch the outlet on|power up the plug"),
    "general_joke": ("冗談を言ってもらう|笑える話をせがむ|ジョークをリクエストする", "ask for a joke|request something funny|have the assistant tell a joke"),
    "qa_definition": ("言葉の意味を尋ねる|用語の定義を調べる|単語の説明を求める", "ask for a definition|look up what a word means|request an explanation of a term"),
    "social_query": ("SNSの投稿や通知を確認する|ソーシャルメディアの状況を調べる|フォロワーの反応を尋ねる", "check social media updates|ask about my social feed|see what is new on social media"),
    "music_settings": ("音楽の再生設定を変える|リピートやシャッフルを設定する|プレーヤーの設定を調整する", "change music playback settings|set repeat or shuffle|adjust the music player options"),
    "audio_volume_other": ("音量を指定の値にする|ボリュームを特定のレベルへ合わせる|音量を細かく調整する", "set the volume to a specific level|adjust the volume in some other way|change the sound level to a given value"),
    "calendar_remove": ("予定を削除する|カレンダーの用事を取り消す|スケジュールから外す", "delete a calendar event|cancel an appointment|remove an item from my schedule"),
    "iot_hue_lightdim": ("照明を暗くする|ライトを薄暗くする|部屋の明かりを落とす", "dim the lights|lower the light brightness|make the room darker"),
    "calendar_query": ("予定を確認する|スケジュールを調べる|いつ何があるか尋ねる", "check my calendar|look up my schedule|ask what events I have"),
    "email_sendemail": ("メールを送る|メッセージを送信する|相手に電子メールを書いて出す", "send an email|write and send a message|email someone"),
    "iot_cleaning": ("ロボット掃除機で掃除させる|掃除を開始する|部屋の掃除を頼む", "start the vacuum robot|have the house cleaned|tell the cleaner to begin"),
    "audio_volume_down": ("音量を下げる|音を小さくする|ボリュームを絞る", "turn the volume down|make it quieter|lower the sound level"),
    "play_radio": ("ラジオを聴く|放送局を流す|ラジオ番組をかける", "play the radio|tune in to a station|put on a radio show"),
    "cooking_query": ("料理に関する質問をする|調理について尋ねる|食材や火加減について聞く", "ask a cooking question|inquire about preparing food|ask about ingredients or cooking times"),
    "datetime_convert": ("時刻を別のタイムゾーンに換算する|時差を計算する|日時の表記を変換する", "convert a time between time zones|work out the time difference|translate a date or time"),
    "qa_maths": ("計算問題を解く|数学の問いに答えてもらう|数式の結果を求める", "solve a math problem|ask the assistant to calculate|work out an arithmetic result"),
    "iot_hue_lightoff": ("照明を消す|ライトをオフにする|部屋の明かりを切る", "turn the lights off|switch the lamp off|kill the room lights"),
    "iot_hue_lighton": ("照明をつける|ライトをオンにする|部屋の明かりを点ける", "turn the lights on|switch the lamp on|light up the room"),
    "transport_query": ("交通手段や経路について尋ねる|移動方法を調べる|行き方を確認する", "ask about transport or directions|find out how to get somewhere|check travel options"),
    "music_likeness": ("曲が気に入ったと伝える|この音楽が好きだと表明する|流れている曲を褒める", "say a song is liked|tell the assistant I love this music|praise the track playing"),
    "email_query": ("受信メールを確認する|メールの内容を調べる|新着のメッセージがあるか尋ねる", "check my emails|look for new messages|ask about my inbox"),
    "play_music": ("音楽を再生する|曲をかける|好みの楽曲を流す", "play some music|put a song on|start playing a track"),
    "audio_volume_mute": ("音を消す|ミュートにする|音声を無音にする", "mute the sound|silence the audio|turn the sound off"),
    "social_post": ("SNSに投稿する|ソーシャルメディアに書き込む|つぶやきを発信する", "post on social media|publish a status update|share something online"),
    "alarm_set": ("アラームを設定する|目覚ましをセットする|指定の時刻に鳴らす準備をする", "set an alarm|schedule a wake-up call|have an alarm ring at a given time"),
    "qa_factoid": ("事実について質問する|豆知識や一般的な情報を尋ねる|ものごとの答えを教えてもらう", "ask a factual question|look up a general fact|ask for a piece of trivia"),
    "calendar_set": ("予定を登録する|カレンダーに用事を入れる|スケジュールに追加する", "add a calendar event|schedule an appointment|put something on my calendar"),
    "play_game": ("ゲームを始める|ゲームで遊ぶ|ゲームを起動する", "play a game|start a game|launch a game"),
    "alarm_remove": ("アラームを削除する|目覚ましを取り消す|設定した警報を解除する", "delete an alarm|cancel a wake-up alarm|turn off a scheduled alarm"),
    "lists_remove": ("リストから項目を消す|やることリストから外す|買い物リストの品を削除する", "remove an item from a list|strike something off my to-do list|delete an entry from a list"),
    "transport_taxi": ("タクシーを呼ぶ|配車を依頼する|車を手配して迎えに来てもらう", "call a taxi|book a ride|order a car to pick me up"),
    "recommendation_movies": ("おすすめの映画を教えてもらう|観る映画の候補を尋ねる|映画を選んでもらう", "ask for movie recommendations|find a film to watch|get suggestions for a movie"),
    "iot_coffee": ("コーヒーメーカーでコーヒーを淹れる|コーヒーを作らせる|一杯のコーヒーを用意させる", "make a coffee|brew coffee with the machine|start the coffee maker"),
    "music_query": ("今流れている曲について尋ねる|音楽の情報を調べる|曲名やアーティストを確認する", "ask about the current song|look up music information|check the title or artist of a track"),
    "play_podcasts": ("ポッドキャストを再生する|番組の配信を聴く|ポッドキャストの最新話を流す", "play a podcast|listen to a podcast episode|start the latest episode of a show"),
    "lists_query": ("リストの中身を確認する|やることリストを読み上げてもらう|買い物リストに何があるか尋ねる", "check the contents of a list|read out my to-do list|ask what is on my list"),
}
MASSIVE_INTENT_PHRASES: dict[str, dict] = {}
for _name, (_ja, _en) in _I.items():
    _j, _e = _ja.split("|"), _en.split("|")
    MASSIVE_INTENT_PHRASES[_name] = {"ja": (_j[:2], _j[2:]), "en": (_e[:2], _e[2:])}
del _name, _ja, _en, _j, _e

MASSIVE_WRAP = {
    "ja": (["{d}", "{d}こと", "{d}依頼", "{d}ための操作", "要するに、{d}"], ["{d}要求", "ユーザーの意図は、{d}こと"]),
    "en": (["{d}", "wants to {d}", "the user wants to {d}", "a request to {d}", "the goal is to {d}"], ["asks the assistant to {d}", "intent: {d}"]),
}

# index 順（HF AmazonScience/massive の ClassLabel 名。parquet メタデータと一致することを load 時に検証する）
MASSIVE_SCENARIOS = ['social', 'transport', 'calendar', 'play', 'news', 'datetime', 'recommendation', 'email', 'iot', 'general', 'audio', 'lists', 'qa', 'cooking', 'takeaway', 'music', 'alarm', 'weather']
MASSIVE_INTENTS = ['datetime_query', 'iot_hue_lightchange', 'transport_ticket', 'takeaway_query', 'qa_stock', 'general_greet', 'recommendation_events', 'music_dislikeness', 'iot_wemo_off', 'cooking_recipe', 'qa_currency', 'transport_traffic', 'general_quirky', 'weather_query', 'audio_volume_up', 'email_addcontact', 'takeaway_order', 'email_querycontact', 'iot_hue_lightup', 'recommendation_locations', 'play_audiobook', 'lists_createoradd', 'news_query', 'alarm_query', 'iot_wemo_on', 'general_joke', 'qa_definition', 'social_query', 'music_settings', 'audio_volume_other', 'calendar_remove', 'iot_hue_lightdim', 'calendar_query', 'email_sendemail', 'iot_cleaning', 'audio_volume_down', 'play_radio', 'cooking_query', 'datetime_convert', 'qa_maths', 'iot_hue_lightoff', 'iot_hue_lighton', 'transport_query', 'music_likeness', 'email_query', 'play_music', 'audio_volume_mute', 'social_post', 'alarm_set', 'qa_factoid', 'calendar_set', 'play_game', 'alarm_remove', 'lists_remove', 'transport_taxi', 'recommendation_movies', 'iot_coffee', 'music_query', 'play_podcasts', 'lists_query']
assert set(MASSIVE_INTENTS) == set(MASSIVE_INTENT_PHRASES) and len(MASSIVE_INTENTS) == 60

# WRIME
EMOTIONS = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"]
EMOTION_COLS = ["Joy", "Sadness", "Anticipation", "Surprise", "Anger", "Fear", "Disgust", "Trust"]
EMOTION_OPPOSITE = {"joy": "sadness", "sadness": "joy", "trust": "disgust", "disgust": "trust",
                    "anticipation": "surprise", "surprise": "anticipation", "anger": "fear", "fear": "anger"}
EMOTION_PHRASES = {
    "joy": (["喜んでいる", "うれしく楽しい気持ち", "ハッピーで満たされている", "楽しくてうきうきしている"], ["幸福感に浸っている", "ほっこり嬉しい"]),
    "sadness": (["悲しんでいる", "沈んだ寂しい気持ち", "落ち込んでいる", "切なくて涙が出そう"], ["哀愁を帯びた気分", "胸が痛むほど悲しい"]),
    "anticipation": (["楽しみにしている", "わくわくして待っている", "期待を膨らませている", "これからに期待している"], ["心待ちにしている", "先の展開にうずうずしている"]),
    "surprise": (["驚いている", "びっくりしている", "意外で戸惑っている", "予想外のことに目を丸くしている"], ["思いがけず仰天している", "不意を突かれて面食らっている"]),
    "anger": (["怒っている", "腹を立てている", "苛立っている", "憤りを感じている"], ["かっとなっている", "不満が爆発しそうだ"]),
    "fear": (["怖がっている", "不安で恐れている", "心配でびくびくしている", "危機感を抱いている"], ["恐怖を覚えている", "心細くて怯えている"]),
    "disgust": (["嫌悪感を抱いている", "うんざりしている", "不快に思っている", "嫌でたまらない"], ["吐き気がするほど嫌だ", "げんなりしている"]),
    "trust": (["信頼している", "安心して頼りにしている", "信じて任せている", "親しみと信頼を感じている"], ["全幅の信頼を寄せている", "心を許している"]),
    "none": (["特に強い感情は見られない", "感情の動きはほとんどない", "平静で淡々とした気持ち", "これといった感情は抱いていない"], ["感情の起伏が感じられない", "ごく平坦な心の状態"]),
}
POLARITY_PHRASES = {
    "positive": (["ポジティブな内容", "前向きで明るい気分", "好意的で肯定的な気持ち", "良い印象を受ける内容"], ["晴れやかな気持ちが伝わる", "プラスの感情が表れている"]),
    "negative": (["ネガティブな内容", "暗く沈んだ否定的な気持ち", "不快で否定的な感情", "悪い印象を受ける内容"], ["曇った気持ちが伝わる", "マイナスの感情が表れている"]),
    "neutral": (["どちらでもない中立的な内容", "感情の偏りがなく淡々としている", "良くも悪くもない", "ポジティブとネガティブのどちらとも言えない"], ["感情をあまり含まない事実の描写", "フラットで落ち着いた調子"]),
}

# RouteLLM: 候補（local = 小さいローカルモデルで足りる / large = 大きいモデルが要る）。(train, eval)
ROUTE_PHRASES = {
    "en": {
        "local": (["a small local model is good enough for this", "handle it with the lightweight on-device model", "route it to the cheap small model",
                   "no need for a big model; keep it local", "the compact model can answer this well", "keep this on the small model"],
                  ["this is simple enough for a modest model", "a basic model should manage this request"]),
        "large": (["this needs a large, high-capability model", "send it to the strongest model available", "escalate to the big model; the small one would struggle",
                   "only a powerful model can handle this well", "route it to the expensive large model", "a heavyweight model is required for a good answer"],
                  ["this is too demanding for a small model", "a more capable model is needed to respond properly"]),
    },
    "ja": {
        "local": (["小さなローカルモデルで十分に答えられる", "軽量な手元のモデルで処理してよい", "大きなモデルを使うまでもない",
                   "安価な小型モデルに任せて問題ない", "オンデバイスのモデルだけで対応できる", "このまま小さいモデルで回答する"],
                  ["簡単な依頼なので小型のモデルで足りる", "高性能でなくても対応できる内容だ"]),
        "large": (["高性能な大型モデルが必要", "大きなモデルへ回して回答させる", "小さいモデルでは荷が重いので上位モデルに任せる",
                   "強力なモデルでないと適切に答えられない", "コストをかけても大規模モデルを使うべき", "重い推論なので大型モデルにエスカレーションする"],
                  ["小型モデルには難しすぎる依頼だ", "より能力の高いモデルで応答する必要がある"]),
    },
}
ROUTE_LOCAL_MIN_SCORE = 4  # mixtral_score >= 4 なら「小さいモデルで足りる」（仮定。生の score は extra に保存）


def tier(split: str) -> int:
    """0 = 学習用の言い回し、1 = 評価（val/test）用の言い回し。"""
    return 0 if split == "train" else 1


# ---------------------------------------------------------------------------
# ローダ（IO）。変換関数は dict の行を受け取る。
# ---------------------------------------------------------------------------

def load_jsonl(path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_massive(path, locale: str) -> list[dict]:
    """MASSIVE parquet → {id, locale, partition, scenario(名), intent(名), utt}。ClassLabel の名前は定数表と突き合わせる。"""
    import pyarrow.parquet as pq

    t = pq.read_table(str(path), columns=["id", "locale", "partition", "scenario", "intent", "utt"])
    meta = t.schema.metadata or {}
    if b"huggingface" in meta:
        feats = json.loads(meta[b"huggingface"])["info"]["features"]
        assert feats["intent"]["names"] == MASSIVE_INTENTS, "MASSIVE intent 名が定数表と不一致"
        assert feats["scenario"]["names"] == MASSIVE_SCENARIOS, "MASSIVE scenario 名が定数表と不一致"
    rows = []
    for r in t.to_pylist():
        rows.append({"id": r["id"], "locale": r["locale"] or locale, "partition": r["partition"],
                     "scenario": MASSIVE_SCENARIOS[r["scenario"]], "intent": MASSIVE_INTENTS[r["intent"]], "utt": r["utt"]})
    return rows


def load_wrime(path) -> list[dict]:
    import csv

    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for i, r in enumerate(rows):
        r["_idx"] = i
    return rows


# ---------------------------------------------------------------------------
# 変換（純関数）。None を返した行は捨てる。
# ---------------------------------------------------------------------------

def conv_jcqa(row: dict, split: str, orig_split: str) -> dict | None:
    cands = [str(row[f"choice{i}"]).strip() for i in range(5)]
    q = str(row["question"]).strip()
    if not q or not all(cands) or len(set(cands)) != 5 or not 0 <= int(row["label"]) < 5:
        return None
    rng = item_rng("jcqa", row["q_id"])
    return make_item("jcqa", row["q_id"], split, "commonsense", "ja", sanitize(q), pick(rng, Q_JCQA), cands, int(row["label"]),
                     {"orig_id": row["q_id"], "orig_split": orig_split, "label": int(row["label"])})


def conv_jnli(row: dict, split: str, orig_split: str) -> dict | None:
    s1, s2, lab = str(row["sentence1"]).strip(), str(row["sentence2"]).strip(), row["label"]
    if not s1 or not s2 or lab not in JNLI_PHRASES:
        return None
    rng = item_rng("jnli", row["sentence_pair_id"])
    others = [x for x in JNLI_PHRASES if x != lab]
    if rng.random() < 0.5:
        labels = [lab] + others          # 3 候補
    else:
        labels = [lab, pick(rng, others)]  # 2 候補（gold + 1 つ）
    phr = {k: v[tier(split)] for k, v in JNLI_PHRASES.items()}
    cands, gold = phrase_candidates(rng, labels, lab, phr)
    return make_item("jnli", row["sentence_pair_id"], split, "nli", "ja", sanitize(s1), pick(rng, Q_JNLI).format(h=s2), cands, gold,
                     {"orig_id": row["sentence_pair_id"], "yjcaptions_id": row.get("yjcaptions_id"), "orig_split": orig_split,
                      "label": lab, "labels_shown": labels})


def conv_massive(row: dict, split: str, orig_split: str) -> dict | None:
    utt = sanitize(str(row["utt"]))
    gold_i = row["intent"]
    if not utt or gold_i not in MASSIVE_INTENT_PHRASES:
        return None
    lang = "ja" if str(row["locale"]).startswith("ja") else "en"
    rng = item_rng("massive", f"{row['locale']}:{row['id']}")
    k = rng.choices([3, 4, 5], weights=[35, 35, 30])[0]
    scen = row["scenario"]
    same = [i for i in MASSIVE_INTENTS if i != gold_i and _scenario_of(i) == _scenario_of(gold_i)]
    rng.shuffle(same)
    negs = same[: min(len(same), rng.randint(1, k - 1))]   # 同 scenario の hard negative を 1 個以上（あれば）
    rest = [i for i in MASSIVE_INTENTS if i != gold_i and i not in same]
    rng.shuffle(rest)
    negs += rest[: k - 1 - len(negs)]
    labels = [gold_i] + negs
    phr = {i: MASSIVE_INTENT_PHRASES[i][lang][tier(split)] for i in labels}
    wrap = pick(rng, MASSIVE_WRAP[lang][tier(split)])
    cands, gold = phrase_candidates(rng, labels, gold_i, phr, wrapper=wrap)
    return make_item("massive", f"{row['locale']}:{row['id']}", split, "intent", lang, utt, pick(rng, Q_MASSIVE[lang]), cands, gold,
                     {"orig_id": row["id"], "locale": row["locale"], "orig_split": orig_split, "scenario": scen, "intent": gold_i,
                      "intents_shown": labels})


def _scenario_of(intent: str) -> str:
    """intent 名の接頭辞（alarm_set → alarm）。MASSIVE の intent は全て <scenario>_<name> 形式で、接頭辞 = scenario。"""
    return intent.split("_")[0]


def _f(x) -> float:
    return float(x)


def wrime_vectors(row: dict) -> dict:
    writer = [_f(row[f"Writer_{c}"]) for c in EMOTION_COLS]
    readers = [[_f(row[f"Reader{j}_{c}"]) for c in EMOTION_COLS] for j in (1, 2, 3)]
    reader_avg = [sum(r[i] for r in readers) / 3.0 for i in range(8)]
    r_sents = [_f(row[f"Reader{j}_Sentiment"]) for j in (1, 2, 3)]
    return {"writer": writer, "reader_avg": reader_avg, "writer_sent": _f(row["Writer_Sentiment"]),
            "reader_sent": r_sents, "reader_sent_avg": sum(r_sents) / 3.0}


def _soft(v: list[float]) -> list[float]:
    s = sum(v)
    return [round(x / s, 4) for x in v] if s > 0 else [round(1 / len(v), 4)] * len(v)


def conv_wrime(row: dict, split: str, orig_split: str) -> dict | None:
    text = sanitize(str(row["Sentence"]))
    if len(text) < 2:
        return None
    rng = item_rng("wrime", row["_idx"])
    vec = wrime_vectors(row)
    task = "emotion" if rng.random() < 0.6 else "polarity"
    subject = "writer" if rng.random() < 0.5 else "reader"
    extra = {"orig_idx": row["_idx"], "orig_split": orig_split, "user_id": row.get("UserID"), "task": task, "subject": subject,
             "emotions": EMOTIONS, "writer": vec["writer"], "reader_avg": [round(x, 3) for x in vec["reader_avg"]],
             "writer_sent": vec["writer_sent"], "reader_sent": vec["reader_sent"], "reader_sent_avg": round(vec["reader_sent_avg"], 3)}
    q = pick(rng, Q_WRIME[(task, subject)])
    t = tier(split)
    if task == "polarity":
        s = vec["writer_sent"] if subject == "writer" else vec["reader_sent_avg"]
        lab = "positive" if s >= 0.5 else "negative" if s <= -0.5 else "neutral"
        others = [x for x in POLARITY_PHRASES if x != lab]
        labels = [lab] + (others if rng.random() < 0.5 else [pick(rng, others)])
        phr = {kk: v[t] for kk, v in POLARITY_PHRASES.items()}
        cands, gold = phrase_candidates(rng, labels, lab, phr)
        if subject == "writer":
            votes = {"positive": float(s > 0), "neutral": float(s == 0), "negative": float(s < 0)}
        else:
            votes = {"positive": sum(x > 0 for x in vec["reader_sent"]), "neutral": sum(x == 0 for x in vec["reader_sent"]),
                     "negative": sum(x < 0 for x in vec["reader_sent"])}
        extra.update({"label": lab, "labels_shown": labels, "soft": _soft([votes["negative"], votes["neutral"], votes["positive"]]), "soft_order": ["negative", "neutral", "positive"]})
        return make_item("wrime", row["_idx"], split, "sentiment", "ja", text, q, cands, gold, extra)
    v = vec["writer"] if subject == "writer" else vec["reader_avg"]
    order = sorted(range(8), key=lambda i: (-v[i], rng.random()))
    top, second = v[order[0]], v[order[1]]
    k = rng.choices([3, 4, 5], weights=[35, 35, 30])[0]
    tied: list[str] = []
    if top < 1.0:                           # 強い感情なし → 'none' が gold、強度の高い感情を hard negative に
        gold_label = "none"
        labels = ["none"] + [EMOTIONS[i] for i in order[: k - 1]]
    else:
        if top - second < 0.34:             # 同点首位 → gold 無し（teacher の soft 分布だけ使う）
            tied = [EMOTIONS[i] for i in order if v[i] >= top - 1e-9]
        if len(tied) >= 2:
            gold_label = None
            labels = tied[:k]
        else:
            gold_label = EMOTIONS[order[0]]
            labels = [gold_label]
            opp = EMOTION_OPPOSITE[gold_label]
            if opp not in labels:
                labels.append(opp)
        if len(labels) < k and "none" not in labels and rng.random() < 0.3:
            labels.append("none")
        for i in order:
            if len(labels) >= k:
                break
            if EMOTIONS[i] not in labels:
                labels.append(EMOTIONS[i])
    phr = {kk: vv[t] for kk, vv in EMOTION_PHRASES.items()}
    cands, gold = phrase_candidates(rng, labels, gold_label, phr)
    extra.update({"label": gold_label if gold_label else ("tie:" + "+".join(tied)), "labels_shown": labels, "soft": _soft(v), "soft_order": EMOTIONS})
    return make_item("wrime", row["_idx"], split, "sentiment", "ja", text, q, cands, gold, extra)


# --- When2Call ---------------------------------------------------------------

def _tool_line(raw: str, desc_max: int = 110) -> str:
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
        name = d.get("name", "?")
        desc = " ".join(str(d.get("description", "")).split())
        params = d.get("parameters", {}) or {}
        props = list((params.get("properties") or {}).keys())
        req = params.get("required") or d.get("required") or []
        plist = ", ".join((p + "*" if p in req else p) for p in props[:6]) + ("…" if len(props) > 6 else "")
        return f"- {name}({plist}): {clip(desc, desc_max)}"
    except Exception:  # noqa: BLE001
        return "- " + clip(" ".join(str(raw).split()), 160)


def w2c_context(tools: list, question: str, max_tools_chars: int = 550, max_q: int = 400) -> str:
    lines = [_tool_line(t) for t in tools]
    block = "\n".join(lines)
    if len(block) > max_tools_chars:
        block = clip(block, max_tools_chars)
    return sanitize(f"Available tools:\n{block}\nUser request: {clip(' '.join(question.split()), max_q)}")


def w2c_candidate(text: str, max_len: int = 240) -> str:
    return clip(sanitize(text), max_len)


def normalize_toolcall(s: str) -> str:
    """MCQ の tool_call（JSON 文字列）を train 側の '<TOOLCALL>[{...}]</TOOLCALL>' 形式に揃える。"""
    t = s.strip()
    if t.startswith("<TOOLCALL>"):
        return t
    try:
        obj = json.loads(t)
        return "<TOOLCALL>" + json.dumps(obj if isinstance(obj, list) else [obj], ensure_ascii=False) + "</TOOLCALL>"
    except Exception:  # noqa: BLE001
        return "<TOOLCALL>[" + t + "]</TOOLCALL>"


_RE_CANNOT = re.compile(r"unable|cannot|can't|can not|apolog|sorry|not able|don't have", re.I)


def guess_type(text: str) -> str:
    """pref 側の応答タイプの簡易推定（ヒューリスティック。extra に参考として残すだけで gold には使わない）。"""
    if text.lstrip().startswith("<TOOLCALL>"):
        return "tool_call"
    if _RE_CANNOT.search(text[:240]):
        return "cannot_answer"
    if "?" in text:
        return "request_for_info"
    return "direct"


def conv_w2c_pref(row: dict, idx: int, split: str, orig_split: str = "train_pref") -> dict | None:
    msgs = row.get("messages") or []
    user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), None)
    chosen = (row.get("chosen_response") or {}).get("content")
    rejected = (row.get("rejected_response") or {}).get("content")
    if not user or not chosen or not rejected:
        return None
    c, r = w2c_candidate(chosen), w2c_candidate(rejected)
    if c == r:
        return None
    rng = item_rng("when2call", f"pref:{idx}")
    cands, types = [c, r], [guess_type(chosen), guess_type(rejected)]
    order = [0, 1]
    rng.shuffle(order)
    cands, types = [cands[i] for i in order], [types[i] for i in order]
    gold = order.index(0)
    ctx = w2c_context(row.get("tools") or [], user)
    return make_item("when2call", f"pref:{idx}", split, "agent_gate", lang_of(user), ctx, pick(rng, Q_W2C), cands, gold,
                     {"origin": "pref", "orig_id": idx, "orig_split": orig_split, "types_guess": types})


W2C_KEYS = ("direct", "tool_call", "request_for_info", "cannot_answer")


def conv_w2c_mcq(row: dict, split: str, orig_split: str = "test_mcq") -> dict | None:
    ans = row.get("answers") or {}
    if not all(ans.get(k) for k in W2C_KEYS) or row.get("correct_answer") not in W2C_KEYS:
        return None
    rng = item_rng("when2call", f"mcq:{row['uuid']}")
    keys = list(W2C_KEYS)
    if rng.random() < 0.3:  # 一部は誤答 1 つを落として 3 候補にする（K を散らす）
        drop = pick(rng, [k for k in keys if k != row["correct_answer"]])
        keys.remove(drop)
    rng.shuffle(keys)
    cands = [w2c_candidate(normalize_toolcall(ans[k]) if k == "tool_call" else ans[k]) for k in keys]
    if len(set(cands)) != len(cands):
        return None
    ctx = w2c_context(row.get("tools") or [], row["question"])
    return make_item("when2call", f"mcq:{row['uuid']}", split, "agent_gate", lang_of(row["question"]), ctx, pick(rng, Q_W2C), cands, keys.index(row["correct_answer"]),
                     {"origin": "mcq", "orig_id": row["uuid"], "source_id": row.get("source_id"), "orig_split": orig_split,
                      "correct_answer": row["correct_answer"], "types": keys})


# --- RouteLLM -----------------------------------------------------------------

def route_context(prompt: str, max_len: int = 700, tail: int = 200) -> str:
    return clip(sanitize(prompt), max_len, tail)


def conv_routellm(row: dict, idx: int, split: str, orig_split: str) -> dict | None:
    prompt = route_context(str(row["prompt"]))
    if len(prompt) < 3:
        return None
    score = int(row["mixtral_score"])
    local = score >= ROUTE_LOCAL_MIN_SCORE
    rng = item_rng("routellm", f"{orig_split}:{idx}")
    ql = "ja" if (KANA_RE.search(prompt) or rng.random() < 0.3) else "en"
    phr = {k: v[tier(split)] for k, v in ROUTE_PHRASES[ql].items()}
    lab = "local" if local else "large"
    cands, gold = phrase_candidates(rng, ["local", "large"], lab, phr)
    return make_item("routellm", f"{orig_split}:{idx}", split, "routing", ql, prompt, pick(rng, Q_ROUTE[ql]), cands, gold,
                     {"orig_idx": idx, "orig_split": orig_split, "mixtral_score": score, "label": lab,
                      "label_rule": f"mixtral_score>={ROUTE_LOCAL_MIN_SCORE} -> local",
                      "prompt_source": row.get("source"), "prompt_chars": len(str(row["prompt"]))})


# ---------------------------------------------------------------------------
# source ごとの組み立て（サンプリング・split 割当）
# ---------------------------------------------------------------------------

def _rng(source: str, tag: str) -> random.Random:
    return random.Random(f"{SEED}:{source}:{tag}")


def convert_all(fn, rows: list[dict], split: str, orig_split: str, cap: int | None, source: str) -> list[dict]:
    """rows から cap 件を決定的にサンプリング → 変換（None は捨てて補充）。"""
    rng = _rng(source, f"{orig_split}->{split}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = []
    for i in order:
        it = fn(rows[i], split, orig_split)
        if it is not None:
            out.append(it)
            if cap is not None and len(out) >= cap:
                break
    return out


def build_jcqa(root: Path, caps: dict) -> list[dict]:
    d = Path(root) / "jglue" / "jcommonsenseqa"
    items = []
    for orig, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        items += convert_all(conv_jcqa, load_jsonl(d / f"{orig}.jsonl"), split, orig, caps.get(split), "jcqa")
    return items


def build_jnli(root: Path, caps: dict) -> list[dict]:
    d = Path(root) / "jglue" / "jnli"
    items = []
    for orig, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        items += convert_all(conv_jnli, load_jsonl(d / f"{orig}.jsonl"), split, orig, caps.get(split), "jnli")
    return items


def build_massive(root: Path, caps: dict) -> list[dict]:
    d = Path(root) / "massive"
    items = []
    for orig, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        cap = caps.get(split)
        ja_cap = None if cap is None else (cap * 2) // 3
        en_cap = None if cap is None else cap - ja_cap
        for loc, c in (("ja-JP", ja_cap), ("en-US", en_cap)):
            items += convert_all(conv_massive, load_massive(d / loc / f"{orig}.parquet", loc), split, orig, c, f"massive_{loc}")
    return items


def build_wrime(root: Path, caps: dict) -> list[dict]:
    rows = load_wrime(Path(root) / "wrime" / "wrime-ver2.tsv")
    by = {}
    for r in rows:
        by.setdefault(r["Train/Dev/Test"], []).append(r)
    items = []
    for orig, split in (("train", "train"), ("dev", "val"), ("test", "test")):
        items += convert_all(conv_wrime, by.get(orig, []), split, orig, caps.get(split), "wrime")
    return items


def build_when2call(root: Path, caps: dict) -> list[dict]:
    d = Path(root) / "when2call"
    pref = load_jsonl(d / "train" / "when2call_train_pref.jsonl")
    mcq = load_jsonl(d / "test" / "when2call_test_mcq.jsonl")
    cap_val, cap_test, cap_train = caps.get("val"), caps.get("test"), caps.get("train")
    half = lambda c: None if c is None else c // 2  # noqa: E731
    # MCQ: source_id（同一ベース問題の変種をまとめる）単位で val/test/残り(train) に割当てる
    groups: dict = {}
    for r in mcq:
        groups.setdefault(r.get("source_id") or r["uuid"], []).append(r)
    gids = sorted(groups, key=lambda g: hashlib.sha1(f"{SEED}:w2c:{g}".encode()).hexdigest())
    mcq_val, mcq_test, mcq_rest = [], [], []
    need_v, need_t = half(cap_val) or 0, half(cap_test) or 0
    for g in gids:
        if len(mcq_val) < need_v:
            mcq_val += groups[g]
        elif len(mcq_test) < need_t:
            mcq_test += groups[g]
        else:
            mcq_rest += groups[g]
    items = []
    items += convert_all(lambda r, s, o: conv_w2c_mcq(r, s, o), mcq_val, "val", "test_mcq", half(cap_val), "when2call_mcq_val")
    items += convert_all(lambda r, s, o: conv_w2c_mcq(r, s, o), mcq_test, "test", "test_mcq", half(cap_test), "when2call_mcq_test")
    # pref: インデックスで val/test/train に分ける
    idx = list(range(len(pref)))
    _rng("when2call", "pref-split").shuffle(idx)
    pv, pt = idx[: half(cap_val) or 0], idx[half(cap_val) or 0: (half(cap_val) or 0) + (half(cap_test) or 0)]
    ptr = idx[(half(cap_val) or 0) + (half(cap_test) or 0):]
    for ids_, split in ((pv, "val"), (pt, "test")):
        for i in ids_:
            it = conv_w2c_pref(pref[i], i, split)
            if it:
                items.append(it)
    train = []
    for i in sorted(ptr):
        it = conv_w2c_pref(pref[i], i, "train")
        if it:
            train.append(it)
    # 全件 or 上限: pref 全件 + MCQ 残り（K=3/4 の分布を足す）を cap まで
    rest = convert_all(lambda r, s, o: conv_w2c_mcq(r, s, o), mcq_rest, "train", "test_mcq_residual", None, "when2call_mcq_train")
    for it in rest:
        it["extra"]["orig_split"] = "test_mcq_residual"
    pool = train
    if cap_train is not None and len(pool) >= cap_train:
        pool = sample_capped(_rng("when2call", "train-cap"), pool, cap_train)
    else:
        room = None if cap_train is None else cap_train - len(pool)
        pool = pool + sample_capped(_rng("when2call", "mcq-rest-cap"), rest, room)
    return items + pool


def build_routellm(root: Path, caps: dict) -> list[dict]:
    d = Path(root) / "routellm"
    items = []
    seen: set = set()  # train と valid をまたいだ prompt 重複も除く
    for orig_file, orig_split, splits in (("train.jsonl", "train", ("train",)), ("valid.jsonl", "valid", ("val", "test"))):
        rows = load_jsonl(d / orig_file)
        pool = []
        for i, r in enumerate(rows):
            p = str(r.get("prompt", "")).strip()
            if len(p) < 3 or p in seen or r.get("mixtral_score") is None:
                continue
            seen.add(p)
            pool.append((i, r))
        small = [x for x in pool if int(x[1]["mixtral_score"]) >= ROUTE_LOCAL_MIN_SCORE]
        large = [x for x in pool if int(x[1]["mixtral_score"]) < ROUTE_LOCAL_MIN_SCORE]
        for split in splits:
            cap = caps.get(split)
            n_large = None if cap is None else int(round(cap * 0.45))
            n_small = None if cap is None else cap - n_large
            rng = _rng("routellm", f"{orig_split}->{split}")
            ls, ss = list(large), list(small)
            rng.shuffle(ls)
            rng.shuffle(ss)
            take = (ls[:n_large] if n_large is not None else ls) + (ss[:n_small] if n_small is not None else ss)
            # 同じ元データから val/test を分けるときは重ならないよう取り除く
            taken = {i for i, _ in take}
            large = [x for x in large if x[0] not in taken]
            small = [x for x in small if x[0] not in taken]
            for i, r in take:
                it = conv_routellm(r, i, split, orig_split)
                if it:
                    items.append(it)
    return items


BUILDERS = {"jcqa": build_jcqa, "jnli": build_jnli, "massive": build_massive, "wrime": build_wrime,
            "when2call": build_when2call, "routellm": build_routellm}


def build_items(root, sources=SOURCE_ORDER, caps: dict | None = None) -> dict[str, list[dict]]:
    caps = {s: dict(DEFAULT_CAPS[s]) for s in DEFAULT_CAPS} if caps is None else caps
    out = {}
    for s in sources:
        out[s] = BUILDERS[s](Path(root), caps.get(s, DEFAULT_CAPS[s]))
    return out


def assign_ids(by_source: dict[str, list[dict]], first_id: int, existing_keys: set | None = None) -> list[dict]:
    """train（source を混ぜて shuffle）→ val → test の順に item_id を振る。
    train を混ぜるのは、途中まで採点した時点で全 source が含まれるようにするため。重複 (context, question, candidates) は捨てる。"""
    seen = set() if existing_keys is None else set(existing_keys)
    rows = []
    for s in by_source:
        for it in by_source[s]:
            k = (it["context"], it["question"], json.dumps(it["candidates"], ensure_ascii=False))
            if k in seen:
                continue
            seen.add(k)
            rows.append(it)
    out = []
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split]
        _rng("all", f"order:{split}").shuffle(rs)
        out += rs
    for i, r in enumerate(out):
        r["item_id"] = first_id + i
    return out


def summarize(rows: list[dict]) -> dict:
    import collections

    res: dict = {}
    for r in rows:
        d = res.setdefault(r["source"], {}).setdefault(r["split"], {"n": 0, "k": collections.Counter(), "gold": 0, "lang": collections.Counter(), "cat": collections.Counter()})
        d["n"] += 1
        d["k"][len(r["candidates"])] += 1
        d["gold"] += r["gold"] is not None
        d["lang"][r["lang"]] += 1
        d["cat"][r["category"]] += 1
    for s in res.values():
        for d in s.values():
            d["k"] = dict(sorted(d["k"].items()))
            d["lang"] = dict(d["lang"])
            d["cat"] = dict(d["cat"])
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def teacher_stats(conn) -> dict:
    """source × split ごとの Teacher の採点済み件数・gold 一致率・平均最大確率・ランダム基準（採点済み分のみ）。"""
    res: dict = {}
    q = ("SELECT i.source AS source, i.split AS split, i.gold AS gold, i.candidates AS cands, t.probs AS probs "
         "FROM items i JOIN teacher t ON t.item_id = i.item_id")
    for r in conn.execute(q):
        probs = json.loads(r["probs"])
        k = len(probs)
        d = res.setdefault(r["source"], {}).setdefault(r["split"], {"scored": 0, "gold": 0, "hit": 0, "maxp": 0.0, "rand": 0.0, "rand_gold": 0.0})
        d["scored"] += 1
        d["maxp"] += max(probs)
        d["rand"] += 1.0 / k
        if r["gold"] is not None:
            d["gold"] += 1
            d["hit"] += int(max(range(k), key=probs.__getitem__) == r["gold"])
            d["rand_gold"] += 1.0 / k
    out: dict = {}
    for src, by in res.items():
        for sp, d in by.items():
            out.setdefault(src, {})[sp] = {
                "scored": d["scored"], "with_gold": d["gold"],
                "teacher_gold_acc": round(d["hit"] / d["gold"], 4) if d["gold"] else None,
                "random_gold_acc": round(d["rand_gold"] / d["gold"], 4) if d["gold"] else None,
                "mean_max_prob": round(d["maxp"] / d["scored"], 4)}
    return out


def update_manifest(root: str, summary: dict) -> None:
    """MANIFEST.json に取り込み件数（source・split 別）と変換方式の要約を書き足す。jcqa/jnli は manifest 上は jglue。"""
    mpath = Path(root) / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    method = {
        "jcqa": "選択肢 5 つをそのまま候補。gold=label。context=問題文、question=定型の指示文",
        "jnli": "含意/矛盾/中立の言い回し（train 5 / 評価 2 種）から 2〜3 候補。gold=label。context=文1、question=文2 を含む定型文",
        "massive": "intent を言い回し（desc×wrapper。train/評価で別集合）に変換。gold + 同 scenario の hard negative + 他 scenario で 3〜5 候補",
        "wrime": "感情(8+なし)または極性。書き手/読者平均。gold=最大強度（同点は gold 無し）、soft 分布を extra に保存",
        "when2call": "pref: chosen/rejected の 2 候補、MCQ: 4（一部 3）選択肢の実文。context=ツール要約+依頼文",
        "routellm": "mixtral_score>=4 → 小さいローカルモデルで足りる、<=3 → 大きいモデルが要る（仮定）。2 候補",
    }
    for src, by_split in summary.items():
        name = "jglue" if src in ("jcqa", "jnli") else src
        ent = manifest.setdefault(name, {})
        ent.setdefault("imported", {})[src] = {sp: v["n"] for sp, v in by_split.items()}
        ent.setdefault("conversion", {})[src] = method[src]
    mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))


def backup_db(src: str, dst: str) -> dict:
    """sqlite3 の online backup API で一貫したコピーを作る（WAL 稼働中でも可）。件数を突き合わせて返す。"""
    import sqlite3

    if Path(dst).exists():
        raise SystemExit(f"backup target exists: {dst}")
    a = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    b = sqlite3.connect(dst)
    a.backup(b)
    res = {}
    for t in ("items", "teacher"):
        res[t] = (a.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], b.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    res["integrity"] = b.execute("PRAGMA integrity_check").fetchone()[0]
    a.close()
    b.close()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="data/external → items（既存 DB には追記のみ）")
    b.add_argument("--root", default="data/external")
    b.add_argument("--db", default="data/replay.sqlite")
    b.add_argument("--sources", default=",".join(SOURCE_ORDER))
    b.add_argument("--caps", default="", help="例 jnli.train=5000,wrime.val=300（DEFAULT_CAPS を上書き。'all' で無制限）")
    b.add_argument("--dry-run", action="store_true", help="DB へ書かず統計と例だけ出す")
    b.add_argument("--report", default="", help="統計 JSON の出力先")
    b.add_argument("--preview", default="", help="先頭数件ずつの JSONL（目視確認用）の出力先")
    st = sub.add_parser("stats", help="source×split ごとの Teacher gold 一致率（採点済み分）")
    st.add_argument("--db", default="data/replay.sqlite")
    k = sub.add_parser("backup", help="DB のバックアップ")
    k.add_argument("--db", default="data/replay.sqlite")
    k.add_argument("--to", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "stats":
        from tb250distill import replay

        conn = replay.connect(args.db, readonly=True)
        print(json.dumps(teacher_stats(conn), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "backup":
        print(json.dumps(backup_db(args.db, args.to)))
        return 0

    caps = {s: dict(v) for s, v in DEFAULT_CAPS.items()}
    for part in [p for p in args.caps.split(",") if p]:
        key, _, val = part.partition("=")
        s, _, sp = key.partition(".")
        caps[s][sp] = None if val == "all" else int(val)
    sources = [s for s in args.sources.split(",") if s]
    by_source = build_items(args.root, sources, caps)

    from tb250distill import replay

    if args.dry_run:
        conn = replay.connect(args.db, readonly=True) if Path(args.db).exists() else None
        keys = replay.existing_keys(conn) if conn is not None else set()
        first = (replay.max_item_id(conn) + 1) if conn is not None else 1
    else:
        conn = replay.connect(args.db)
        have = {r[0] for r in conn.execute("SELECT DISTINCT source FROM items")}
        dup = have & set(sources)
        if dup:
            raise SystemExit(f"items already exist for source(s) {sorted(dup)}; refusing to re-insert")
        keys = replay.existing_keys(conn)
        first = replay.max_item_id(conn) + 1
    rows = assign_ids(by_source, first, keys)
    summ = summarize(rows)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(f"total {len(rows)} items (ids {first}..{first + len(rows) - 1}) train={sum(r['split'] == 'train' for r in rows)}")
    if args.report:
        Path(args.report).write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    if args.preview:
        seen: dict = {}
        with open(args.preview, "w", encoding="utf-8") as f:
            for r in rows:
                kk = (r["source"], r["split"])
                if seen.get(kk, 0) < 3:
                    seen[kk] = seen.get(kk, 0) + 1
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not args.dry_run:
        replay.insert_items(conn, rows)
        print(f"inserted {len(rows)} items")
        update_manifest(args.root, summ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
