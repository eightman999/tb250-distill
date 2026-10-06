import collections
import json
import re

import pytest

from tb250distill.data import synth

N_TRAIN, N_VAL, N_TEST, N_ROB = 600, 100, 100, 40


@pytest.fixture(scope="module")
def rows():
    return synth.build_dataset(N_TRAIN, N_VAL, N_TEST, N_ROB, master_seed=123)


def by(rows, split):
    return [r for r in rows if r["split"] == split]


def test_deterministic(rows):
    again = synth.build_dataset(N_TRAIN, N_VAL, N_TEST, N_ROB, master_seed=123)
    assert again == rows
    other = synth.build_dataset(50, 10, 10, 5, master_seed=124)
    assert [r["context"] for r in other[:20]] != [r["context"] for r in rows[:20]]


def test_split_sizes_and_ids(rows):
    assert len(by(rows, "train")) == N_TRAIN
    assert len(by(rows, "val")) == N_VAL
    assert len(by(rows, "test")) == N_TEST
    assert len(by(rows, "robust")) == N_ROB * 6
    ids = [r["item_id"] for r in rows]
    assert ids == list(range(1, len(rows) + 1))


def test_category_ratio(rows):
    tr = by(rows, "train")
    cnt = collections.Counter(r["category"] for r in tr)
    for cat, pct in synth.CATEGORY_RATIOS.items():
        assert abs(cnt[cat] - N_TRAIN * pct / 100) <= 1, (cat, cnt)


def test_lang_mix(rows):
    tr = by(rows, "train")
    ja = sum(r["lang"] == "ja" for r in tr) / len(tr)
    assert 0.5 < ja < 0.7
    for r in tr[:100]:
        has_ja = re.search(r"[぀-ヿ一-鿿]", r["question"] + "".join(r["candidates"]))
        assert bool(has_ja) == (r["lang"] == "ja")


def test_candidates_semantic_not_labels(rows):
    for r in rows:
        k = len(r["candidates"])
        assert 2 <= k <= 5
        assert len(set(r["candidates"])) == k
        for c in r["candidates"]:
            assert not re.fullmatch(r"[A-Ea-e1-5][.)]?", c.strip()), c
            assert len(c) >= 2
        if r["gold"] is not None:
            assert 0 <= r["gold"] < k


def test_k_distribution_covers_2_to_5(rows):
    ks = collections.Counter(len(r["candidates"]) for r in by(rows, "train"))
    assert set(ks) == {2, 3, 4, 5}


def test_gold_policy(rows):
    tr = by(rows, "train")
    for r in tr:
        if r["category"] == "ambiguous":
            assert r["gold"] is None
    assert any(r["gold"] is not None for r in tr if r["category"] == "nli")
    assert all(r["gold"] is not None for r in tr if r["category"] in ("nli", "intent", "ranking", "sentiment"))
    assert any(r["gold"] is None for r in tr if r["category"] == "state_action")


def test_nli_gold_consistent():
    """NLI は構成的: 同じ前提から entail/contra/neutral が別ラベルになる。"""
    from tb250distill.data.synth import make_scenario, render
    seen = collections.Counter()
    for seed in range(300):
        scn = make_scenario("nli", "ja" if seed % 2 else "en", seed)
        r = render(scn)
        role = scn.cands[scn.gold].role
        seen[role] += 1
        assert r["gold"] == scn.gold
    assert set(seen) == {"nli_entail", "nli_contra", "nli_neutral"}


def test_context_length_range(rows):
    lens = {"ja": [], "en": []}
    for r in by(rows, "train"):
        lens[r["lang"]].append(synth._text_len(r["lang"], r["context"]))
    for L, v in lens.items():
        assert min(v) >= 10
        assert max(v) <= synth.CTX_LEN_RANGE[L][1] * 2
        assert max(v) - min(v) > 40  # ばらつく


def test_no_duplicates_across_splits(rows):
    keys = collections.Counter(synth._key(r) for r in rows if r["split"] != "robust")
    assert max(keys.values()) == 1


def test_robust_variants(rows):
    rob = by(rows, "robust")
    base = {r["item_id"]: r for r in by(rows, "test")}
    by_var = collections.Counter(r["variant"] for r in rob)
    assert set(by_var) == set(synth.VARIANT_KINDS)
    assert set(by_var.values()) == {N_ROB}
    for r in rob:
        b = base[r["variant_of"]]
        assert r["category"] == b["category"] and r["lang"] == b["lang"]
        v = r["variant"]
        if v == "perm":
            assert sorted(r["candidates"]) == sorted(b["candidates"])
            assert r["candidates"] != b["candidates"]
            if b["gold"] is not None:
                assert r["candidates"][r["gold"]] == b["candidates"][b["gold"]]
            assert r["context"] == b["context"] and r["question"] == b["question"]
        elif v in ("cand_paraphrase", "unseen_cand"):
            assert r["context"] == b["context"] and r["question"] == b["question"]
            assert r["gold"] == b["gold"]
            assert r["candidates"] != b["candidates"]
            assert len(r["candidates"]) == len(b["candidates"])
        elif v == "ctx_paraphrase":
            assert r["context"] != b["context"] and r["candidates"] == b["candidates"]
        elif v == "irrelevant_ctx":
            assert len(r["context"]) > len(b["context"]) and r["candidates"] == b["candidates"]
        elif v == "ambiguous":
            assert r["question"] != b["question"] and r["context"] == b["context"]


def all_role_strings(tier):
    out = set()
    for role in synth.ROLES.values():
        for L in ("ja", "en"):
            out.update(role[L][tier])
    return out


def test_unseen_pool_disjoint_from_train_and_para():
    tr, pa, un = all_role_strings("train"), all_role_strings("para"), all_role_strings("unseen")
    assert not (tr & un) and not (pa & un) and not (tr & pa)
    for t in synth.TOOLS.values():
        for L in ("ja", "en"):
            assert t[L][0] != t[L][1]
    for role in synth.ROLES.values():
        for L in ("ja", "en"):
            assert all(len(role[L][t]) >= 2 for t in ("train", "para", "unseen"))


def test_unseen_cands_never_in_train_items(rows):
    train_cands = set()
    for r in rows:
        if r["split"] != "robust" or r["variant"] != "unseen_cand":
            train_cands.update(r["candidates"])
    unseen_cands = set()
    for r in by(rows, "robust"):
        if r["variant"] == "unseen_cand":
            unseen_cands.update(r["candidates"])
    # ranking の候補は数値込みで別文字列になるため、役割文言だけ厳密に比較する
    role_unseen = all_role_strings("unseen")
    tool_unseen = {n for t in synth.TOOLS.values() for L in ("ja", "en") for n in [t[L][1]]}
    seen_unseen_roles = [c for c in unseen_cands if c in role_unseen or any(n in c for n in tool_unseen)]
    assert seen_unseen_roles
    for c in seen_unseen_roles:
        assert c not in train_cands, c
    for n in tool_unseen:
        assert not any(n in c for c in train_cands), n
    # ranking の名詞も分離
    for dom in synth.RANK_DOMAINS.values():
        for L in ("ja", "en"):
            assert not set(dom[L]["train"]) & set(dom[L]["unseen"])


def test_replay_roundtrip_with_synth(tmp_path, rows):
    from tb250distill import replay
    conn = replay.connect(str(tmp_path / "r.sqlite"))
    assert replay.insert_items(conn, rows[:50]) == 50
    got = replay.get_items(conn, [1, 2, 3])
    assert got[0]["candidates"] == rows[0]["candidates"]
