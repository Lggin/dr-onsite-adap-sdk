#!/usr/bin/env python3
"""A/B — 크래프트 목표 리플레이 200장이 무작위 200 / 기존 400 보다 나은가.

용량 제약으로 리플레이가 400 → 200 으로 정해졌다. 무작위 200 은 회귀가 기준선
(−0.005)을 넘나든다고 이미 측정됐으므로(`ab_replay_pick.md`), 장수를 줄이는 대신
**지키려는 대상을 좁힌** 셋을 만들었다 — 학습셋 전반 대신 **크래프트 갈색 Carton**.
그게 실제로 통하는지 잰다.

  (참고) k400 무작위 층화        2026-08-03 측정값 재사용
  (참고) A k200 무작위 층화      2026-08-18 측정값 재사용 (대조군)
  (참고) B k200 큐레이션         2026-08-18 측정값 재사용 (다른 목적의 큐레이션, 실패)
  C  k200 크래프트               ← 신규
  D  k200 크래프트 (보장 완화)    ← 신규

모든 조건이 **동일한 새 라벨 20장 · 동일 학습중 검증셋 · 동일 회귀 채점셋 1,391장**을
쓴다(`round_n20_k400/split_manifest.json` 재사용). 리플레이 선정만 다르다.

## 이번에 추가한 측정 — 회귀를 세 조각으로 가른다

전체 회귀 mAP 하나로는 "크래프트를 지켰는가"를 볼 수 없다. 목표가 좁아졌으므로 채점도
좁혀야 한다. 회귀 채점셋을 크래프트 근접도로 갈라 따로 잰다.

  regress_kraft : Carton 이 있는 이미지 중 **크래프트 근접 절반** — 여기가 떨어지면
                  목표 실패
  regress_cfar  : Carton 이 있는 이미지 중 **먼 절반** — 하락은 예상되나 크기는 알아야 한다
  regress_other : **Carton 이 없는 이미지** — 다른 클래스(슈링크랩·토트·팔레트) 보존

⚠️ 회귀 채점셋 1,391장 중 Carton 이 있는 것은 352장뿐이고 1,039장은 Carton 이 없다.
그래서 단순히 근/원 둘로 가르면 '먼 쪽'이 사실상 다른 클래스 점수가 되어 해석이 엉킨다.

## ⚠️ 교란 요인

크래프트 선정은 "거리가 비슷하면 Carton 이 많은 이미지 먼저"를 쓰므로 같은 200장에
**라벨이 더 많다**(C 4,358 / D 4,879 vs A 2,327 / k400 3,285). C·D 가 이기더라도
"크래프트라서"인지 "인스턴스가 많아서"인지 이 실험만으로는 못 가른다. 가르려면
'무작위 + 인스턴스 많은 순' arm 이 하나 더 필요하다.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/dr/lee3/thirdtask/bench")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_stage1 import (BASE, REG_VALID, SDK, T, TM_TO_CARTON, TM_VALID,   # noqa: E402
                          evaluate, run_step)
from kraft_pick import load_reference, score_images                          # noqa: E402

TARGETS, R = "all-linear", 8
OUT = f"{T}/bench/ab_kraft_replay.json"
RUNS = f"{T}/bench/runs"
BASE_DOM = f"{T}/final_dataset_ownsource_v2"
CONDS = [("C_k200_kraft", f"{T}/bench/round_C_k200_kraft", "크래프트(보장 유지)"),
         ("D_k200_kraftlight", f"{T}/bench/round_D_k200_kraftlight",
          "크래프트(보장 완화)"),
         # E = 교란 분리용. A(무작위 층화)와 선정 규칙·Carton 비중이 같고 **중복 등록만
         # 제거**했다. 소스는 현장 40장을 50번 등록하므로 image_id 로 뽑는 stratified_pick
         # 은 같은 사진이 여러 슬롯을 먹는다 — A 는 220장 중 sam_hard 가 54장(고유 40),
         # kraft 선정은 파일명 기준이라 0~7장이다. 즉 C/D 는 실사진 비중이 애초에 낮았다.
         # E 가 C 처럼 나쁘면 원인은 중복 제거(실사진 손실), A 처럼 나오면 원인은 크래프트
         # 선정 쪽이다.
         ("E_k200_strat_dedup", f"{T}/bench/round_E_k200_strat_dedup",
          "무작위 층화 + 중복 제거")]
S = os.environ["S"]


def split_reg_by_kraft(reg_ids):
    """회귀 채점셋을 세 조각으로: Carton-크래프트 / Carton-먼쪽 / Carton 없음."""
    coco = json.load(open(REG_VALID[0]))
    cname = {c["id"]: c["name"] for c in coco["categories"]}
    _, MU, SD, ZA, _ = load_reference(S)
    _, dist, _, _ = score_images(coco, REG_VALID[1], MU, SD, ZA,
                                 cache=f"{S}/kraftscore_basedom_valid.pkl",
                                 cat_name_of=lambda c: cname[c])
    id2n = {im["id"]: im["file_name"] for im in coco["images"]}
    scored = [(i, dist.get(id2n[i], np.inf)) for i in reg_ids]
    fin = [(i, d) for i, d in scored if np.isfinite(d)]
    other = [i for i, d in scored if not np.isfinite(d)]
    if not fin:
        return [], [], other, float("nan")
    med = float(np.median([d for _, d in fin]))
    near = [i for i, d in fin if d <= med]
    cfar = [i for i, d in fin if d > med]
    print(f"회귀 채점셋 {len(reg_ids)}장 → Carton 크래프트 {len(near)} / "
          f"Carton 먼쪽 {len(cfar)} / Carton 없음 {len(other)}  (경계 거리 {med:.2f})")
    return near, cfar, other, med


def main():
    stage1 = json.load(open(f"{T}/bench/sweep_stage1.json"))
    base = stage1["_base_v2"]
    prev_k = json.load(open(f"{T}/bench/sweep_ktrain.json"))
    prev_ab = json.load(open(f"{T}/bench/ab_replay_pick.json"))
    results = {"_base_v2": base,
               "_ref_k400_strat": prev_k.get("k400"),
               "_ref_k200_strat": prev_ab.get("A_k200_strat"),
               "_ref_k200_curated": prev_ab.get("B_k200_curated")}
    if os.path.isfile(OUT):
        for k, v in json.load(open(OUT)).items():
            if not k.startswith("_") and "adapt" in v:
                results[k] = v
                print(f"[{k}] 기존 결과 재사용", flush=True)

    man = json.load(open(f"{T}/bench/round_n20_k400/split_manifest.json"))
    reg_ids = man["held_out_regression_ids"]
    near, cfar, other, med = split_reg_by_kraft(reg_ids)
    SLICES = (("kraft", near), ("cfar", cfar), ("other", other))
    results["_reg_split"] = {"kraft_n": len(near), "cfar_n": len(cfar),
                             "other_n": len(other), "boundary": med}

    # 베이스의 조각별 점수 — Δ 를 계산할 기준선
    if "_base_split" not in results:
        bsp = {f"regress_{k}": evaluate(BASE, *REG_VALID, keep_ids=v)
               for k, v in SLICES if v}
        results["_base_split"] = bsp
        print("베이스 회귀 조각: " + '  '.join(f"{k} {v:.4f}" for k, v in bsp.items()),
              flush=True)
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    bsp = results["_base_split"]

    for tag, data_dir, nm in CONDS:
        if tag in results:
            continue
        d = f"{RUNS}/ab_{tag}"
        os.makedirs(d, exist_ok=True)
        log = f"{d}/trial.log"
        env = {**os.environ, "PYTHONPATH": SDK,
               "DEPAL_BASE_WEIGHTS": BASE,
               "DEPAL_TRAIN_DATA": data_dir,
               "DEPAL_LORA_SAFE": f"{d}/adapter", "DEPAL_LORA_RUN": f"{d}/run",
               "DEPAL_FINAL": f"{d}/merged.pth", "DEPAL_NUM_CLASSES": "5",
               "DEPAL_EPOCHS": "25",
               "DEPAL_LORA_TARGETS": TARGETS, "DEPAL_LORA_R": str(R),
               "DEPAL_LORA_ALPHA": str(2 * R)}
        print(f"\n═══ {tag} — {nm} ═══", flush=True)
        t0 = time.time()
        rc = run_step("dr_onsite_adap.pipeline.train_lora", env, log)
        # 종료시점 SIGABRT 는 결과물이 있으면 성공으로 본다(1단계에서 관측된 현상)
        if rc != 0 and os.path.isfile(f"{d}/adapter/adapter_config.json"):
            print(f"  ⚠️ rc={rc} 이지만 어댑터 저장됨 → 계속", flush=True)
            rc = 0
        if rc != 0:
            results[tag] = {"error": f"train rc={rc}"}
            json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
            print(f"  ✗ 학습 실패(rc={rc})", flush=True)
            continue
        rc = run_step("dr_onsite_adap.pipeline.merge", env, log)
        if rc != 0:
            results[tag] = {"error": f"merge rc={rc}"}
            json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
            print(f"  ✗ 병합 실패(rc={rc})", flush=True)
            continue
        ck = f"{d}/merged.pth"
        ad = evaluate(ck, *TM_VALID, remap=TM_TO_CARTON)
        rg = evaluate(ck, *REG_VALID, keep_ids=reg_ids)
        nlab = len(json.load(open(f"{data_dir}/train/_annotations.coco.json"))["annotations"])
        rec = {"pick": nm, "k_train": 200, "train_labels": nlab,
               "adapt": ad, "regress": rg,
               "adapt_gain": ad - base["adapt"],
               "regress_delta": rg - base["regress"],
               "minutes": round((time.time() - t0) / 60, 1)}
        for k, v in SLICES:
            if not v:
                continue
            sc = evaluate(ck, *REG_VALID, keep_ids=v)
            rec[f"regress_{k}"] = sc
            rec[f"regress_{k}_delta"] = sc - bsp[f"regress_{k}"]
        results[tag] = rec
        print(f"  적응 {ad:.4f} ({rec['adapt_gain']:+.4f})  회귀 {rg:.4f} "
              f"({rec['regress_delta']:+.4f})", flush=True)
        print("  조각별 Δ회귀: " + '  '.join(
            f"{k} {rec[f'regress_{k}_delta']:+.4f}" for k, v in SLICES
            if f"regress_{k}_delta" in rec), flush=True)
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

    # ── 보고서 ────────────────────────────────────────────────────────────
    L = ["# 크래프트 목표 리플레이 A/B (k=200)", "",
         f"베이스 v2: 적응 {base['adapt']:.4f} / 회귀 {base['regress']:.4f}",
         "베이스 회귀 조각: " + '  '.join(f"{k} {v:.4f}" for k, v in bsp.items()),
         "",
         "모든 조건이 동일한 새 라벨 20장 · 동일 학습중 검증셋 · 동일 회귀 채점셋"
         f" {len(reg_ids)}장. 리플레이 선정만 다름.",
         f"회귀 채점셋을 세 조각으로 갈라 따로 채점 — Carton 크래프트 {len(near)}장 / "
         f"Carton 먼쪽 {len(cfar)}장 / Carton 없음 {len(other)}장 (경계 {med:.2f}).", "",
         "| 조건 | 선정 | 장수 | 학습 라벨 | 적응 | Δ적응 | 회귀 | Δ회귀 | "
         "Δ크래프트 | Δ먼쪽 | Δ타클래스 | 판정 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    rows = [("_ref_k400_strat", "무작위 층화 (중복 포함)", 400),
            ("_ref_k200_strat", "무작위 층화 (중복 포함)", 200),
            ("_ref_k200_curated", "큐레이션(옛)", 200),
            ("E_k200_strat_dedup", "무작위 층화 + 중복 제거", 200),
            ("C_k200_kraft", "크래프트(보장 유지)", 200),
            ("D_k200_kraftlight", "크래프트(보장 완화)", 200)]
    for key, nm, k in rows:
        v = results.get(key)
        if not v or "adapt" not in v:
            continue
        ok = "통과" if v["regress_delta"] >= -0.005 else "**회귀 초과**"
        lab = v.get("train_labels", "—")
        cells = [(f"{v[f'regress_{s}_delta']:+.4f}" if f"regress_{s}_delta" in v else "—")
                 for s in ("kraft", "cfar", "other")]
        L.append(f"| {key.lstrip('_')} | {nm} | {k} | {lab} | {v['adapt']:.4f} | "
                 f"{v['adapt_gain']:+.4f} | {v['regress']:.4f} | "
                 f"{v['regress_delta']:+.4f} | " + " | ".join(cells) + f" | {ok} |")
    L += ["",
          "판정 기준: 회귀 Δ ≥ −0.005. 리플레이는 회귀를 막는 장치이므로 적응이 비슷하면",
          "회귀가 작은 쪽이 안전하다.", "",
          "⚠️ **교란 요인**: 크래프트 선정은 Carton 이 많은 이미지를 먼저 뽑으므로 같은",
          "200장에 학습 라벨이 더 많다. C·D 가 이기더라도 '크래프트라서'인지 '인스턴스가",
          "많아서'인지 이 실험만으로는 못 가른다 — 가르려면 '무작위 + 인스턴스 많은 순'",
          "arm 이 하나 더 필요하다.",
          "⚠️ **중복 등록 교란**: 소스는 현장 실촬 40장을 COCO images 에 각각 50번",
          "등록한다. `stratified_pick` 은 image_id 로 뽑으므로 같은 사진이 여러 슬롯을",
          "먹는다 — k400 은 420장 중 sam_hard 77장(고유 40), A 는 220장 중 54장이었다.",
          "kraft 선정은 파일명 기준이라 0~7장이다. **즉 참고 arm 들은 실사진 비중이",
          "훨씬 높은 조건에서 측정된 값이다.** E 가 그 차이를 분리한다.",
          "⚠️ 합성 렌더 도메인의 **대리 지표**다. 400장 채택 근거도 같은 벤치이므로",
          "비교 자체는 공정하다.", ""]
    open(OUT.replace(".json", ".md"), "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
