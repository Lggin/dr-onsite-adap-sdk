#!/usr/bin/env python3
"""A/B — 리플레이 200장을 '어떻게 뽑느냐'가 회귀 방어에 영향을 주는가.

배경: 2026-08-03 측정에서 k=200 은 회귀 −0.0054 로 기준(−0.005)을 넘겨 "보호 부족"으로
판정됐고 k=400 이 채택됐다. 그런데 그때 200 장은 **무작위 층화**로 뽑은 것이었다.
팀 요청으로 200 으로 줄이되, 선정 방식을 바꾸면(이상치 제외 + 분위 층화) 400 장급 보호가
나오는지 확인한다.

  A = k200 무작위 층화   (2026-08-03 과 같은 방식, 재현 겸 대조군)
  B = k200 큐레이션      (극단 이상치 제외 + 5분위 층화 + 클래스 최소 보장)

두 조건은 **동일한 새 라벨 20장 · 동일 학습중 검증셋 · 동일 회귀 채점셋**을 쓴다
(round_n20_k400/split_manifest.json 재사용). 리플레이 선정만 다르다.
비교 기준선은 sweep_ktrain.json 의 k200 / k400 결과.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_stage1 import (BASE, REG_VALID, SDK, T, TM_TO_CARTON, TM_VALID,
                          evaluate, run_step)

TARGETS, R = "all-linear", 8
OUT = f"{T}/bench/ab_replay_pick.json"
RUNS = f"{T}/bench/runs"
CONDS = [("A_k200_strat", f"{T}/bench/round_A_k200_strat"),
         ("B_k200_curated", f"{T}/bench/round_B_k200_curated")]


def main():
    stage1 = json.load(open(f"{T}/bench/sweep_stage1.json"))
    base = stage1["_base_v2"]
    prev = json.load(open(f"{T}/bench/sweep_ktrain.json"))
    results = {"_base_v2": base,
               "_ref_k200_2026-08-03": prev.get("k200"),
               "_ref_k400_2026-08-03": prev.get("k400")}
    if os.path.isfile(OUT):
        for k, v in json.load(open(OUT)).items():
            if not k.startswith("_") and "adapt" in v:
                results[k] = v
                print(f"[{k}] 기존 결과 재사용: 적응 {v['adapt']:.4f} / 회귀 {v['regress']:.4f}",
                      flush=True)

    man = json.load(open(f"{T}/bench/round_n20_k400/split_manifest.json"))
    reg_ids = man["held_out_regression_ids"]

    for tag, data_dir in CONDS:
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
        print(f"\n═══ {tag} ═══", flush=True)
        t0 = time.time()
        rc = run_step("dr_onsite_adap.pipeline.train_lora", env, log)
        # 종료시점 SIGABRT 는 결과물이 있으면 성공으로 본다(1단계에서 관측된 현상).
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
        ad = evaluate(f"{d}/merged.pth", *TM_VALID, remap=TM_TO_CARTON)
        rg = evaluate(f"{d}/merged.pth", *REG_VALID, keep_ids=reg_ids)
        results[tag] = {"pick": tag.split("_")[-1], "k_train": 200,
                        "adapt": ad, "regress": rg,
                        "adapt_gain": ad - base["adapt"],
                        "regress_delta": rg - base["regress"],
                        "minutes": round((time.time() - t0) / 60, 1)}
        print(f"  적응 {ad:.4f} ({ad-base['adapt']:+.4f})   "
              f"회귀 {rg:.4f} ({rg-base['regress']:+.4f})", flush=True)
        json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)

    L = ["# 리플레이 선정 방식 A/B (k=200 고정)", "",
         f"베이스 v2: 적응 {base['adapt']:.4f} / 회귀 {base['regress']:.4f}",
         "두 조건 모두 동일한 새 라벨 20장 · 동일 검증셋 · 동일 채점셋. 리플레이 선정만 다름.",
         "", "| 조건 | 선정 방식 | 적응 | Δ적응 | 회귀 | Δ회귀 | 판정 |",
         "|---|---|---|---|---|---|---|"]
    for tag, _ in CONDS:
        v = results.get(tag)
        if not v or "adapt" not in v:
            continue
        ok = "통과" if v["regress_delta"] >= -0.005 else "**회귀 초과**"
        nm = "무작위 층화" if "strat" in tag else "큐레이션(이상치제외+분위층화)"
        L.append(f"| {tag} | {nm} | {v['adapt']:.4f} | {v['adapt_gain']:+.4f} | "
                 f"{v['regress']:.4f} | {v['regress_delta']:+.4f} | {ok} |")
    for key, nm in (("_ref_k200_2026-08-03", "k200 (08-03, 무작위)"),
                    ("_ref_k400_2026-08-03", "k400 (08-03, 무작위)")):
        v = results.get(key)
        if v and "adapt" in v:
            ok = "통과" if v["regress_delta"] >= -0.005 else "**회귀 초과**"
            L.append(f"| (참고) | {nm} | {v['adapt']:.4f} | {v['adapt_gain']:+.4f} | "
                     f"{v['regress']:.4f} | {v['regress_delta']:+.4f} | {ok} |")
    L += ["", "판정 기준: 회귀 Δ ≥ −0.005 이면 통과. 리플레이는 회귀를 막는 장치이므로",
          "적응이 비슷하면 회귀가 작은 쪽이 안전하다.", ""]
    open(OUT.replace(".json", ".md"), "w").write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
