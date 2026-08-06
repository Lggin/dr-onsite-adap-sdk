#!/usr/bin/env python3
"""[라벨] 촬영해 둔 프레임을 한 장씩 몰아서 라벨합니다.

현장에서는 찍기만 하고(`dr-onsite-capture` 또는 GUI '📷 자동 촬영'), 라벨은 나중에
이걸로 몰아서 합니다. 폴더의 `cap_*.jpg` 를 순서대로 열어 라이브 앱과 **똑같은 SAM
라벨러**를 띄우고, 저장한 장은 `labeled/` 로 옮겨 진행 상황을 남깁니다(중간에 끊고
다시 실행하면 남은 것부터 이어서).

⚠️ 평가셋(held-out)용으로 찍었다면 **그 프레임에 보이는 객체를 전부** 라벨하세요.
   빠뜨린 객체는 학습에서 배경(negative)으로 쓰여 해당 클래스를 무너뜨립니다
   (2026-08-04 실측: 팔레트 20개를 빼자 팔레트 검출이 120/120 → 0/120).

라벨러 조작: 클릭=SAM 마스크, 숫자키=클래스 확정, Enter=저장하고 다음, ESC=이 장 건너뜀,
q=세션 종료. 클래스 번호는 실행 시 화면에 표시됩니다.

사용법:
  dr-onsite-label                        # DEPAL_FIELD_CAPTURE_DIR 의 미라벨 프레임
  dr-onsite-label --dir ~/field_captures --limit 20
  dr-onsite-label --list                 # 무엇이 남았는지만 보기
다음 단계:
  dr-onsite-ingest --holdout --data-dir <촬영폴더>     # 평가셋에 적재(학습과 분리)
"""
import argparse
import glob
import os
import shutil

import cv2

from .. import config
from .._logging import ensure_utf8_output, setup_cli_logging


def pending(cap_dir):
    """아직 라벨 안 한 촬영 프레임 목록(정렬)."""
    return sorted(glob.glob(os.path.join(cap_dir, "cap_*.jpg")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=config.FIELD_CAPTURE_DIR, help="촬영 폴더")
    ap.add_argument("--limit", type=int, default=0, help="이번에 이만큼만(0=전부)")
    ap.add_argument("--list", action="store_true", help="남은 장수만 출력하고 종료")
    args = ap.parse_args()
    ensure_utf8_output()
    setup_cli_logging()

    cap_dir = os.path.expanduser(args.dir)
    todo = pending(cap_dir)
    done_dir = os.path.join(cap_dir, "labeled")
    n_done = len(glob.glob(os.path.join(done_dir, "cap_*.jpg")))
    print(f"촬영 폴더: {cap_dir}")
    print(f"  남은 프레임 {len(todo)}장 / 라벨 완료 {n_done}장")
    if args.list:
        for p in todo[:50]:
            print("   ", os.path.basename(p))
        if len(todo) > 50:
            print(f"    … 외 {len(todo) - 50}장")
        return 0
    if not todo:
        print("라벨할 프레임이 없습니다.")
        return 0
    if args.limit:
        todo = todo[:args.limit]

    from ..runtime.labeler import SAMLabeler

    os.makedirs(done_dir, exist_ok=True)
    labeler = SAMLabeler(config.SAM_CHECKPOINT, cap_dir,
                         {c: n for c, (n, _col) in config.CLASSES.items()},
                         sam_type=config.SAM_TYPE, device=config.SAM_DEVICE)
    keys = "  ".join(f"{c}={n}" for c, (n, _col) in sorted(config.CLASSES.items()))
    print(f"클래스 키: {keys}")
    print("Enter=저장/다음, ESC=건너뜀, q=종료.  보이는 객체는 전부 라벨하세요.\n")

    saved = skipped = 0
    for i, path in enumerate(todo, 1):
        img = cv2.imread(path)
        if img is None:
            print(f"[{i}/{len(todo)}] 읽기 실패, 건너뜀: {os.path.basename(path)}")
            continue
        print(f"[{i}/{len(todo)}] {os.path.basename(path)}", flush=True)
        try:
            ok = labeler.label_frame(img)
        except KeyboardInterrupt:
            print("\n중단(Ctrl-C).")
            break
        if ok:
            shutil.move(path, os.path.join(done_dir, os.path.basename(path)))
            npz = os.path.splitext(path)[0] + ".npz"
            if os.path.isfile(npz):
                shutil.move(npz, os.path.join(done_dir, os.path.basename(npz)))
            saved += 1
        else:
            skipped += 1
            print("   건너뜀(저장 안 함) — 폴더에 남겨둡니다.")

    left = len(pending(cap_dir))
    print(f"\n✅ 이번 세션: 저장 {saved}장 / 건너뜀 {skipped}장 | 남은 프레임 {left}장")
    if saved:
        print(f"평가셋에 넣기:  dr-onsite-ingest --holdout --data-dir {cap_dir}")
        print("판정 실행:      dr-onsite-gate --base <이전.pth> --cand <새것.pth>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
