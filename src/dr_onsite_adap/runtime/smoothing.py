"""실시간 검출 시간축 안정화 (hysteresis / keep-alive).

문제: 신뢰도가 검출 임계(예 0.5) 근처에서 떠는 박스는 매 프레임 검출/탈락을 반복해
화면에서 통째로 깜빡인다(실시간 모드에서만 보임).

해결: 검출을 2D 중심으로 프레임 간 매칭해 트랙을 유지한다. 한 트랙이 어느 프레임에
안 잡혀도 `keep_frames` 동안은 직전 검출을 그대로 표시(=사라짐 흡수). 새 검출은
`min_hits` 프레임 잡히면 표시. **검출 임계는 그대로** 두고 표시만 안정화하므로
오검출을 새로 만들지 않는다(이미 검출된 것만 잠깐 유지).

run_detection() 의 state(dict)를 받아 표시용 state 를 돌려준다. 픽 대상(active)과
그 마스크·팔레트를 트랙 기준으로 재구성한다. 단발('d') 모드에선 쓰지 않는다.
"""
import os
from collections import defaultdict

from .. import config

KEEP_FRAMES = int(os.environ.get("DEPAL_SMOOTH_KEEP", "10"))   # 사라져도 유지할 프레임 수
MIN_HITS = int(os.environ.get("DEPAL_SMOOTH_MIN_HITS", "1"))   # 표시 시작에 필요한 연속 검출 수
MATCH_DIST = float(os.environ.get("DEPAL_SMOOTH_MATCH_PX", "45"))  # 같은 물체로 볼 중심 거리(px)


class DetectionSmoother:
    def __init__(self, keep_frames=KEEP_FRAMES, min_hits=MIN_HITS, match_dist=MATCH_DIST):
        self.keep = keep_frames
        self.min_hits = min_hits
        self.match_d2 = match_dist * match_dist
        self.tracks = []      # 픽 대상 트랙: {det, center, ttl, hits}
        self.pallet_tracks = []  # 팔레트 트랙: {pallet, center, ttl, hits}

    def reset(self):
        self.tracks = []
        self.pallet_tracks = []

    @staticmethod
    def _match(tracks, items, center_of, match_d2):
        """tracks 를 items 에 1:1 근접 매칭. 매칭된 item 인덱스 set 반환, 트랙 갱신/감쇠."""
        used = set()
        for tr in tracks:
            cx, cy = tr["center"]
            best, best_d2 = -1, match_d2
            for i, it in enumerate(items):
                if i in used:
                    continue
                c = center_of(it)
                if c is None:
                    continue
                d2 = (c[0] - cx) ** 2 + (c[1] - cy) ** 2
                if d2 < best_d2:
                    best_d2, best = d2, i
            if best >= 0:
                used.add(best)
                tr["item"] = items[best]
                tr["center"] = center_of(items[best])
                tr["ttl"] = None  # refreshed below by caller
                tr["_matched"] = True
            else:
                tr["_matched"] = False
        return used

    def _step(self, tracks, items, center_of):
        used = self._match(tracks, items, center_of, self.match_d2)
        for tr in tracks:
            if tr.get("_matched"):
                tr["ttl"] = self.keep
                tr["hits"] += 1
            else:
                tr["ttl"] -= 1
        for i, it in enumerate(items):
            if i not in used and center_of(it) is not None:
                tracks.append({"item": it, "center": center_of(it),
                               "ttl": self.keep, "hits": 1})
        tracks[:] = [t for t in tracks if t["ttl"] > 0]
        # 표시 대상: 충분히 본 트랙. 항목은 마지막으로 본 검출(유지 중엔 직전 것).
        return [t["item"] for t in tracks if t["hits"] >= self.min_hits]

    def update(self, state):
        active = self._step(self.tracks, state["active"],
                            lambda d: d["res"]["center_2d"])
        pallets = self._step(self.pallet_tracks, state.get("pallets", []),
                            lambda p: p.get("center_2d"))

        # 마스크 오버레이를 트랙 기준으로 재구성(픽 대상 + 팔레트).
        mwc = [(d["mask"], d["color_bgr"]) for d in active]
        pallet_bgr = config.CLASSES.get(config.PALLET_CID, (None, (130, 100, 50)))[1]
        for p in pallets:
            mwc.append((p["mask"], pallet_bgr))

        counts = defaultdict(int)
        for d in active:
            counts[d["cid"]] += 1
        for _ in pallets:
            counts[config.PALLET_CID] += 1

        new_state = dict(state)
        new_state["active"] = active
        new_state["pallets"] = pallets
        new_state["masks_with_color"] = mwc
        new_state["fail_markers"] = []   # 안정화 모드에선 전이성 실패마커는 숨김
        new_state["item_count"] = len(active)
        new_state["counts"] = dict(counts)
        new_state["done"] = bool(pallets) and len(active) == 0
        return new_state
