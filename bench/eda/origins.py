"""파일명 → 출처 판정 (단일 기준).

extract_all.py 의 origin() 은 `render_*` 를 놓쳐 '기타' 로 떨어뜨렸다.
확인 결과 `render_*.jpg` 8,266장은 **슈링크랩 단품 합성 렌더**다
(회색 무지 배경, 팔레트 없음, 프레임당 객체 1개). 이 판정을 단일 소스로 둔다.

전수 추출을 다시 돌릴 필요는 없다 — meta 에 파일명이 있으므로 사후 재판정한다.
"""
import numpy as np

SYNTH = ('합성(궤도렌더)', '합성(현장갭)', '합성(팔레트)', '합성(슈링크랩)')
REAL = ('현장(Thor)', '실촬(기타)', '실촬(DAS)')
EXTERNAL = ('외부(roboflow)',)
AUGMENT = ('슬립시트증강',)


def origin(fn):
    f = fn.lower()
    if f.startswith('sam_hard'):
        return '현장(Thor)'
    if f.startswith('real_'):
        return '실촬(기타)'
    if f.startswith('das_'):
        return '실촬(DAS)'
    if f.startswith('tm_'):
        return '합성(궤도렌더)'
    if f.startswith('fg_'):
        return '합성(현장갭)'
    if f.startswith('slipcp'):
        return '슬립시트증강'
    if f.startswith('render_'):
        return '합성(슈링크랩)'        # ← 고침: 예전엔 '기타'
    if 'synth' in f:
        return '합성(팔레트)'
    if '.rf.' in f or 'roboflow' in f:
        return '외부(roboflow)'
    return '기타'


def relabel(meta):
    """meta 리스트 → 재판정된 출처 배열."""
    return np.array([origin(m[2]) for m in meta])


def domain(org):
    """세부 출처 → 큰 구분(합성/실촬/외부/증강)."""
    return np.where(np.isin(org, SYNTH), '합성',
                    np.where(np.isin(org, REAL), '실촬',
                             np.where(np.isin(org, EXTERNAL), '외부', '증강')))


COLORS = {'합성(궤도렌더)': '#2e86de', '합성(현장갭)': '#54a0ff',
          '합성(팔레트)': '#a4c8f0', '합성(슈링크랩)': '#0abde3',
          '외부(roboflow)': '#8395a7', '현장(Thor)': '#ee5253',
          '실촬(기타)': '#ff9f43', '실촬(DAS)': '#d35400',
          '슬립시트증강': '#8e44ad', '기타': '#c8d6e5'}
