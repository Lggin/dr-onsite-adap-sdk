"""md → 컨플루언스에 붙일 수 있는 단일 HTML. 그림은 base64 로 파일 안에 넣는다.

사용: python3 md2html.py <입력.md> [그림.png ...]
그림은 `(그림 패널 A)` 같은 문구 대신, 본문에서 처음 나오는 h1 뒤에 한 번 삽입한다.
"""
import base64, os, re, sys
import markdown

# 그림을 어느 소제목 **앞**에 넣을지. 없으면 문서 끝.
FIG_AT = {'carton_eda_4패널.png': 'A-1', 'carton_출처비교.png': 'A-3',
          'eda_요약_4패널.png': 'A-1'}

src = sys.argv[1]
figs = sys.argv[2:]
out = os.path.splitext(src)[0] + '.html'
md = open(src, encoding='utf-8').read()
title = md.split('\n', 1)[0].lstrip('# ').strip()

html = markdown.markdown(md, extensions=['tables', 'fenced_code', 'toc', 'attr_list'])

for f in figs:
    b64 = base64.b64encode(open(f, 'rb').read()).decode()
    tag = (f'<figure><img src="data:image/png;base64,{b64}" alt="{os.path.basename(f)}">'
           f'<figcaption>{os.path.basename(f)}</figcaption></figure>')
    # 그림 파일명에 대응하는 소제목 앞에 넣는다(toc 확장이 h2 에 id 를 붙이므로 정규식).
    where = FIG_AT.get(os.path.basename(f))
    m = re.search(r'<h2[^>]*>\s*' + re.escape(where) + r'\.', html) if where else None
    if m:
        html = html[:m.start()] + tag + html[m.start():]
    else:
        print(f"⚠️ {os.path.basename(f)} 삽입 위치를 못 찾아 문서 끝에 붙였다")
        html += tag

CSS = """
:root{--fg:#172b4d;--mut:#5e6c84;--line:#dfe1e6;--bg:#fff;--soft:#f4f5f7;
--accent:#0052cc;--warn:#bf5b00;--warnbg:#fffae6;--code:#f4f5f7}
*{box-sizing:border-box}
body{font-family:-apple-system,'Malgun Gothic','Noto Sans KR',sans-serif;
color:var(--fg);background:var(--bg);max-width:1080px;margin:0 auto;padding:28px 26px 90px;
line-height:1.75;font-size:15px}
h1{font-size:27px;margin:38px 0 14px;padding-bottom:9px;border-bottom:3px solid var(--accent)}
h1:first-child{margin-top:0}
h2{font-size:20px;margin:32px 0 10px;color:#253858}
h3{font-size:16.5px;margin:24px 0 8px;color:#344563}
p{margin:9px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;
display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;white-space:nowrap}
th{background:var(--soft);font-weight:600}
td:nth-child(n+2){white-space:normal}
tr:nth-child(even) td{background:#fafbfc}
code{background:var(--code);padding:1.5px 5px;border-radius:3px;
font-family:ui-monospace,'D2Coding',Consolas,monospace;font-size:12.5px}
pre{background:#0d1117;color:#c9d1d9;padding:15px 18px;border-radius:6px;
overflow-x:auto;line-height:1.55;font-size:12.5px}
pre code{background:none;color:inherit;padding:0}
blockquote{margin:14px 0;padding:11px 16px;background:var(--warnbg);
border-left:4px solid #ffab00;color:#5a4300}
blockquote p{margin:5px 0}
hr{border:0;border-top:1px solid var(--line);margin:34px 0}
figure{margin:20px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:5px}
figcaption{font-size:12px;color:var(--mut);margin-top:6px}
ol,ul{padding-left:26px}li{margin:5px 0}
strong{color:#091e42}
"""
open(out, 'w', encoding='utf-8').write(
    f'<!doctype html><html lang=ko><head><meta charset=utf-8>'
    f'<meta name=viewport content="width=device-width,initial-scale=1">'
    f'<title>{title}</title><style>{CSS}</style></head><body>{html}</body></html>')
print(f"생성: {out} ({os.path.getsize(out)/1024:.0f}KB, 그림 {len(figs)}개)")
