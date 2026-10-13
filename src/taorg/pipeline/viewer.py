"""Self-contained HTML inspection page for a pipeline work directory (offline; no external assets).

Shows, per relationship step: the frame (optional, downscaled JPEG embedded locally), tracker boxes
with IDs, maintained relations as labelled arrows (solid = observed this step, dashed = retained by
temporal memory), takeover-flagged tracks, and a key x time timeline. Frames are omitted with
``with_frames=False`` (use this when the video must not be redistributed).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TAORG pipeline viewer</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--obs:#0072B2;--ret:#E69F00;--flag:#CC79A7;--grid:#e5e5ea}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--grid:#2c2c30}}
body{margin:0;padding:16px;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:18px;margin:0 0 4px} .mut{color:var(--mut)} .row{display:flex;gap:16px;flex-wrap:wrap}
canvas{max-width:100%;border:1px solid var(--grid);border-radius:6px} #ctl{margin:8px 0;display:flex;gap:8px;align-items:center}
input[type=range]{flex:1} .key{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}
</style></head><body>
<h1>TAORG pipeline viewer</h1>
<div class="mut" id="meta"></div>
<div id="ctl"><button id="play">Play</button><input type="range" id="t" min="0" value="0"><span id="lab"></span></div>
<div class="row"><canvas id="c" width="640" height="360"></canvas><canvas id="tl" width="640" height="360"></canvas></div>
<p class="mut"><span class="key" style="background:var(--obs)"></span>observed this step
<span class="key" style="background:var(--ret)"></span>retained by temporal memory
<span class="key" style="background:var(--flag)"></span>takeover flag on a track</p>
<script>
const D=__DATA__, F=__FRAMES__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const c=document.getElementById('c'),x=c.getContext('2d'),tl=document.getElementById('tl'),y=tl.getContext('2d');
const T=document.getElementById('t'); T.max=D.frames.length-1;
document.getElementById('meta').textContent=`${D.method} · tau ${D.tau} · every ${D.stride}th frame · ${D.frames.length} steps · ${D.fps.toFixed(1)} fps source`;
const keys=[...new Set(D.frames.flatMap(f=>f.edges.map(e=>`${e.s}→${e.o} ${e.p}`)))].sort();
const sx=c.width/D.width, sy=c.height/D.height, img=new Image(); let playing=false;
function draw(i){const f=D.frames[i]; document.getElementById('lab').textContent=`frame ${f.frame} (${f.t.toFixed(2)} s)`;
 const go=()=>{x.clearRect(0,0,c.width,c.height); if(F[f.frame]) x.drawImage(img,0,0,c.width,c.height);
  x.lineWidth=2; x.font='12px system-ui';
  for(const [t,b] of Object.entries(f.tracks)){x.strokeStyle=f.flagged.includes(+t)?css('--flag'):'#999';
   x.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy); x.fillStyle=x.strokeStyle; x.fillText(t,b[0]*sx+2,b[1]*sy+12);}
  for(const e of f.edges){const a=f.tracks[e.s],b=f.tracks[e.o]; if(!a||!b) continue;
   const ax=(a[0]+a[2])/2*sx, ay=(a[1]+a[3])/2*sy, bx=(b[0]+b[2])/2*sx, by=(b[1]+b[3])/2*sy;
   x.strokeStyle=e.observed?css('--obs'):css('--ret'); x.setLineDash(e.observed?[]:[5,4]);
   x.beginPath(); x.moveTo(ax,ay); x.lineTo(bx,by); x.stroke(); x.setLineDash([]);
   x.fillStyle=x.strokeStyle; x.fillText(e.p,(ax+bx)/2,(ay+by)/2-3);}};
 if(F[f.frame]){img.onload=go; img.src='data:image/jpeg;base64,'+F[f.frame];} else go();
 y.clearRect(0,0,tl.width,tl.height); const h=Math.max(4,Math.min(18,(tl.height-20)/Math.max(1,keys.length))), w=tl.width/D.frames.length;
 D.frames.forEach((g,j)=>g.edges.forEach(e=>{const k=keys.indexOf(`${e.s}→${e.o} ${e.p}`);
  y.fillStyle=e.observed?css('--obs'):css('--ret'); y.fillRect(j*w,k*h,Math.max(1,w),h-1);}));
 y.fillStyle=css('--fg'); y.fillRect(i*w,0,1,tl.height);}
T.oninput=()=>draw(+T.value);
document.getElementById('play').onclick=e=>{playing=!playing; e.target.textContent=playing?'Pause':'Play';
 const tick=()=>{if(!playing) return; T.value=(+T.value+1)%D.frames.length; draw(+T.value); setTimeout(tick,1000/Math.max(1,D.fps/D.stride));}; tick();};
draw(0);
</script></body></html>"""


def build_viewer(work: Path, with_frames: bool = True, width: int = 640) -> Path:
    """Write work/viewer.html from work/graphs.json (and frames, downscaled, if requested)."""
    import cv2
    g = json.loads((work / "graphs.json").read_text(encoding="utf-8"))
    frames = {}
    if with_frames:
        for f in g["frames"]:
            im = cv2.imread(str(work / "frames" / f"{f['frame']:06d}.jpg"))
            if im is None:
                continue
            h = int(im.shape[0] * width / im.shape[1])
            ok, buf = cv2.imencode(".jpg", cv2.resize(im, (width, h)), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                frames[f["frame"]] = base64.b64encode(buf.tobytes()).decode()
    html = PAGE.replace("__DATA__", json.dumps(g)).replace("__FRAMES__", json.dumps(frames))
    out = work / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out
