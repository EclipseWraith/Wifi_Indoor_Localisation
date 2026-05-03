#!/usr/bin/env python3
"""
Indoor Navigation Server — Cosine KNN Engine
----------------------------------------------
- Uses Cosine Similarity KNN (K=5) for WiFi localization
- Winner of 27-model LOLO comparison: 3.51m mean, 80% within 5m
- Robust to phone-to-phone signal strength differences
- Dashboard shows live blue dot + Dijkstra navigation

Usage:
    python3 app.py
"""

import os
import sys
import json
import math
import time
import heapq
import socket
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

from knn import CosineKNN

app = Flask(__name__)

# ── Globals ──
MODEL = None
RECORDS = []
GRAPH = None

# Tracking state
STATE = {
    "x": 0.0, "y": 0.0,
    "node": "unknown",
    "label": "",
    "confidence": 0.0,
    "ts": None,
    "filter_info": "",
}

# ── Position Smoothing ──
SMOOTH_ALPHA = 0.35       # EMA weight for new readings (0.0=ignore new, 1.0=no smoothing)
MAX_JUMP_DIST = 8.0       # If prediction jumps more than this (meters), dampen heavily
HISTORY = []              # Keep last N predictions for averaging
HISTORY_SIZE = 5
FIRST_FIX = True          # First reading gets no smoothing


# ──────────────────────── Graph building ────────────────────────
def build_graph(records, max_d=4.0):
    """Build a graph of unique locations with edges between nearby points."""
    locs = {}
    for r in records:
        key = (round(r["x"], 1), round(r["y"], 1))
        if key not in locs:
            locs[key] = []
        locs[key].append(r.get("label", ""))

    nodes = {}
    pos = []
    
    BLOCKED_EDGE_PATTERNS = [
        # Wing A explicit blockades
        ("A-411 opposite", "EL-41B"),
        ("A-411 opposite", "EL-41A"),
        ("A-412", "Fire Exit"),
        ("A-413", "A-410"),
        ("A-415", "A-409"),
        ("A-410", "A-414"),
        ("A-414", "A-409"),
        ("A-404 opposite", "A-416"),
        ("A-418", "A-403 opposite"),
        ("A-401", "HCD Board"),
        ("HCD Board", "Elevator 4"),
        ("Elevator 4", "Fire Extinguisher Elevator"),
        
        # Wing B explicit blockades
        ("B-411 opposite", "EL-41B"),
        ("B-412", "Stairs B Exit"),
        ("B-413", "B-410"),
        ("B-415", "B-409"),
        ("B-410", "B-414"),
        ("B-414", "B-409"),
        ("B-404 opposite", "B-416"),
        ("B-404 opposite", "B-417"),
        ("B-418", "B-403 opposite"),
        ("B-401", "CAI Board"),
        ("CAI Board", "Elevator 4"),
        ("B-413", "Stairs B Exit"),
        ("B-413", "EL-41B"),
        ("CAI (Wing B)", "B-401"),
        ("B-403 opposite", "B-417 door"),   
    ]
    BLOCKADE_NODES = ["pillar1", "pillar2", "A - 419 pillar", "after pillar lobby"]

    for (x, y), labels in locs.items():
        labels = [l for l in labels if l]
        name = max(set(labels), key=labels.count) if labels else f"pt_{x}_{y}"
        name = name.strip()
        base, c = name, 2
        while name in nodes:
            name = f"{base}_{c}"
            c += 1
        if not any(b.lower() in name.lower() for b in BLOCKADE_NODES):
            nodes[name] = {"x": x, "y": y}
            pos.append((name, x, y))

    edges = []
    added = set()
    
    def is_blocked(n1, n2):
        n1_lower = n1.lower()
        n2_lower = n2.lower()
        for p1, p2 in BLOCKED_EDGE_PATTERNS:
            p1_l = p1.lower()
            p2_l = p2.lower()
            if (p1_l in n1_lower and p2_l in n2_lower) or (p1_l in n2_lower and p2_l in n1_lower):
                return True
        return False
        
    for i, (n1, x1, y1) in enumerate(pos):
        ds = sorted([(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), n2)
                     for j, (n2, x2, y2) in enumerate(pos) if i != j])
        cn = 0
        for d, n2 in ds:
            ek = tuple(sorted([n1, n2]))
            if ek in added or is_blocked(n1, n2):
                continue
            if d <= max_d or cn < 1:
                edges.append({"from": n1, "to": n2, "distance": round(d, 2)})
                added.add(ek)
                cn += 1
            if cn >= 3:
                break
    return {"nodes": nodes, "edges": edges}


def find_nearest_node(x, y):
    best, bd = None, math.inf
    for name, nd in GRAPH["nodes"].items():
        d = math.sqrt((nd["x"] - x) ** 2 + (nd["y"] - y) ** 2)
        if d < bd:
            bd = d
            best = name
    return best


def dijkstra_path(start, end):
    adj = {n: [] for n in GRAPH["nodes"]}
    for e in GRAPH["edges"]:
        if e["from"] in adj and e["to"] in adj:
            adj[e["from"]].append((e["to"], e["distance"]))
            adj[e["to"]].append((e["from"], e["distance"]))
    dist = {n: math.inf for n in adj}
    prev = {n: None for n in adj}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[end] == math.inf:
        return None, math.inf
    path = []
    cur = end
    while cur:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[end]


def make_directions(path):
    if not path or len(path) < 2:
        return []
    nodes = GRAPH["nodes"]
    dirs = [f"Start at {path[0]}"]
    for i in range(len(path) - 1):
        c, n = nodes[path[i]], nodes[path[i + 1]]
        seg = round(math.sqrt((n["x"] - c["x"]) ** 2 + (n["y"] - c["y"]) ** 2), 1)
        if i == 0:
            dirs.append(f"Walk {seg}m toward {path[i+1]}")
        else:
            p = nodes[path[i - 1]]
            a1 = math.atan2(c["y"] - p["y"], c["x"] - p["x"])
            a2 = math.atan2(n["y"] - c["y"], n["x"] - c["x"])
            diff = math.degrees(a2 - a1) % 360
            if diff > 180:
                diff -= 360
            t = "Continue straight" if abs(diff) < 30 else ("Turn left" if diff > 0 else "Turn right")
            dirs.append(f"{t}, walk {seg}m to {path[i+1]}")
    dirs.append(f"Arrived at {path[-1]}")
    return dirs


# ──────────────────────── Init ────────────────────────
def init():
    global MODEL, RECORDS, GRAPH
    import pickle
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(script_dir, "probabilistic_map.pkl")
    
    if not os.path.exists(map_path):
        print(f"ERROR: {map_path} not found. Run build_prob_map.py first.")
        sys.exit(1)
    # 1. Init Model
    print("[Init] Loading KNN with TF-IDF and Exponential Smoothing (K=3)")
    MODEL = CosineKNN(map_path)
    
    # Build graph from locations in the probabilistic map
    with open(map_path, "rb") as f:
        pmap = pickle.load(f)
        
    pseudo_records = []
    for (x, y), data in pmap.items():
        pseudo_records.append({"x": x, "y": y, "label": data["label"]})
        
    GRAPH = build_graph(pseudo_records)
    print(f"[Init] Graph: {len(GRAPH['nodes'])} nodes, {len(GRAPH['edges'])} edges")
    print(f"[Init] Ready for live localization")


def save_fingerprint(data):
    if not data or not data.get("networks"): 
        return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(script_dir, "fingerprints_collected.json")
    try:
        with open(out_file, "r") as f:
            records = json.load(f)
    except Exception:
        records = []
    
    x = data.get("x", 0.0)
    y = data.get("y", 0.0)
    label = data.get("label", "Unknown")
    networks = data.get("networks", {})
    
    records.append({
        "x": x,
        "y": y,
        "label": label,
        "networks": networks,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)


# ──────────────────────── Routes ────────────────────────
@app.route("/api/locate", methods=["POST"])
def api_locate():
    global STATE
    
    if MODEL is None:
        return jsonify({"success": False, "error": "No model"}), 500

    data = request.get_json() or {}
    scan = data.get("networks", {})
    imu = data.get("imu", {}) or {}

    if not scan:
        return jsonify({"success": False, "error": "No networks"}), 400

    # Run Cosine KNN prediction
    heading = imu.get("heading_deg", 0.0) if imu else 0.0
    result = MODEL.predict(scan, heading)
    pred_x = result["x"]
    pred_y = result["y"]

    # ── Position Smoothing ──
    global FIRST_FIX
    if FIRST_FIX:
        # First reading: snap directly, no smoothing
        smooth_x, smooth_y = pred_x, pred_y
        FIRST_FIX = False
    else:
        # Calculate jump distance from current position
        jump = math.sqrt((pred_x - STATE["x"]) ** 2 + (pred_y - STATE["y"]) ** 2)
        
        # Adjust alpha based on confidence and jump distance
        conf = result.get("confidence", 0.5)
        alpha = SMOOTH_ALPHA * conf  # Higher confidence = trust new reading more
        
        if jump > MAX_JUMP_DIST:
            # Big jump detected — dampen heavily
            alpha *= 0.3
        
        # EMA blend
        smooth_x = STATE["x"] + alpha * (pred_x - STATE["x"])
        smooth_y = STATE["y"] + alpha * (pred_y - STATE["y"])
        
        # Also average with recent history for extra stability
        HISTORY.append((pred_x, pred_y, conf))
        if len(HISTORY) > HISTORY_SIZE:
            HISTORY.pop(0)
        if len(HISTORY) >= 3:
            total_w = 0
            avg_x, avg_y = 0, 0
            for hx, hy, hc in HISTORY:
                w = max(hc, 0.1)
                avg_x += hx * w
                avg_y += hy * w
                total_w += w
            avg_x /= total_w
            avg_y /= total_w
            # Blend EMA with history average
            smooth_x = 0.6 * smooth_x + 0.4 * avg_x
            smooth_y = 0.6 * smooth_y + 0.4 * avg_y
    
    # Snap to nearest graph node for label
    node = find_nearest_node(smooth_x, smooth_y)
    label = result["label"]

    STATE["x"] = round(smooth_x, 2)
    STATE["y"] = round(smooth_y, 2)
    STATE["node"] = node or "unknown"
    STATE["label"] = label
    STATE["confidence"] = result.get("confidence", 0.8)
    STATE["ts"] = datetime.now().strftime("%H:%M:%S")
    STATE["filter_info"] = "KNN TF-IDF K=3 + EMA Smoothing"

    print(f"[LOCATE] Raw:({pred_x},{pred_y}) Smooth:({STATE['x']},{STATE['y']}) Node: {label} Conf: {result['confidence']:.2f}")

    return jsonify({
        "success": True,
        "x": pred_x, "y": pred_y,
        "label": STATE["label"],
        "confidence": STATE["confidence"],
        "nearest_node": node,
        "fusion_info": STATE["filter_info"],
    })


@app.route("/api/manual", methods=["POST"])
def api_manual():
    """Also accepts manual scans (for extending data collection if needed)."""
    data = request.get_json() or {}
    save_fingerprint(data)
    return jsonify({"success": True, "note": "manual endpoint - data collection only"})


@app.route("/api/current")
def api_current():
    return jsonify(STATE)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global STATE
    STATE = {
        "x": 0.0, "y": 0.0,
        "node": "unknown",
        "label": "",
        "confidence": 0.0,
        "ts": None,
        "filter_info": "",
    }
    return jsonify({"success": True})


@app.route("/api/navigate", methods=["POST"])
def api_navigate():
    data = request.get_json() or {}
    dest = data.get("to", "")
    start = data.get("from", "")
    if not start:
        start = STATE.get("node", "unknown")
        if start == "unknown":
            start = find_nearest_node(STATE["x"], STATE["y"])
    if not start or not dest:
        return jsonify({"success": False, "error": "Need start and dest"}), 400
    if start not in GRAPH["nodes"]:
        return jsonify({"success": False, "error": f"'{start}' not found"}), 404
    if dest not in GRAPH["nodes"]:
        return jsonify({"success": False, "error": f"'{dest}' not found"}), 404
    path, dist = dijkstra_path(start, dest)
    if not path:
        return jsonify({"success": False, "error": "No path"}), 404
    return jsonify({
        "success": True,
        "path": [{"name": n, "x": GRAPH["nodes"][n]["x"], "y": GRAPH["nodes"][n]["y"]} for n in path],
        "path_names": path,
        "total_distance": round(dist, 2),
        "directions": make_directions(path),
    })


@app.route("/api/graph")
def api_graph():
    return jsonify(GRAPH or {"nodes": {}, "edges": []})


@app.route("/api/nodes")
def api_nodes():
    if not GRAPH: return jsonify([])
    bad_words = ["opposite", "lobby", "start", "end", "pillar", "between", "beside"]
    
    seen_names = set()
    result = []
    
    import re
    for node in sorted(GRAPH["nodes"].keys()):
        node_lower = node.lower()
        if any(w in node_lower for w in bad_words):
            continue
            
        display_name = node
        # Remove Door tags
        display_name = re.sub(r' door\d*$', '', display_name, flags=re.IGNORECASE)
        
        # Aggressive Grouping with Wing Separation
        dn_lower = display_name.lower()
        
        if "watercooler" in dn_lower:
            if "wing a" in dn_lower or "washroom" in dn_lower: display_name = "Watercooler (Wing A)"
            else: display_name = "Watercooler (Wing B)"
                
        elif "stair" in dn_lower and "fire" not in dn_lower:
            if "open hall" in dn_lower: display_name = "Stairs (Open Hall)"
            else: display_name = "Stairs (Wing B)" # All other non-fire stairs (FHC, Window, Stairs B) are Wing B
                
        elif "fire exit" in dn_lower or "firexit" in dn_lower:
            if "wing b" in dn_lower: display_name = "Fire Exit (Wing B)"
            else: display_name = "Fire Exit (Wing A)" # Fire Exit Stairs/Way are in Wing A
                
        elif "fire extinguisher" in dn_lower:
            if " a" in dn_lower: display_name = "Fire Extinguisher (Wing A)"
            elif " b" in dn_lower: display_name = "Fire Extinguisher (Wing B)"
            else: display_name = "Fire Extinguisher (Elevator)"
                
        elif "elevator" in dn_lower and "extinguisher" not in dn_lower: display_name = "Elevator"
        elif "cai" in dn_lower: display_name = "CAI"
        elif "hcd" in dn_lower: display_name = "HCD"
            
        if display_name not in seen_names:
            seen_names.add(display_name)
            result.append({"id": node, "name": display_name})
            
    return jsonify(result)


@app.route("/")
def index():
    return render_template_string(DASH)


# ──────────────────────── Dashboard ────────────────────────
DASH = r"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Nav</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0b0d14;color:#eee;font-family:-apple-system,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
.bar{display:flex;gap:6px;padding:8px 10px;background:#1a1a2e;align-items:center;flex-wrap:wrap;border-bottom:1px solid #252d44;z-index:10}
.bar select,.bar button{padding:7px 12px;border-radius:8px;border:1px solid #252d44;background:#171c2c;color:#eee;font-size:13px}
.bar button{cursor:pointer;font-weight:600}
.bar .g{background:#1D9E75;border-color:#1D9E75;color:#fff}
.bar .b{background:#3B82F6;border-color:#3B82F6;color:#fff}
.bar .lbl{font-size:11px;color:#7b8299}
.mw{flex:1;position:relative;overflow:hidden;cursor:grab;background:#0b0d14}
.mw:active{cursor:grabbing}
canvas{position:absolute;top:0;left:0}
.pn{background:#111520;padding:10px 14px;border-top:1px solid #252d44;max-height:30vh;overflow-y:auto;z-index:10}
#loc{color:#3B82F6;font-weight:700;font-size:14px;margin-bottom:4px}
#info{font-size:11px;color:#fb923c;margin-bottom:4px}
#dirs{font-size:13px;color:#aaa;line-height:1.7}
.dt{color:#fb923c;font-weight:600;margin-bottom:6px}
</style>
</head><body>
<div class="bar">
<span class="lbl">To:</span>
<select id="dest"></select>
<button class="g" onclick="nav()">Navigate</button>
<button onclick="stopN()">Stop</button>
<button onclick="resetState()">Reset</button>
<span class="lbl" id="live" style="margin-left:auto">--</span>
</div>
<div class="mw" id="wrap"><canvas id="cv"></canvas></div>
<div class="pn">
<div id="loc">Waiting for scan...</div>
<div id="info"></div>
<div id="dirs"></div>
</div>
<script>
var cv=document.getElementById('cv'),c=cv.getContext('2d'),W=document.getElementById('wrap');
var G=null,cur=null,NP=null,dest=null;
var Z=1,PX=0,PY=0,dr=false,DX,DY;
var MX=0,MXX=1,MY=0,MXY=1;
var edgeCache=null,nodeCache=null;
var lastLocStr="";

// Smooth animation variables
var dispX=0,dispY=0,targetX=0,targetY=0,dotReady=false;
var LERP=0.12; // interpolation speed (0=frozen, 1=instant)
var animId=null;

async function init(){
  G=await(await fetch('/api/graph')).json();
  var ns=await(await fetch('/api/nodes')).json();
  var s=document.getElementById('dest');
  ns.forEach(function(n){var o=document.createElement('option');o.value=n.id;o.textContent=n.name;s.appendChild(o)});
  var xs=[],ys=[];
  for(var k in G.nodes){xs.push(G.nodes[k].x);ys.push(G.nodes[k].y)}
  MX=Math.min.apply(null,xs)-3;MXX=Math.max.apply(null,xs)+3;
  MY=Math.min.apply(null,ys)-3;MXY=Math.max.apply(null,ys)+3;
  buildCaches();
  rs();draw();
  setInterval(poll,500);
  animLoop();
}

function buildCaches(){
  if(!G)return;
  edgeCache=[];
  for(var i=0;i<G.edges.length;i++){
    var e=G.edges[i];
    if(G.nodes[e.from]&&G.nodes[e.to]){
      edgeCache.push([G.nodes[e.from].x,G.nodes[e.from].y,G.nodes[e.to].x,G.nodes[e.to].y]);
    }
  }
  nodeCache=[];
  for(var k in G.nodes){nodeCache.push([G.nodes[k].x,G.nodes[k].y,k])}
}

function rs(){cv.width=W.clientWidth;cv.height=W.clientHeight}
window.onresize=function(){rs();draw()};

function w2s(wx,wy){
  var cw=cv.width,ch=cv.height;
  var rx=MXX-MX||1,ry=MXY-MY||1;
  var bs=Math.min((cw-60)/rx,(ch-60)/ry);
  var s=bs*Z,cx=cw/2+PX,cy=ch/2+PY;
  var mx=(MX+MXX)/2,my=(MY+MXY)/2;
  return[cx+(wx-mx)*s,cy-(wy-my)*s];
}

function draw(){
  if(!G)return;
  var w=cv.width,h=cv.height;
  c.clearRect(0,0,w,h);
  c.fillStyle='#0b0d14';c.fillRect(0,0,w,h);

  c.strokeStyle='rgba(37,45,68,0.5)';c.lineWidth=0.5;
  c.beginPath();
  for(var i=0;i<edgeCache.length;i++){
    var e=edgeCache[i];
    var p1=w2s(e[0],e[1]),p2=w2s(e[2],e[3]);
    c.moveTo(p1[0],p1[1]);c.lineTo(p2[0],p2[1]);
  }
  c.stroke();

  if(NP&&NP.length>=2){
    c.strokeStyle='#E74C3C';c.lineWidth=3;c.lineCap='round';
    c.beginPath();
    var s0=w2s(NP[0].x,NP[0].y);c.moveTo(s0[0],s0[1]);
    for(var i=1;i<NP.length;i++){var si=w2s(NP[i].x,NP[i].y);c.lineTo(si[0],si[1])}
    c.stroke();
    c.fillStyle='#22c55e';c.beginPath();c.arc(s0[0],s0[1],6,0,6.28);c.fill();
    var se=w2s(NP[NP.length-1].x,NP[NP.length-1].y);
    c.fillStyle='#E74C3C';c.beginPath();c.arc(se[0],se[1],6,0,6.28);c.fill();
  }

  c.fillStyle='#1D9E75';
  for(var i=0;i<nodeCache.length;i++){
    var n=nodeCache[i];
    var p=w2s(n[0],n[1]);
    c.beginPath();c.arc(p[0],p[1],2.5,0,6.28);c.fill();
  }

  if(Z>=1.5){
    c.fillStyle='rgba(200,200,220,0.6)';c.font='8px sans-serif';c.textAlign='center';
    for(var i=0;i<nodeCache.length;i++){
      var n=nodeCache[i];
      var p=w2s(n[0],n[1]);
      c.fillText(n[2].substring(0,18),p[0],p[1]-7);
    }
  }

  if(dotReady){
    var p=w2s(dispX,dispY);
    // Confidence-based pulse
    var conf=cur?cur.confidence:0.5;
    var pulseR=15+Math.max(5,(1-conf)*15);
    c.fillStyle='rgba(59,130,246,0.12)';
    c.beginPath();c.arc(p[0],p[1],pulseR,0,6.28);c.fill();
    c.fillStyle='#3B82F6';c.strokeStyle='#fff';c.lineWidth=2;
    c.beginPath();c.arc(p[0],p[1],7,0,6.28);c.fill();c.stroke();
    c.fillStyle='#fff';c.font='bold 8px sans-serif';c.textAlign='center';
    c.fillText('ME',p[0],p[1]+3);
  }
}

W.onwheel=function(e){e.preventDefault();Z*=e.deltaY<0?1.2:.8;Z=Math.max(.3,Math.min(12,Z));draw()};
W.onmousedown=function(e){dr=true;DX=e.clientX;DY=e.clientY};
W.onmousemove=function(e){if(!dr)return;PX+=e.clientX-DX;PY+=e.clientY-DY;DX=e.clientX;DY=e.clientY;draw()};
W.onmouseup=function(){dr=false};
W.onmouseleave=function(){dr=false};
var lt=0;
W.ontouchstart=function(e){
  if(e.touches.length==1){dr=true;DX=e.touches[0].clientX;DY=e.touches[0].clientY}
  if(e.touches.length==2){lt=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY)}
};
W.ontouchmove=function(e){
  e.preventDefault();
  if(e.touches.length==1&&dr){PX+=e.touches[0].clientX-DX;PY+=e.touches[0].clientY-DY;DX=e.touches[0].clientX;DY=e.touches[0].clientY;draw()}
  if(e.touches.length==2){var d=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);if(lt>0){Z*=d/lt;Z=Math.max(.3,Math.min(12,Z));draw()}lt=d}
};
W.ontouchend=function(){dr=false;lt=0};

function animLoop(){
  if(dotReady){
    var dx=targetX-dispX, dy=targetY-dispY;
    if(Math.abs(dx)>0.01||Math.abs(dy)>0.01){
      dispX+=dx*LERP;
      dispY+=dy*LERP;
      draw();
    }
  }
  requestAnimationFrame(animLoop);
}

async function poll(){
  try{
    var r=await fetch('/api/current');var d=await r.json();
    var ls=d.x+','+d.y+','+d.node+','+d.confidence;
    if(ls!==lastLocStr){
      cur=d;lastLocStr=ls;
      targetX=d.x;targetY=d.y;
      if(!dotReady){dispX=d.x;dispY=d.y;dotReady=true;}
      document.getElementById('loc').textContent=
        d.label+' ('+d.x.toFixed(1)+', '+d.y.toFixed(1)+')  conf:'+(d.confidence*100).toFixed(0)+'%';
      document.getElementById('info').textContent=d.filter_info||'';
      document.getElementById('live').textContent=d.label+' | '+d.ts;
      if(dest)updateNav();
      draw();
    }
  }catch(e){}
}

var arrivedFrames = 0;
async function nav(){
  dest=document.getElementById('dest').value;
  arrivedFrames = 0;
  if(!dest)return;
  await updateNav();
}

async function updateNav(){
  if(!dest)return;
  try{
    var body={to:dest};
    if(cur&&cur.node!=='unknown')body.from=cur.node;
    var r=await fetch('/api/navigate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    var d=await r.json();
    if(d.success){
      NP=d.path;
      if(d.total_distance < 2.5) {
        arrivedFrames++;
      } else {
        arrivedFrames = 0;
      }
      
      if(arrivedFrames >= 10) {
        document.getElementById('dirs').innerHTML = '<div style="color:#10B981;font-weight:bold;font-size:1.2em;margin-top:10px;text-align:center;">🎉 Location Reached!</div>';
        NP = []; // clear the green line
        dest = ""; // clear destination
        document.getElementById('dest').value = ""; // clear dropdown
        arrivedFrames = 0;
      } else {
        var h='<div class="dt">'+d.total_distance+'m total</div>';
        if(arrivedFrames > 0) {
           h += '<div style="color:#10B981;font-size:0.95em;margin-bottom:8px;">Validating arrival... (' + arrivedFrames + '/10)</div>';
        }
        d.directions.forEach(function(s,i){h+='<div>'+(i+1)+'. '+s+'</div>'});
        document.getElementById('dirs').innerHTML=h;
      }
      draw();
    }
  }catch(e){}
}

function stopN(){dest=null;NP=null;document.getElementById('dirs').innerHTML='';draw()}
async function resetState(){await fetch('/api/reset',{method:'POST'});cur=null;lastLocStr='';draw()}

init();
</script>
</body></html>"""


if __name__ == "__main__":
    print("=" * 55)
    print("  Indoor Navigation Server with IMU Fusion")
    print("=" * 55)
    init()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        print(f"\n  Dashboard:  http://localhost:5001")
        print(f"  On phone:   http://{ip}:5001")
    except Exception:
        print(f"\n  Dashboard: http://localhost:5001")
    print()
    app.run(host="0.0.0.0", port=5001, debug=False)