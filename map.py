from pathlib import Path
import json, math, heapq
from collections import deque
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----- DATA -----
ROOT = Path(__file__).resolve().parents[2]   # UB-ROUTES/
DATA_DIR = ROOT / "data"
ADJ_PATH = DATA_DIR / "ub_graph.json"
NODES_PATH = DATA_DIR / "ub_nodes.json"
if not ADJ_PATH.exists() or not NODES_PATH.exists():
    raise RuntimeError("data/ub_graph.json эсвэл data/ub_nodes.json олдсонгүй")

adj = {int(k): v for k, v in json.loads(ADJ_PATH.read_text()).items()}
nodes = {int(k): v for k, v in json.loads(NODES_PATH.read_text()).items()}

# ----- HELPERS -----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def nearest_node(lat: float, lon: float):
    best_id, best_d = None, 1e30
    for i, c in nodes.items():
        d = haversine(lat, lon, c["lat"], c["lon"])
        if d < best_d:
            best_id, best_d = i, d
    return best_id, best_d

def bfs(start, goal):
    parent = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal: break
        for e in adj.get(u, []):
            v = e["to"]
            if v not in parent:
                parent[v] = u
                q.append(v)
    if goal not in parent: return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur); cur = parent[cur]
    return list(reversed(path))

def dijkstra(start, goal):
    dist = {start: 0.0}
    parent = {start: None}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal: break
        if d != dist.get(u, 1e18): continue
        for e in adj.get(u, []):
            v, w = e["to"], float(e["w"])
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v], parent[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if goal not in parent: return None, None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur); cur = parent[cur]
    return list(reversed(path)), dist[goal]

# ----- API -----
app = FastAPI(title="UB Routes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteReq(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    algo: str = "dijkstra"  # or "bfs"

@app.get("/health")
def health():
    return {"ok": True, "nodes": len(nodes), "edges": sum(len(v) for v in adj.values())}

@app.get("/nearest")
def api_nearest(lat: float, lon: float):
    nid, d = nearest_node(lat, lon)
    return {"node_id": nid, "distance_m": d, "coord": nodes[nid]}

@app.post("/route")
def api_route(req: RouteReq):
    s_id, s_d = nearest_node(req.start_lat, req.start_lon)
    t_id, t_d = nearest_node(req.end_lat, req.end_lon)
    if req.algo.lower() == "bfs":
        path = bfs(s_id, t_id)
        if not path: raise HTTPException(404, "Path not found")
        poly = [(nodes[i]["lat"], nodes[i]["lon"]) for i in path]
        return {"path_node_ids": path, "length_m": None,
                "start_snap_m": s_d, "end_snap_m": t_d, "polyline": poly}
    elif req.algo.lower() == "dijkstra":
        path, L = dijkstra(s_id, t_id)
        if not path: raise HTTPException(404, "Path not found")
        poly = [(nodes[i]["lat"], nodes[i]["lon"]) for i in path]
        return {"path_node_ids": path, "length_m": L,
                "start_snap_m": s_d, "end_snap_m": t_d, "polyline": poly}
    else:
        raise HTTPException(400, "algo must be 'bfs' or 'dijkstra'")
