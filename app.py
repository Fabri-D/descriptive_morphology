# app.py
import os
import numpy as np
from PIL import Image

from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go


# ----------------------------
# Config
# ----------------------------
ROOT_DIR = os.environ.get("INSECT_MODELS_DIR", "./3d_models")

# “familias” lógicas (para tu narrativa de 4)
# Cada entry busca por substring en el nombre del archivo .obj
FAMILY_QUERIES = {
    "Coleoptera (beetle)": ["cyclommatinus", "bicolor"],
    "Diptera (fly)": ["lucilia", "caesar"],
    "Hymenoptera (wasp)": ["pepsis"],  # mañana puede ser pepsis_terminata
    "Lepidoptera (butterfly)": ["ornithoptera", "alexandrae"],  # mañana
}

WASP_CAMERA_PRESET = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-0.9181418596667209, y=1.165610252570674, z=0.6681478386418584),
    projection=dict(type="perspective"),
)

CAMERA_PRESETS = {
    "Coleoptera (beetle)": {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {
            "x": 1.2018496829718561,
            "y": 2.714496593465817,
            "z": 0.823263610052456
        },
        "projection": {"type": "perspective"},
    },
    "Diptera (fly)": {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {
            "x": -1.3106698789567073,
            "y": 1.349977094964864,
            "z": 0.7633008265318245
        },
        "projection": {"type": "perspective"},
    },
    "Hymenoptera (wasp)": WASP_CAMERA_PRESET,
    "Lepidoptera (butterfly)": {
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {
            "x": 1.0811848886971742,
            "y": 3.156032777358698,
            "z": -0.026646052497576386
        },
        "projection": {"type": "perspective"},
    },
}

""" BEETLE_CALLOUTS = [
    {
        "label": "Mandibles",
        "pos": (0.0, 0.42, 0.48),
    },
    {
        "label": "Pronotum",
        "pos": (0.0, 0.25, 0.30),
    },
    {
        "label": "Elytra (hardened wings)",
        "pos": (-0.12, -0.15, 0.10),
    },
]

FLY_CALLOUTS = [
    {
        "label": "Compound eyes",
        "pos": (0.18, 0.28, 0.35),
    },
    {
        "label": "Thorax (flight muscles)",
        "pos": (0.0, 0.10, 0.20),
    },
    {
        "label": "Wings (single pair)",
        "pos": (-0.35, 0.05, 0.18),
    },
]

WASP_CALLOUTS = [
    {
        "label": "Head (compound eyes / antennae)",
        "pos": (0.05, 0.32, 0.34),
    },
    {
        "label": "Thorax (flight muscles)",
        "pos": (0.00, 0.12, 0.22),
    },
    {
        "label": "Petiole (the 'waist')",
        "pos": (0.02, -0.02, 0.15),
    },
    {
        "label": "Abdomen (gaster)",
        "pos": (0.00, -0.22, 0.12),
    },
    {
        "label": "Wings (forewing)",
        "pos": (-0.30, 0.06, 0.22),
    },
] """

BEETLE_CALLOUTS = []
FLY_CALLOUTS = []
WASP_CALLOUTS = []
BUTTERFLY_CALLOUTS = []
CALLOUTS_BY_FAMILY = {
    "Coleoptera (beetle)": BEETLE_CALLOUTS,
    "Diptera (fly)": FLY_CALLOUTS,
    "Hymenoptera (wasp)": WASP_CALLOUTS,   # si todavía no lo tenés, ponelo vacío []
    "Lepidoptera (butterfly)": BUTTERFLY_CALLOUTS,  # idem
}






# ----------------------------
# OBJ loader (simple, robust enough)
# ----------------------------
def _parse_face_token(tok: str) -> int:
    """
    Face tokens can be like:
      "12" or "12/3" or "12/3/9" or "12//9"
    OBJ uses 1-based indexing; negatives allowed (relative).
    We return the vertex index (0-based) later after adjusting.
    """
    if "/" in tok:
        tok = tok.split("/")[0]
    return int(tok)



def normalize_vertices(V: np.ndarray):
    """
    Center + scale so the largest bbox dimension becomes 1.0
    """
    # center
    center = V.mean(axis=0)
    Vc = V - center

    # scale
    mins = Vc.min(axis=0)
    maxs = Vc.max(axis=0)
    span = float(np.max(maxs - mins))
    if span <= 1e-9:
        return Vc
    return Vc / span


def find_mtl_for_obj(obj_path: str):
    """
    Busca el .mtl asociado:
    - Primero intenta leer 'mtllib ...' dentro del obj
    - Si no encuentra, intenta mismo nombre .mtl al lado del obj
    """
    obj_dir = os.path.dirname(obj_path)

    # 1) parse mtllib dentro del OBJ
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("mtllib "):
                    # puede haber espacios; mtllib suele ser el nombre de archivo
                    mtl_name = line.split(None, 1)[1].strip()
                    mtl_path = os.path.join(obj_dir, mtl_name)
                    if os.path.exists(mtl_path):
                        return mtl_path
    except Exception:
        pass

    # 2) fallback: mismo nombre que el obj
    base, _ = os.path.splitext(obj_path)
    mtl_path = base + ".mtl"
    return mtl_path if os.path.exists(mtl_path) else None


def add_callouts(fig, callouts, camera):
    if not camera:
        # fallback simple si no hay cámara
        camera = dict(
            eye=dict(x=1.6,y=1.6,z=1.0),
            up=dict(x=0,y=0,z=1),
            center=dict(x=0,y=0,z=0),
            projection=dict(type="perspective"),
        )
    eye = np.array([camera["eye"]["x"], camera["eye"]["y"], camera["eye"]["z"]], dtype=np.float32)
    upv = np.array([camera["up"]["x"], camera["up"]["y"], camera["up"]["z"]], dtype=np.float32)
    ctr = np.array([camera["center"]["x"], camera["center"]["y"], camera["center"]["z"]], dtype=np.float32)

    forward = ctr - eye
    forward /= (np.linalg.norm(forward) + 1e-9)

    right = np.cross(forward, upv)
    right /= (np.linalg.norm(right) + 1e-9)

    up2 = np.cross(right, forward)
    up2 /= (np.linalg.norm(up2) + 1e-9)

    for c in callouts:
        x,y,z = c["pos"]
        anchor = np.array([x,y,z], dtype=np.float32)

        d  = float(c.get("d", 0.25))
        dx = float(c.get("dx", 0.0))
        dy = float(c.get("dy", 0.0))

        away = anchor - eye
        away /= (np.linalg.norm(away) + 1e-9)

        label_pos = anchor + d*away + dx*right + dy*up2
        lx,ly,lz = label_pos.tolist()

        # punto ancla
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers",
            marker=dict(size=4, color="black"),
            showlegend=False
        ))

        # línea
        fig.add_trace(go.Scatter3d(
            x=[x, lx], y=[y, ly], z=[z, lz],
            mode="lines",
            line=dict(width=4, color="red"),
            showlegend=False
        ))

        # texto
        fig.add_trace(go.Scatter3d(
            x=[lx], y=[ly], z=[lz],
            mode="text",
            text=[c["label"]],
            textposition="middle right",
            textfont=dict(size=14, color="black"),
            showlegend=False
        ))




def compute_label_point(anchor, camera, d=0.25):
    ax, ay, az = anchor
    eye = camera.get("eye", {"x":1.6,"y":1.6,"z":1.0})
    center = camera.get("center", {"x":0,"y":0,"z":0})

    vx = eye["x"] - center["x"]
    vy = eye["y"] - center["y"]
    vz = eye["z"] - center["z"]
    n = (vx*vx + vy*vy + vz*vz) ** 0.5
    if n < 1e-9:
        return (ax, ay, az)

    ux, uy, uz = vx/n, vy/n, vz/n
    return (ax + d*ux, ay + d*uy, az + d*uz)




def parse_mtl_maps(obj_path: str, mtl_path: str):
    """
    Devuelve:
      maps[material] = {"map_Kd": <path_png_or_None>}
    """
    if not mtl_path or not os.path.exists(mtl_path):
        return {}

    #print("[DEBUG] --- MTL texture lines ---")
    #with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        #for line in f:
            #low = line.lower()
            #if "map_" in low or "adobe_map_" in low or "bump" in low:
                #print("[DEBUG]", line.strip())
            #if low.startswith("kd "):
                #print("[DEBUG]", line.strip())
    #print("[DEBUG] ------------------------")

    

    obj_dir = os.path.dirname(obj_path)
    maps = {}
    current = None

    def resolve(tex_ref: str):
        tex_ref = tex_ref.strip().strip('"')
        p1 = os.path.normpath(os.path.join(obj_dir, tex_ref))
        if os.path.exists(p1): return p1
        p2 = os.path.normpath(os.path.join(obj_dir, os.path.basename(tex_ref)))
        if os.path.exists(p2): return p2
        return None

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()

            if low.startswith("newmtl "):
                current = line.split(None, 1)[1].strip()
                maps.setdefault(current, {})

            # dentro de parse_mtl_maps(), en el loop:
            elif low.startswith("map_kd ") and current:
                tex_ref = line.split(None, 1)[1].strip()
                maps[current]["map_Kd"] = resolve(tex_ref)

            elif (low.startswith("adobe_map_base_color ") or low.startswith("adobe_map_basecolor ")) and current:
                tex_ref = line.split(None, 1)[1].strip()
                maps[current]["map_Kd"] = resolve(tex_ref)   # lo guardamos como si fuera map_Kd

            elif low.startswith("adobe_map_albedo ") and current:
                tex_ref = line.split(None, 1)[1].strip()
                maps[current]["map_Kd"] = resolve(tex_ref)

            elif low.startswith("adobe_map_diffuse ") and current:
                tex_ref = line.split(None, 1)[1].strip()
                maps[current]["map_Kd"] = resolve(tex_ref)

            elif low.startswith("kd ") and current:
                parts = line.split()
                if len(parts) >= 4:
                    r = float(parts[1]); g = float(parts[2]); b = float(parts[3])
                    # Kd suele venir 0..1
                    maps[current]["Kd"] = (r, g, b)

    # --- Fallback: si un material no trae map_Kd pero existe un *baseColor.png* con su nombre ---
    # Busca archivos tipo "*fur*basecolor*.png" o "*fur*_Mat_baseColor.png" en subcarpeta del modelo
    model_subdir = os.path.join(os.path.dirname(obj_path), os.path.splitext(os.path.basename(obj_path))[0])
    # En tu caso: ...\AdobeStock_319491375\Pepsis_terminata_1\
    if os.path.isdir(model_subdir):
        pngs = [fn for fn in os.listdir(model_subdir) if fn.lower().endswith(".png")]

        for mat, d in maps.items():
            if d.get("map_Kd"):
                continue

            key = mat.lower()
            # Heurísticas: "fur", "body", "wings" suelen estar en el nombre del archivo
            candidates = [fn for fn in pngs if ("basecolor" in fn.lower() and any(tok in fn.lower() for tok in key.split("_")))]
            if not candidates:
                # fallback más simple: si el material contiene "fur", buscamos "fur" en el filename
                if "fur" in key:
                    candidates = [fn for fn in pngs if ("fur" in fn.lower() and "basecolor" in fn.lower())]

            if candidates:
                # elegimos el más corto (suele ser el principal)
                chosen = sorted(candidates, key=len)[0]
                d["map_Kd"] = os.path.join(model_subdir, chosen)



    return maps

def load_texture_cached(path: str, cache: dict):
    if not path:
        return None
    if path in cache:
        return cache[path]
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img, dtype=np.uint8)
    cache[path] = (arr, img.size[0], img.size[1])
    return cache[path]


def _parse_face_token_full(tok: str):
    # "v", "v/vt", "v//vn", "v/vt/vn"
    parts = tok.split("/")
    v = int(parts[0]) if parts[0] else 0
    vt = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return v, vt  # ignoramos vn

def load_obj_mesh_uv_facecolors(obj_path: str):
    """
    Retorna:
      V (N,3)
      F (M,3)
      face_mtl (M) material name por triángulo
      face_uv  (M,2) uv centroid por triángulo (0..1)
    """
    verts = []
    uvs = []
    faces = []
    face_mtl = []
    face_uv = []

    current_mtl = None

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])

            elif line.startswith("vt "):
                p = line.split()
                # OBJ: vt u v [w]
                uvs.append([float(p[1]), float(p[2])])

            elif line.lower().startswith("usemtl "):
                current_mtl = line.split(None, 1)[1].strip()

            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue

                # parse v + vt indices (pueden ser 0 si no hay vt)
                vv = [_parse_face_token_full(t) for t in parts]

                nV = len(verts)
                nT = len(uvs)

                v_idx = []
                t_idx = []
                for v_i, vt_i in vv:
                    # v index
                    if v_i < 0: v0 = nV + v_i
                    else:       v0 = v_i - 1
                    v_idx.append(v0)

                    # vt index (puede venir 0)
                    if vt_i == 0:
                        t_idx.append(None)
                    else:
                        if vt_i < 0: t0 = nT + vt_i
                        else:        t0 = vt_i - 1
                        t_idx.append(t0)

                # triangulación fan
                for t in range(1, len(v_idx) - 1):
                    tri = [v_idx[0], v_idx[t], v_idx[t+1]]
                    faces.append(tri)
                    face_mtl.append(current_mtl)

                    # uv centroid si hay vt para los 3
                    tri_t = [t_idx[0], t_idx[t], t_idx[t+1]]
                    if all(x is not None for x in tri_t):
                        uv0 = np.array(uvs[tri_t[0]], dtype=np.float32)
                        uv1 = np.array(uvs[tri_t[1]], dtype=np.float32)
                        uv2 = np.array(uvs[tri_t[2]], dtype=np.float32)
                        uv = (uv0 + uv1 + uv2) / 3.0
                        face_uv.append(uv.tolist())                      
                        
                    else:
                        face_uv.append([0.5, 0.5])  # fallback

    V = np.asarray(verts, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)

    if V.size == 0 or F.size == 0:
        raise ValueError(f"Empty mesh from OBJ: {obj_path}")
    
    if len(face_uv) == 0:
        #print("[DEBUG] UV stats: NO UVs found")
        pass
    else:
        uv = np.array(face_uv, dtype=np.float32)

        #print("  count:", len(uv))
        #print("  u min/max:", float(np.min(uv[:,0])), float(np.max(uv[:,0])))
        #print("  v min/max:", float(np.min(uv[:,1])), float(np.max(uv[:,1])))
        #print("  % in [0,1] u:", float(np.mean((uv[:,0] >= 0) & (uv[:,0] <= 1))) )
        #print("  % in [0,1] v:", float(np.mean((uv[:,1] >= 0) & (uv[:,1] <= 1))) )
        #print("  first10:", uv[:10])

    return V, F, face_mtl, np.asarray(face_uv, dtype=np.float32)


def sample_facecolors_from_textures(
    face_mtl,
    face_uv,
    mtl_maps,
    flip_v: bool = True,
    uv_mode: str = "wrap",          # "wrap" o "clamp"
    samples: int = 5,               # 1, 5, 9 (5 = cross, 9 = 3x3)
    jitter: float = 0.0025,         # pequeño jitter UV para promediar (en espacio UV)
    alpha_threshold: int = 10,      # alpha mínimo para considerarlo “visible”
    alpha_mode: str = "blend",      # "blend" o "skip"
    background_rgb=(255, 255, 255), # para blend (alas suelen verse mejor con blanco)
    default_rgb=(160, 160, 160),
    debug: bool = False,
):
    """
    Devuelve lista len M con 'rgb(r,g,b)' por cara.
    - face_uv: centroid UV por cara (0..1 idealmente)
    - uv_mode:
        - "wrap": usa módulo (repite textura) -> común en OBJ
        - "clamp": recorta a 0..1
    - samples:
        - 1: solo centroid
        - 5: centro + cruz
        - 9: 3x3
    - alpha_mode:
        - "blend": mezcla con background según alpha (recomendado para alas)
        - "skip": ignora píxeles transparentes; si todos transparentes => default

    IMPORTANT:
    - Si el material NO tiene map_Kd (textura difusa), intenta usar Kd (color difuso del .mtl)
      si está disponible en mtl_maps[mtl_name]["Kd"] como (r,g,b) en rango 0..1 o 0..255.
    """
    tex_cache = {}
    out = []

    default = f"rgb({default_rgb[0]},{default_rgb[1]},{default_rgb[2]})"

    # offsets para multisampling en UV (no en píxeles)
    if samples <= 1:
        offsets = [(0.0, 0.0)]
    elif samples <= 5:
        offsets = [
            (0.0, 0.0),
            (+jitter, 0.0),
            (-jitter, 0.0),
            (0.0, +jitter),
            (0.0, -jitter),
        ]
    else:
        # 3x3
        d = jitter
        offsets = [(dx, dy) for dy in (-d, 0.0, d) for dx in (-d, 0.0, d)]

    # stats debug
    n_total = 0
    n_tex = 0
    n_fallback_no_tex = 0
    n_fallback_kd = 0
    n_fallback_alpha = 0

    def _norm_uv(val: float):
        if uv_mode == "clamp":
            return float(np.clip(val, 0.0, 1.0))
        # wrap
        return float(val % 1.0)

    def _kd_to_rgb_str(kd):
        """
        Acepta Kd como:
          - (r,g,b) en 0..1
          - (r,g,b) en 0..255
        Devuelve "rgb(r,g,b)" o None si no es válido.
        """
        if kd is None:
            return None
        try:
            r, g, b = kd
            r = float(r); g = float(g); b = float(b)
        except Exception:
            return None

        # Heurística: si todo <= 1.0 asumimos 0..1, si no 0..255
        if max(r, g, b) <= 1.0:
            r *= 255.0; g *= 255.0; b *= 255.0

        rr = int(np.clip(r, 0, 255))
        gg = int(np.clip(g, 0, 255))
        bb = int(np.clip(b, 0, 255))

        if rr == 0 and gg == 0 and bb == 0:
            return None
        
        return f"rgb({rr},{gg},{bb})"

    for mtl_name, (u0, v0) in zip(face_mtl, face_uv):
        n_total += 1

        tex_path = None
        kd = None

        if mtl_name and mtl_name in mtl_maps:
            tex_path = mtl_maps[mtl_name].get("map_Kd")
            kd = mtl_maps[mtl_name].get("Kd")  # opcional (color difuso)

        # 1) Si no hay textura difusa, intentamos Kd
        if not tex_path:
            rgb_kd = _kd_to_rgb_str(kd)
            if rgb_kd is not None:
                n_fallback_kd += 1
                out.append(rgb_kd)
            else:
                n_fallback_no_tex += 1
                out.append(default)
            continue

        # 2) Cargamos textura
        tex = load_texture_cached(tex_path, tex_cache)
        if tex is None:
            # si falla textura, también intentamos Kd como fallback
            rgb_kd = _kd_to_rgb_str(kd)
            if rgb_kd is not None:
                n_fallback_kd += 1
                out.append(rgb_kd)
            else:
                n_fallback_no_tex += 1
                out.append(default)
            continue

        n_tex += 1
        arr, W, H = tex

        # acumuladores
        acc = np.zeros(3, dtype=np.float32)
        wsum = 0.0
        any_visible = False

        for du, dv in offsets:
            u = _norm_uv(u0 + du)

            # Tu lógica (con detalle): flip en V antes del wrap/clamp final
            # - Si flip_v: v_un = 1 - (v0 + dv)
            # - Si no:     v_un = (v0 + dv)
            v_un = (1.0 - (v0 + dv)) if flip_v else (v0 + dv)
            v = _norm_uv(v_un)

            x = int(u * (W - 1))
            y = int(v * (H - 1))

            r, g, b, a = arr[y, x]
            a_f = float(a) / 255.0

            if alpha_mode == "skip":
                if a <= alpha_threshold:
                    continue
                any_visible = True
                acc += np.array([r, g, b], dtype=np.float32)
                wsum += 1.0
            else:
                # blend con background
                # (aunque a sea bajo, mezclamos igual; luego usamos any_visible para evitar parches raros)
                any_visible = any_visible or (a > alpha_threshold)
                bg = np.array(background_rgb, dtype=np.float32)
                fg = np.array([r, g, b], dtype=np.float32)
                mixed = fg * a_f + bg * (1.0 - a_f)
                acc += mixed
                wsum += 1.0

        # 3) Resolución por alpha
        if wsum <= 0.0:
            # No pudimos muestrear nada útil (skip total o textura rara)
            # probamos Kd y si no default
            rgb_kd = _kd_to_rgb_str(kd)
            if rgb_kd is not None:
                n_fallback_kd += 1
                out.append(rgb_kd)
            else:
                n_fallback_alpha += 1
                out.append(default)
            continue

        mean = acc / wsum

        if alpha_mode != "skip":
            # si TODO era prácticamente transparente, mejor Kd/default (evita “parches” raros)
            if not any_visible:
                rgb_kd = _kd_to_rgb_str(kd)
                if rgb_kd is not None:
                    n_fallback_kd += 1
                    out.append(rgb_kd)
                else:
                    n_fallback_alpha += 1
                    out.append(default)
                continue

        rr, gg, bb = [int(np.clip(x, 0, 255)) for x in mean]
        out.append(f"rgb({rr},{gg},{bb})")

    if debug:
        print("[DEBUG] facecolor sampling stats:")
        print("  faces:", n_total)
        print("  with texture:", n_tex)
        print("  fallback (no tex):", n_fallback_no_tex)
        print("  fallback (Kd):", n_fallback_kd)
        print("  fallback (alpha/transparent):", n_fallback_alpha)

    return out








def debug_materials(obj_path: str):
    mtl_path = find_mtl_for_obj(obj_path)
    obj_dir = os.path.dirname(obj_path)

    # materiales declarados en mtl
    mtl_mats = set()
    if mtl_path and os.path.exists(mtl_path):
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("newmtl "):
                    mtl_mats.add(line.split(None, 1)[1].strip())

    # materiales usados en obj
    used = set()
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("usemtl "):
                used.add(line.split(None, 1)[1].strip())

    #print("\n[DEBUG] OBJ:", obj_path)
    #print("[DEBUG] MTL:", mtl_path)
    #print("[DEBUG] usemtl in OBJ:", sorted(used))
    #print("[DEBUG] newmtl in MTL:", sorted(mtl_mats))
    missing = used - mtl_mats
    if missing:
        print("[DEBUG] WARNING: these usemtl are missing in MTL:", sorted(missing))



# ----------------------------
# Find .obj files
# ----------------------------
def find_all_obj_files(root_dir: str):
    out = []
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".obj"):
                out.append(os.path.join(root, fn))
    return out

def pick_best_obj_for_family(obj_paths, query_tokens):
    """
    Choose the first OBJ whose filename/path contains all query tokens (case-insensitive).
    If none match all, try any token match (fallback).
    """
    q = [t.lower() for t in query_tokens if t]
    if not q:
        return None

    def score(p):
        s = p.lower()
        return sum(1 for t in q if t in s)

    # perfect match: all tokens
    perfect = [p for p in obj_paths if all(t in p.lower() for t in q)]
    if perfect:
        # pick the shortest path among perfect matches (often the “main” one)
        return sorted(perfect, key=len)[0]

    # fallback: highest score
    scored = sorted(obj_paths, key=lambda p: (-score(p), len(p)))
    return scored[0] if scored and score(scored[0]) > 0 else None

def build_family_to_obj_map(root_dir: str):
    obj_paths = find_all_obj_files(root_dir)
    family_map = {}
    for family, tokens in FAMILY_QUERIES.items():
        match = pick_best_obj_for_family(obj_paths, tokens)
        if match:
            family_map[family] = match
    return family_map

# ----------------------------
# Plotly figure builder
# ----------------------------
def make_mesh_figure(V, F, title="", camera=None, face_colors=None):
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    i, j, k = F[:, 0], F[:, 1], F[:, 2]

    mesh_kwargs = dict(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=1.0,
        flatshading=True,
    )

    if face_colors is not None and len(face_colors) == F.shape[0]:
        mesh_kwargs["facecolor"] = face_colors  # <- la clave del color

    fig = go.Figure(data=[go.Mesh3d(**mesh_kwargs)])

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            
        ),
        showlegend=False
    )

    if camera is not None:
        fig.update_layout(scene_camera=camera)

    fig.update_layout(clickmode="event+select")


    return fig


# ----------------------------
# Dash app
# ----------------------------
app = Dash(__name__)
family_map = build_family_to_obj_map(ROOT_DIR)

available_families = list(family_map.keys())
default_family = available_families[0] if available_families else None

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "20px auto", "fontFamily": "Arial"},
    children=[
        html.H3("Insect Morphology Explorer (OBJ → Plotly Mesh3d)"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Family / Model:"),
                        dcc.Dropdown(
                            id="family-dd",
                            options=[{"label": f, "value": f} for f in available_families],
                            value=default_family,
                            clearable=False,
                            style={"width": "420px"},
                        ),

                        # Store camera state between interactions
                        dcc.Store(id="camera-store", data=None),

                        
                    ],
                    style={"display": "inline-block", "verticalAlign": "top"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),

        dcc.Store(id="mesh-cache", data=None),
        dcc.Graph(id="mesh-graph", style={"height": "750px"}),
        
        
        

        
    ],
)


@app.callback(
    Output("mesh-graph", "figure"),
    Output("mesh-cache", "data"),
    Output("camera-store", "data"),
    Input("family-dd", "value"),
    Input("mesh-graph", "relayoutData"),
    State("mesh-cache", "data"),
    State("camera-store", "data"),
)
def update_graph(family, relayoutData, cache, camera_store):
    if not family:
        return go.Figure(), cache, camera_store

    path = family_map.get(family)
    if not path or not os.path.exists(path):
        return go.Figure(), cache, camera_store


    # ------------------------------------------------------------
    # Trigger detection: evita que relayoutData viejo arruine el preset
    # ------------------------------------------------------------

    props = [t["prop_id"] for t in (callback_context.triggered or [])]
    triggered_by_family = any(p.startswith("family-dd.value") for p in props)
    triggered_by_relayout = any(p.startswith("mesh-graph.relayoutData") for p in props)

    # ------------------------------------------------------------
    # Cámara desde relayoutData (solo si el trigger fue relayout)
    # ------------------------------------------------------------
    cam = None

    if triggered_by_relayout and isinstance(relayoutData, dict):
        cam = relayoutData.get("scene.camera")

        if cam is None:
            # Merge parcial si viene en keys tipo "scene.camera.eye"
            eye = relayoutData.get("scene.camera.eye")
            up = relayoutData.get("scene.camera.up")
            center = relayoutData.get("scene.camera.center")
            proj = relayoutData.get("scene.camera.projection")

            if any(v is not None for v in (eye, up, center, proj)):
                cam = {}
                if eye is not None: cam["eye"] = eye
                if up is not None: cam["up"] = up
                if center is not None: cam["center"] = center
                if proj is not None: cam["projection"] = proj

        # merge con lo último guardado para no perder campos (shallow)
        if cam is not None and isinstance(camera_store, dict):
            merged = dict(camera_store)
            merged.update(cam)
            cam = merged

        if cam is not None and "projection" not in cam:
            cam["projection"] = {"type": "perspective"}

    # ------------------------------------------------------------
    # Persistencia / presets según trigger
    # ------------------------------------------------------------
    if triggered_by_relayout and cam is not None:
        # El usuario movió la cámara: guardamos
        camera_store = cam

    if triggered_by_family:
        # Cambio de modelo: ignoramos relayoutData viejo y aplicamos preset
        if family in CAMERA_PRESETS:
            camera_store = CAMERA_PRESETS[family]
        else:
            camera_store = None  # fuerza default abajo



    # Si todavía no hay cámara guardada, usamos una default razonable
    if camera_store is None:
        camera_store = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.6, y=1.6, z=1.0),
            projection=dict(type="perspective"),
        )


    # Si solo moviste cámara y el cache es de la misma familia -> NO recargo nada
    if cache and cache.get("family") == family:
        Vn = np.array(cache["Vn"], dtype=np.float32)
        F  = np.array(cache["F"], dtype=np.int32)
        face_colors = cache["face_colors"]

        fig = make_mesh_figure(Vn, F, title=family, camera=camera_store, face_colors=face_colors)
        return fig, cache, camera_store

    # Si no hay cache (o cambió familia), recargo todo
    debug_materials(path)

    V, F, face_mtl, face_uv = load_obj_mesh_uv_facecolors(path)
    Vn = normalize_vertices(V)

    mtl_path = find_mtl_for_obj(path)
    #print("[DEBUG] mtl_path:", mtl_path)

    mtl_maps = parse_mtl_maps(path, mtl_path)

    face_colors = sample_facecolors_from_textures(
        face_mtl, face_uv, mtl_maps,
        flip_v=True,
        uv_mode="wrap",
        samples=9,
        jitter=0.0025,
        alpha_mode="blend",
        background_rgb=(255,255,255),
        default_rgb=(160,160,160),
        debug=False
    )

    fig = make_mesh_figure(Vn, F, title=family, camera=camera_store, face_colors=face_colors)

    #status = f"Loaded: {V.shape[0]} vertices, {F.shape[0]} triangles | MTL: {mtl_path or 'None'}"
    #path_label = f"OBJ: {path}"

    new_cache = {
        "family": family,
        "Vn": Vn.tolist(),
        "F": F.tolist(),
        "face_colors": face_colors,
        
    }

    return fig, new_cache, camera_store



if __name__ == "__main__":
    if not os.path.isdir(ROOT_DIR):
        print(f"[ERROR] ROOT_DIR not found: {ROOT_DIR}")
        print('Set INSECT_MODELS_DIR to your folder path (or place models in "./3d_models").')
    else:
        print("Detected OBJ models:")
        #for fam, p in family_map.items():
            #print(f" - {fam}: {p}")
        app.run(debug=False)

