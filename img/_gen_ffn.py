# -*- coding: utf-8 -*-
"""Genere img/ffn_comparison.svg avec des aretes parfaitement attachees
aux neurones (trim sur le bord des cercles) et un panneau Modern reorganise
en deux voies paralleles (Gate / Up) qui se rejoignent sur le produit ⊙."""
import math

def edge(sx, sy, sr, tx, ty, tr, color, marker, opacity, width, dash=None, gap=3.0):
    """Ligne entre deux noeuds, rognee sur le bord de chacun."""
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy) or 1.0
    ux, uy = dx / d, dy / d
    x1, y1 = sx + ux * sr, sy + uy * sr
    x2, y2 = tx - ux * (tr + gap), ty - uy * (tr + gap)
    da = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" opacity="{opacity}"{da} '
            f'marker-end="url(#{marker})"/>')

P = []  # morceaux du SVG

# ----------------------------------------------------------------- HEADER
P.append('''<svg viewBox="0 0 1100 760" xmlns="http://www.w3.org/2000/svg" width="1100" height="760" preserveAspectRatio="xMidYMid meet">
  <title>FFN Architecture Comparison - Classic ReLU vs Modern SwiGLU</title>
  <defs>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&amp;display=swap');
      text { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
    </style>
    <filter id="shadow-sm" x="-5%" y="-5%" width="115%" height="125%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#00000018"/>
    </filter>
    <filter id="shadow-md" x="-8%" y="-8%" width="120%" height="130%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#00000020"/>
    </filter>
    <marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#64748B"/>
    </marker>
    <marker id="arrow-red" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#F43F5E"/>
    </marker>
    <marker id="arrow-teal" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#06B6D4"/>
    </marker>
    <marker id="arrow-violet" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#8B5CF6"/>
    </marker>
    <marker id="arrow-amber" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
      <polygon points="0 0, 9 3.5, 0 7" fill="#F59E0B"/>
    </marker>
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#E2E8F0" stroke-width="0.5"/>
    </pattern>
  </defs>

  <!-- Background -->
  <rect width="1100" height="760" fill="#FFFFFF"/>
  <rect width="1100" height="760" fill="url(#grid)" opacity="0.55"/>

  <!-- TITLE -->
  <text x="40" y="48" font-size="20" font-weight="700" fill="#111827">FFN Architecture - Classic vs Modern</text>
  <text x="40" y="70" font-size="12" font-weight="400" fill="#6B7280">Feed-Forward Network  -  2-layer ReLU (left)  vs  3-layer SwiGLU gated variant (right)</text>
  <line x1="40" y1="80" x2="430" y2="80" stroke="#6366F1" stroke-width="2" stroke-linecap="round"/>

  <!-- Vertical divider -->
  <line x1="550" y1="95" x2="550" y2="695" stroke="#E2E8F0" stroke-width="1.5" stroke-dasharray="6,4"/>''')

# ================================================================ LEFT PANEL
P.append('''
  <!-- LEFT PANEL - Classic 2-layer ReLU -->
  <rect x="30" y="100" width="500" height="570" rx="14" fill="#F43F5E0A" stroke="#F43F5E66" stroke-width="1.5" stroke-dasharray="6,4"/>
  <rect x="42" y="90" width="232" height="24" rx="12" fill="#F43F5E"/>
  <text x="158" y="106" font-size="11" font-weight="600" fill="#FFFFFF" text-anchor="middle">1 - Classic - 2-Layer FFN (ReLU)</text>
  <text x="118" y="158" font-size="10" font-weight="600" fill="#64748B" text-anchor="middle" letter-spacing="0.5">INPUT</text>
  <text x="280" y="158" font-size="10" font-weight="600" fill="#F43F5E" text-anchor="middle" letter-spacing="0.5">HIDDEN  x4</text>
  <text x="440" y="158" font-size="10" font-weight="600" fill="#64748B" text-anchor="middle" letter-spacing="0.5">OUTPUT</text>''')

# Noeuds gauche
L_in  = [245, 345, 445]; L_in_x, L_in_r = 118, 22
L_hid = [200, 290, 380, 470]; L_hid_x, L_hid_r = 280, 24
L_out = [270, 345, 420]; L_out_x, L_out_r = 440, 22

# Noeuds d'entree (gauche)
P.append('\n  <!-- INPUT NODES -->')
for k, cy in enumerate(L_in, 1):
    P.append(f'  <circle cx="{L_in_x}" cy="{cy}" r="{L_in_r}" fill="#FFFFFF" stroke="#94A3B8" stroke-width="1.5" filter="url(#shadow-sm)"/>')
    P.append(f'  <text x="{L_in_x}" y="{cy+5}" font-size="12" font-weight="600" fill="#475569" text-anchor="middle">x{chr(0x2080+k)}</text>')

# Aretes Input -> Hidden (3x4), opacites facon "poids"
op_ih = [[0.45, 0.34, 0.22, 0.18],
         [0.20, 0.50, 0.40, 0.28],
         [0.18, 0.28, 0.40, 0.50]]
P.append('\n  <!-- EDGES: Input -> Hidden -->')
for i, sy in enumerate(L_in):
    for j, ty in enumerate(L_hid):
        w = 1.2 if op_ih[i][j] >= 0.5 else 1.0
        P.append(edge(L_in_x, sy, L_in_r, L_hid_x, ty, L_hid_r,
                      "#F43F5E", "arrow-red", op_ih[i][j], w))

# W1 label
P.append('''
  <rect x="163" y="168" width="34" height="20" rx="10" fill="#FFF1F2" stroke="#F43F5E" stroke-width="1"/>
  <text x="180" y="182" font-size="10" font-weight="700" fill="#F43F5E" text-anchor="middle">W₁</text>''')

# Hidden nodes
P.append('\n  <!-- HIDDEN NODES (ReLU) -->')
for k, cy in enumerate(L_hid, 1):
    P.append(f'  <circle cx="{L_hid_x}" cy="{cy}" r="{L_hid_r}" fill="#FFF1F2" stroke="#F43F5E" stroke-width="1.5" filter="url(#shadow-sm)"/>')
    P.append(f'  <text x="{L_hid_x}" y="{cy-4}" font-size="10" font-weight="600" fill="#9F1239" text-anchor="middle">h₀{chr(0x2080+k)}</text>'.replace('h₀', 'h'))
    P.append(f'  <text x="{L_hid_x}" y="{cy+10}" font-size="8" fill="#F43F5E" text-anchor="middle">ReLU</text>')

# Aretes Hidden -> Output (4x3)
op_ho = [[0.40, 0.26, 0.20],
         [0.36, 0.50, 0.28],
         [0.22, 0.36, 0.50],
         [0.18, 0.30, 0.50]]
P.append('\n  <!-- EDGES: Hidden -> Output -->')
for i, sy in enumerate(L_hid):
    for j, ty in enumerate(L_out):
        w = 1.2 if op_ho[i][j] >= 0.5 else 1.0
        P.append(edge(L_hid_x, sy, L_hid_r, L_out_x, ty, L_out_r,
                      "#F43F5E", "arrow-red", op_ho[i][j], w))

# W2 label + output nodes + dims + formula
P.append('''
  <rect x="363" y="168" width="34" height="20" rx="10" fill="#FFF1F2" stroke="#F43F5E" stroke-width="1"/>
  <text x="380" y="182" font-size="10" font-weight="700" fill="#F43F5E" text-anchor="middle">W₂</text>''')
P.append('\n  <!-- OUTPUT NODES -->')
for k, cy in enumerate(L_out, 1):
    P.append(f'  <circle cx="{L_out_x}" cy="{cy}" r="{L_out_r}" fill="#FFFFFF" stroke="#94A3B8" stroke-width="1.5" filter="url(#shadow-sm)"/>')
    P.append(f'  <text x="{L_out_x}" y="{cy+5}" font-size="12" font-weight="600" fill="#475569" text-anchor="middle">y{chr(0x2080+k)}</text>')

P.append('''
  <text x="118" y="505" font-size="9" fill="#94A3B8" text-anchor="middle" font-style="italic">d_model</text>
  <text x="280" y="515" font-size="9" fill="#F43F5E" text-anchor="middle" font-style="italic">d_model x 4</text>
  <text x="440" y="505" font-size="9" fill="#94A3B8" text-anchor="middle" font-style="italic">d_model</text>

  <rect x="50" y="540" width="460" height="110" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#shadow-sm)"/>
  <rect x="50" y="540" width="4" height="110" rx="2" fill="#F43F5E"/>
  <text x="70" y="565" font-size="11" font-weight="600" fill="#1E293B">Formula</text>
  <text x="70" y="589" font-size="13" font-weight="700" fill="#F43F5E">FFN(x) = W₂ · ReLU(W₁x + b₁) + b₂</text>
  <text x="70" y="612" font-size="10" fill="#475569">Two linear projections with a single ReLU gate.</text>
  <text x="70" y="630" font-size="10" fill="#475569">No multiplicative interaction - activation is static.</text>
  <rect x="388" y="548" width="100" height="26" rx="13" fill="#FFF1F2" stroke="#F43F5E" stroke-width="1.5"/>
  <text x="438" y="565" font-size="11" font-weight="700" fill="#F43F5E" text-anchor="middle">2 layers</text>''')

# ================================================================ RIGHT PANEL
P.append('''
  <!-- RIGHT PANEL - Modern 3-layer SwiGLU -->
  <rect x="570" y="100" width="500" height="570" rx="14" fill="#06B6D40A" stroke="#06B6D466" stroke-width="1.5" stroke-dasharray="6,4"/>
  <rect x="582" y="90" width="240" height="24" rx="12" fill="#06B6D4"/>
  <text x="702" y="106" font-size="11" font-weight="600" fill="#FFFFFF" text-anchor="middle">2 - Modern - 3-Layer SwiGLU</text>
  <text x="648" y="158" font-size="10" font-weight="600" fill="#64748B" text-anchor="middle" letter-spacing="0.5">INPUT</text>''')

# Geometrie droite : deux voies paralleles
R_in_x, R_in_r = 648, 20; R_in = [248, 333, 418]
G_x, G_r = 792, 19; G_y = [205, 258, 311]      # voie GATE (haut)
U_x, U_r = 792, 19; U_y = [372, 425, 478]      # voie UP (bas)
HAD_x, HAD_y, HAD_r = 905, 335, 19             # produit element-wise
OUT_cx, OUT_y = 905, 452                        # boite de sortie

# Fonds de voie (groupage visuel des deux flux)
P.append('''
  <!-- Lane backgrounds (two parallel streams) -->
  <rect x="752" y="182" width="80" height="152" rx="14" fill="#06B6D40A" stroke="#06B6D433" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="792" y="176" font-size="9.5" font-weight="700" fill="#0E7490" text-anchor="middle">GATE - SiLU(W_gate·x)</text>
  <rect x="752" y="349" width="80" height="152" rx="14" fill="#8B5CF60A" stroke="#8B5CF633" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="792" y="343" font-size="9.5" font-weight="700" fill="#6D28D9" text-anchor="middle">UP-PROJ - W_up·x</text>''')

# Input nodes
P.append('\n  <!-- INPUT NODES -->')
for k, cy in enumerate(R_in, 1):
    P.append(f'  <circle cx="{R_in_x}" cy="{cy}" r="{R_in_r}" fill="#FFFFFF" stroke="#94A3B8" stroke-width="1.5" filter="url(#shadow-sm)"/>')
    P.append(f'  <text x="{R_in_x}" y="{cy+5}" font-size="12" font-weight="600" fill="#475569" text-anchor="middle">x{chr(0x2080+k)}</text>')

# Aretes Input -> Gate (teal) et Input -> Up (violet pointille)
op_ig = [[0.45, 0.40, 0.24],
         [0.24, 0.50, 0.40],
         [0.20, 0.34, 0.45]]
P.append('\n  <!-- EDGES: Input -> Gate -->')
for i, sy in enumerate(R_in):
    for j, ty in enumerate(G_y):
        w = 1.2 if op_ig[i][j] >= 0.5 else 1.0
        P.append(edge(R_in_x, sy, R_in_r, G_x, ty, G_r, "#06B6D4", "arrow-teal", op_ig[i][j], w))
P.append('\n  <!-- EDGES: Input -> Up-proj -->')
for i, sy in enumerate(R_in):
    for j, ty in enumerate(U_y):
        w = 1.2 if op_ig[i][j] >= 0.5 else 1.0
        P.append(edge(R_in_x, sy, R_in_r, U_x, ty, U_r, "#8B5CF6", "arrow-violet", op_ig[i][j]*0.85, w, dash="5,3"))

# Gate nodes
P.append('\n  <!-- GATE NODES -->')
for k, cy in enumerate(G_y, 1):
    P.append(f'  <circle cx="{G_x}" cy="{cy}" r="{G_r}" fill="#ECFEFF" stroke="#06B6D4" stroke-width="1.5" filter="url(#shadow-sm)"/>')
    P.append(f'  <text x="{G_x}" y="{cy-3}" font-size="10" font-weight="600" fill="#164E63" text-anchor="middle">g{chr(0x2080+k)}</text>')
    P.append(f'  <text x="{G_x}" y="{cy+10}" font-size="7.5" fill="#06B6D4" text-anchor="middle">SiLU</text>')
# Up nodes
P.append('\n  <!-- UP-PROJ NODES -->')
for k, cy in enumerate(U_y, 1):
    P.append(f'  <circle cx="{U_x}" cy="{cy}" r="{U_r}" fill="#F5F3FF" stroke="#8B5CF6" stroke-width="1.5" filter="url(#shadow-sm)"/>')
    P.append(f'  <text x="{U_x}" y="{cy+5}" font-size="10" font-weight="600" fill="#5B21B6" text-anchor="middle">u{chr(0x2080+k)}</text>')

# Gate -> ⊙ et Up -> ⊙
P.append('\n  <!-- Gate -> (x) and Up -> (x) -->')
for cy in G_y:
    P.append(edge(G_x, cy, G_r, HAD_x, HAD_y, HAD_r, "#06B6D4", "arrow-teal", 0.6, 1.5))
for cy in U_y:
    P.append(edge(U_x, cy, U_r, HAD_x, HAD_y, HAD_r, "#8B5CF6", "arrow-violet", 0.6, 1.5))

# ⊙ node + labels + down-proj
P.append(f'''
  <!-- Element-wise product -->
  <circle cx="{HAD_x}" cy="{HAD_y}" r="{HAD_r}" fill="#FFFBEB" stroke="#F59E0B" stroke-width="2" filter="url(#shadow-md)"/>
  <text x="{HAD_x}" y="{HAD_y+6}" font-size="17" font-weight="700" fill="#B45309" text-anchor="middle">⊙</text>
  <text x="{HAD_x+25}" y="{HAD_y-2}" font-size="9" fill="#B45309" text-anchor="start" font-style="italic">element-</text>
  <text x="{HAD_x+25}" y="{HAD_y+9}" font-size="9" fill="#B45309" text-anchor="start" font-style="italic">wise product</text>''')
# Down projection arrow ⊙ -> output
P.append(edge(HAD_x, HAD_y, HAD_r, OUT_cx, OUT_y+19, 19, "#F59E0B", "arrow-amber", 0.75, 2.2))
P.append(f'''
  <rect x="{OUT_cx-30}" y="392" width="60" height="20" rx="10" fill="#FFFBEB" stroke="#F59E0B" stroke-width="1"/>
  <text x="{OUT_cx}" y="406" font-size="10" font-weight="700" fill="#B45309" text-anchor="middle">W_down</text>
  <rect x="{OUT_cx-45}" y="{OUT_y}" width="90" height="38" rx="8" fill="#ECFDF5" stroke="#10B981" stroke-width="1.5" filter="url(#shadow-sm)"/>
  <text x="{OUT_cx}" y="{OUT_y+23}" font-size="11" font-weight="600" fill="#065F46" text-anchor="middle">Output</text>
  <text x="{OUT_cx}" y="{OUT_y+52}" font-size="9" fill="#94A3B8" text-anchor="middle" font-style="italic">d_model</text>''')

# Dim labels for hidden lanes
P.append(f'''
  <text x="648" y="453" font-size="9" fill="#94A3B8" text-anchor="middle" font-style="italic">d_model</text>
  <text x="792" y="513" font-size="9" fill="#8B5CF6" text-anchor="middle" font-style="italic">d_model x 4  (gate &amp; up)</text>''')

# Right formula card + badge
P.append('''
  <rect x="580" y="540" width="460" height="110" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#shadow-sm)"/>
  <rect x="580" y="540" width="4" height="110" rx="2" fill="#06B6D4"/>
  <text x="600" y="565" font-size="11" font-weight="600" fill="#1E293B">Formula</text>
  <text x="600" y="589" font-size="12" font-weight="700" fill="#06B6D4">FFN(x) = W_down · ( SiLU(W_gate·x) ⊙ W_up·x )</text>
  <text x="600" y="612" font-size="10" fill="#475569">Gate stream modulates the up-proj stream element-wise.</text>
  <text x="600" y="630" font-size="10" fill="#475569">Dynamic, learned gating - no static clipping.</text>
  <rect x="916" y="548" width="100" height="26" rx="13" fill="#ECFEFF" stroke="#06B6D4" stroke-width="1.5"/>
  <text x="966" y="565" font-size="11" font-weight="700" fill="#06B6D4" text-anchor="middle">3 layers</text>''')

# ================================================================ LEGEND
P.append('''
  <!-- LEGEND / COMPARISON BAR -->
  <g transform="translate(30, 690)">
    <rect width="1040" height="56" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1"/>
    <text x="16" y="22" font-size="10" font-weight="700" fill="#64748B" letter-spacing="0.5">COMPARISON</text>
    <rect x="16" y="30" width="12" height="12" rx="3" fill="#FFF1F2" stroke="#F43F5E"/>
    <text x="34" y="41" font-size="10" fill="#1E293B">Classic: <tspan font-weight="600">2 x d x 4d</tspan> params  (W₁, W₂)</text>
    <rect x="270" y="30" width="12" height="12" rx="3" fill="#ECFEFF" stroke="#06B6D4"/>
    <text x="288" y="41" font-size="10" fill="#1E293B">SwiGLU: <tspan font-weight="600">3 x d x 4d</tspan> params  (W_gate, W_up, W_down)  - hidden size often reduced to 2/3</text>
    <text x="700" y="22" font-size="10" fill="#64748B" font-weight="600">Activation:</text>
    <rect x="768" y="12" width="46" height="18" rx="9" fill="#FFF1F2" stroke="#F43F5E" stroke-width="1"/>
    <text x="791" y="25" font-size="9" font-weight="600" fill="#F43F5E" text-anchor="middle">ReLU</text>
    <text x="824" y="25" font-size="10" fill="#94A3B8">vs</text>
    <rect x="840" y="12" width="40" height="18" rx="9" fill="#ECFEFF" stroke="#06B6D4" stroke-width="1"/>
    <text x="860" y="25" font-size="9" font-weight="600" fill="#06B6D4" text-anchor="middle">SiLU</text>
    <text x="890" y="25" font-size="10" fill="#64748B">  used in: Llama, Mistral, Gemma...</text>
  </g>
</svg>''')

svg = "\n".join(P)
with open("img/ffn_comparison.svg", "w", encoding="utf-8") as f:
    f.write(svg)

import xml.etree.ElementTree as ET
ET.parse("img/ffn_comparison.svg")
print("SVG genere et XML valide -", len(svg), "octets")
