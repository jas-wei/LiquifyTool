"""Convert SVG geometry into cubic Bezier segments.

Python translation of readSVG_cubics.m, using NumPy and svgpathtools.
P, C, I, F, S, W, D, viewBox = read_svg_cubics('art.svg')

Differences from the provided MATLAB source:
- zero-based indices in C and I
- standalone replacements for missing parse_path/ellipse_to_spline/transform helpers
- XML namespace-aware parsing; SVG transform support; optional inherited SVG styles
- unsupported elements can raise, warn, or be ignored
"""
from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from svgpathtools import parse_path, CubicBezier, QuadraticBezier, Line, Arc

NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
NUMBER_RE = re.compile(NUMBER)
TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
KAPPA = 4 * (math.sqrt(2) - 1) / 3
NAMED_COLORS = {
    'red': (1,0,0), 'green': (0,1,0), 'blue': (0,0,1),
    'black': (0,0,0), 'white': (1,1,1), 'yellow': (1,1,0),
    'cyan': (0,1,1), 'magenta': (1,0,1), 'gray': (.5,.5,.5),
    'grey': (.5,.5,.5), 'aqua': (0,1,1), 'fuchsia': (1,0,1),
    'lime': (0,1,0), 'maroon': (.5,0,0), 'navy': (0,0,.5),
    'olive': (.5,.5,0), 'purple': (.5,0,.5),
    'silver': (.75,.75,.75), 'teal': (0,.5,.5),
}


def _numbers(value):
    return [float(x) for x in NUMBER_RE.findall(value or '')]


def _tag(elem):
    return elem.tag.rsplit('}', 1)[-1].split(':')[-1] if isinstance(elem.tag, str) else ''


def _style(elem, key, default=None, inherited=None):
    attrs = dict(pair.split(':', 1) for part in (elem.get('style') or '').split(';')
                 if ':' in part for pair in [part.strip()])
    # Inline style overrides presentation attributes.
    return attrs.get(key, elem.get(key, (inherited or {}).get(key, default)))


def _color(value, default):
    if value is None:
        return default
    v = value.strip().lower()
    if v == 'none':
        return (math.nan,) * 3
    if v in NAMED_COLORS:
        return NAMED_COLORS[v]
    if re.fullmatch(r'#[0-9a-f]{3}', v):
        return tuple(int(c * 2, 16) / 255 for c in v[1:])
    if re.fullmatch(r'#[0-9a-f]{6}', v):
        return tuple(int(v[i:i+2], 16) / 255 for i in (1, 3, 5))
    m = re.fullmatch(r'rgba?\s*\(([^)]+)\)', v)
    if m:
        parts = [x.strip() for x in m.group(1).split(',')]
        if len(parts) >= 3:
            return tuple(float(x[:-1])/100 if x.endswith('%') else float(x)/255 for x in parts[:3])
    warnings.warn(f'Unsupported color {value!r}; using NaNs')
    return (math.nan,) * 3


def _matrix(transform):
    """3x3 matrices acting on homogeneous column points."""
    result = np.eye(3)
    for name, raw in TRANSFORM_RE.findall(transform or ''):
        v = _numbers(raw)
        m = np.eye(3)
        if name == 'matrix' and len(v) == 6:
            a,b,c,d,e,f = v
            m = np.array([[a,c,e],[b,d,f],[0,0,1]], float)
        elif name == 'translate' and len(v) in (1,2):
            m[0,2] = v[0]; m[1,2] = v[1] if len(v)>1 else 0
        elif name == 'scale' and len(v) in (1,2):
            m[0,0] = v[0]; m[1,1] = v[1] if len(v)>1 else v[0]
        elif name == 'rotate' and len(v) in (1,3):
            a = math.radians(v[0]); c,s = math.cos(a), math.sin(a)
            r = np.array([[c,-s,0],[s,c,0],[0,0,1]],float)
            if len(v)==3:
                x,y = v[1:]
                before=np.eye(3); before[:2,2]=[x,y]
                after=np.eye(3); after[:2,2]=[-x,-y]
                m=before@r@after
            else: m=r
        elif name == 'skewX' and len(v)==1:
            m[0,1] = math.tan(math.radians(v[0]))
        elif name == 'skewY' and len(v)==1:
            m[1,0] = math.tan(math.radians(v[0]))
        else:
            warnings.warn(f'Unsupported SVG transform: {name}({raw})')
            continue
        result = result @ m
    return result


def _transform(points, matrix):
    if not points:
        return points
    arr = np.asarray(points, float)
    return (np.c_[arr, np.ones(len(arr))] @ matrix.T)[:,:2].tolist()


def _line_cubic(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return [a.tolist(), (a+(b-a)/3).tolist(), (a+2*(b-a)/3).tolist(), b.tolist()]


def _complex_pt(z):
    return [float(z.real), float(z.imag)]


def _path_cubics(d):
    cubics=[]
    for segment in parse_path(d):
        if isinstance(segment, CubicBezier):
            pts=[segment.start,segment.control1,segment.control2,segment.end]
            cubics.append([_complex_pt(q) for q in pts])
        elif isinstance(segment, QuadraticBezier):
            p0,p1,p2=segment.start,segment.control,segment.end
            cubics.append([_complex_pt(p0),_complex_pt(p0+2*(p1-p0)/3),
                           _complex_pt(p2+2*(p1-p2)/3),_complex_pt(p2)])
        elif isinstance(segment, Line):
            cubics.append(_line_cubic(_complex_pt(segment.start),_complex_pt(segment.end)))
        elif isinstance(segment, Arc):
            for cubic in segment.as_cubic_curves():
                cubics.append([_complex_pt(q) for q in
                              (cubic.start,cubic.control1,cubic.control2,cubic.end)])
        else:
            raise ValueError(f'Unsupported path segment: {type(segment).__name__}')
    return cubics


def _ellipse_cubics(cx,cy,rx,ry):
    if rx <= 0 or ry <= 0:
        return []
    # Four cubic quarters, counterclockwise in mathematical coordinates.
    base=[[(1,0),(1,KAPPA),(KAPPA,1),(0,1)],
          [(0,1),(-KAPPA,1),(-1,KAPPA),(-1,0)],
          [(-1,0),(-1,-KAPPA),(-KAPPA,-1),(0,-1)],
          [(0,-1),(KAPPA,-1),(1,-KAPPA),(1,0)]]
    return [[[cx+rx*x,cy+ry*y] for x,y in segment] for segment in base]


def _geom(elem):
    name = _tag(elem)
    def attr(key, default=0):
        nums=_numbers(elem.get(key))
        return nums[0] if nums else default
    if name=='path':
        return _path_cubics(elem.get('d',''))
    if name=='circle':
        return _ellipse_cubics(attr('cx'),attr('cy'),attr('r'),attr('r'))
    if name=='ellipse':
        return _ellipse_cubics(attr('cx'),attr('cy'),attr('rx'),attr('ry'))
    if name=='line':
        return [_line_cubic((attr('x1'),attr('y1')),(attr('x2'),attr('y2')))]
    if name in ('polygon','polyline'):
        n=_numbers(elem.get('points'))
        if len(n)%2: raise ValueError('Odd number of values in points')
        pts=list(zip(n[::2],n[1::2]))
        if name=='polygon' and pts and pts[-1]!=pts[0]: pts.append(pts[0])
        return [_line_cubic(a,b) for a,b in zip(pts,pts[1:])]
    if name=='rect':
        x,y,w,h=attr('x'),attr('y'),attr('width'),attr('height')
        if w<=0 or h<=0:return []
        if attr('rx') or attr('ry'):
            warnings.warn('Rounded rect corners not supported; using sharp corners')
        pts=[(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)]
        return [_line_cubic(a,b) for a,b in zip(pts,pts[1:])]
    return []


def read_svg_cubics(filename, unsupported_alert='error'):
    """Return (P,C,I,F,S,W,D,viewBox) as NumPy arrays.

    P: (N,2) float control-point positions
    C: (M,4) int indices into P (0-based)
    I: (M,) int object indices (0-based)
    F,S: (K,3) float RGB colors in 0..1, NaN means none/unknown
    W: (K,) float stroke widths, NaN means unspecified
    D: (K,) int visibility flags
    viewBox: (4,) float SVG viewBox, or an empty array
    """
    if unsupported_alert not in ('error','warning','ignore'):
        raise ValueError("unsupported_alert must be 'error', 'warning', or 'ignore'")
    root=ET.parse(Path(filename)).getroot()
    if _tag(root)!='svg':
        raise ValueError('Expected SVG root element')
    points=[]; curves=[]; owners=[]; fills=[]; strokes=[]; widths=[]; displays=[]
    ignored={'desc','title','metadata','style','script','linearGradient','radialGradient',
             'pattern','filter','text','flowRoot','font','marker','symbol','mask','use',
             'image','foreignObject'}
    containers={'svg','g','defs','clipPath'}
    geometry={'line','polyline','polygon','rect','circle','ellipse','path'}
    inherited_keys={'fill','stroke','stroke-width','display'}

    def alert(message):
        if unsupported_alert=='error':raise ValueError(message)
        if unsupported_alert=='warning':warnings.warn(message)

    def visit(element, parent_matrix, parent_style):
        name=_tag(element)
        local_style={key:_style(element,key,inherited=parent_style) for key in inherited_keys}
        matrix=parent_matrix@_matrix(element.get('transform'))
        if name in containers:
            for child in element:
                visit(child,matrix,local_style)
            return
        if name in ignored:
            if name in {'use','font','marker','symbol','mask','image','foreignObject'}:
                alert(f'Unsupported element <{name}>')
            return
        if name not in geometry:
            alert(f'Unknown SVG element <{name}>')
            return
        segments=_geom(element)
        if not segments:return
        object_id=len(fills)
        color_fill=_color(local_style['fill'],(0,0,0))
        color_stroke=_color(local_style['stroke'],(math.nan,)*3)
        width_num=_numbers(local_style['stroke-width'])
        width=width_num[0] if width_num else (1.0 if not all(math.isnan(v) for v in color_stroke) else math.nan)
        visible=0 if local_style['display']=='none' else 1
        fills.append(color_fill);strokes.append(color_stroke)
        widths.append(width);displays.append(visible)
        for segment in segments:
            offset=len(points)
            points.extend(_transform(segment,matrix))
            curves.append([offset,offset+1,offset+2,offset+3]);owners.append(object_id)

    visit(root,np.eye(3),{})
    def array(items,dtype,columns=None):
        a=np.asarray(items,dtype=dtype)
        return a.reshape((-1,columns)) if columns else a
    return (array(points,float,2),array(curves,int,4),array(owners,int),
            array(fills,float,3),array(strokes,float,3),array(widths,float),
            array(displays,int),np.asarray(_numbers(root.get('viewBox')),float))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Convert SVG geometry into cubic Bezier arrays')
    parser.add_argument('svg_file')
    parser.add_argument('--unsupported-alert', choices=['error','warning','ignore'],default='warning')
    parser.add_argument('--plot',action='store_true',help='Plot curves using matplotlib')
    args=parser.parse_args()
    P,C,I,F,S,W,D,viewBox=read_svg_cubics(args.svg_file,args.unsupported_alert)
    print(f'{len(F)} SVG objects, {len(C)} cubic segments, {len(P)} control points')
    print('viewBox:',viewBox,'\nP:',P,'\nC:',C)
    if args.plot:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        t=np.linspace(0,1,70)[:,None]
        for ids in C:
            p=P[ids]
            q=(1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t**2*p[2]+t**3*p[3]
            ax.plot(q[:,0],q[:,1])
        ax.set_aspect('equal');ax.invert_yaxis();plt.show()
