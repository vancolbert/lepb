#!/usr/bin/env python
import gzip, re, hashlib, numpy as np
def slurp(p):
    with open(p, 'rb') as f: d = f.read()
    if d.startswith(b'\x1f\x8b'): d = gzip.decompress(d)
    return memoryview(d)
def npt(s, byteorder='<'):
    a, e = [], byteorder or ''
    for l in s.split(';'):
        if f := l.strip():
            w, t = f.split(' ', 1)
            a.extend((n, e+t) for n in w.split(','))
    return np.dtype(a)
class Bits:
    __slots__ = ()
    def __init__(b, v):
        f, l = lambda i,n: v & 1<<i > 0, b.__slots__
        if isinstance(v, str):
            assert (s := set(v.split())).issubset(l), f'{v!r} not all in {l}'
            f = lambda i,n: n in s
        for i,n in enumerate(l): setattr(b, n, f(i, n))
    def __int__(b): return sum(getattr(b, n)<<i for i,n in enumerate(b.__slots__))
    def fmt(b): return '|'.join(n for n in reversed(b.__slots__) if getattr(b, n))
    def __str__(b):
        v, f, c = int(b), b.fmt(), b.__class__.__name__
        return f'{c}<{v} 0x{v:x} 0b{v:b}{" "*bool(f)}{f}>'
    def __repr__(b): return str(b)
class Vopts(Bits): __slots__ = 'nor tan uv2 col'.split()
class Vfmt(Bits): __slots__ = 'hpos huv huv2 cnor hind'.split()
def mkvert(vo, vf): return np.dtype(list(filter(None, (('uv', f'2f{4 - 2*vf.huv}'), ('uv2', f'2f{4 - 2*vf.huv2}')*vo.uv2, ('nor', n := 'u2' if vf.cnor else '3f4')*vo.nor, ('tan', n)*vo.tan, ('pos', f'3f{4 - 2*vf.hpos}'), ('col', '4B')*vo.col))))
class Rec:
    def __init__(r, a): r.__dict__ = {k:a[0][k].tolist() for k in a.dtype.fields if k[0] != '_'}
    __str__ = __repr__ = lambda r: 'Rec('+' '.join(f'{k}={v}' for k,v in r.__dict__.items())+')'
def unp(e, d, o=None):
    if o is not None: e.o = o
    a = np.ndarray(1, d, e.b, e.o)
    e.o += a.itemsize
    return Rec(a) if a.dtype.fields else a[0].tolist()
class Ahdr:
    __slots__ = 'size count offset'.split()
    sco, cso = (0,1,2), (1,0,2)
    def __init__(a, t, order=sco): a.size, a.count, a.offset = (t[i] for i in order)
    __str__ = __repr__ = lambda a: 'Ahdr('+' '.join(f'{k}={getattr(a,k)}' for k in a.__slots__)+')'
def unpa(e, d, h=None):
    if h is None: h = unp(e, '3i4')
    if not isinstance(h, Ahdr): h = Ahdr(h)
    if h.count: assert h.size == d.itemsize, f'struct size mismatch: header:{h.size} != dtype:{d.itemsize}'
    return (np.recarray if d.fields else np.ndarray)(h.count, d, e.b, h.offset)
def decode_16bit_unitvectors(a):
    from numpy import empty, where, reciprocal, sqrt, einsum
    x, y, n = (a & 8064) >> 7, a & 127, len(a)
    c, r = x + y >= 127, empty((n, 3), 'f4')
    x = where(c, 127 - x, x).astype('f4')
    y = where(c, 127 - y, y).astype('f4')
    z = 126 - x - y
    r[:,0] = where(a & 32768, -x, x)
    r[:,1] = where(a & 16384, -y, y)
    r[:,2] = where(a & 8192, -z, z)
    r *= reciprocal(sqrt(einsum('...i,...i', r, r))).reshape(n, 1)
    return r
class E3d:
    __slots__ = 'path b o vo vf verts inds nors parts'.split()
    fid, ver = b'e3dx', 257
    dt_fhdr = npt('fid V4; ver i4; md5 V16; od i4')
    dt_dhdr = npt('cso_verts,cso_inds,cso_parts 3i4; vo,vf u1; _0 V2')
    dt_part = npt('seethru u4; tex S128; bbox (2,3)f4; start,end,at,count u4')
    def __init__(e, path=None, buf=None, offset=0, geom=None):
        e.path, e.b, e.o = path, buf, offset
        if path and not e.b and not geom: e.b = slurp(path)
        if e.b: parse_e3d(e)
        if geom: assemble_e3d(e, geom)
    def encode(e): return encode_e3d(e)
    def dump(e): dump_e3d(e)
    __repr__ = __str__ = lambda e: f'E3d({e.path})'
def parse_e3d(e):
    f = unp(e, e.dt_fhdr)
    assert f.fid == e.fid, f'invalid file id {f.fid} (expecting {e.fid})'
    assert f.ver == e.ver, f'invalid version {f.ver} (expecting {e.ver})'
    assert f.md5 == hashlib.md5(e.b[f.od:]).digest(), 'md5 mismatch'
    d, A = unp(e, e.dt_dhdr, f.od), lambda t: Ahdr(t, Ahdr.cso)
    vo, vf = Vopts(d.vo), Vfmt(d.vf)
    e.verts = unpa(e, mkvert(vo, vf), A(d.cso_verts))
    e.inds = unpa(e, np.dtype(f'<u{4 - 2*vf.hind}'), A(d.cso_inds))
    e.parts = unpa(e, e.dt_part, A(d.cso_parts))
    e.nors, e.vo, e.vf = e.verts.nor if vo.nor else [], vo, vf
    if len(e.nors) and vf.cnor: e.nors = decode_16bit_unitvectors(e.nors)
class Geometry:
    __slots__ = 'pos uv nor mtlidx mtls'.split()
    def __init__(g):
        for k in g.__slots__: setattr(g, k, None)
def mkaos(g):
    l = [(g.uv, 2)]
    if g.nor is not None: l.append((g.nor, 3))
    l.append((g.pos, 3))
    return np.hstack([a.reshape(-1, n) for a,n in l])
def assemble_e3d(e, g):
    vo, vf = Vopts(0 if g.nor is None else 'nor'), Vfmt(0)
    dv, va = mkvert(vo, vf), mkaos(g)
    if len(g.mtls) > 1:
        tris = va.reshape(va.shape[0]//3, 3, -1)
        si = np.argsort(g.mtlidx)
        g.mtlidx = g.mtlidx[si]
        va = tris[si].reshape(va.shape)
    verts, inds = np.unique(va, return_inverse=1, axis=0)
    e.verts = np.recarray(nv := len(verts), dv, verts)
    vf.hind = nv < 65536
    e.inds = inds.astype(f'<u{4 - 2*vf.hind}')
    e.nors, e.vo, e.vf = e.verts.nor if vo.nor else [], vo, vf
    e.parts = np.recarray(len(g.mtls), e.dt_part)
    mis, at = np.repeat(g.mtlidx, 3), 0
    for i, m in enumerate(g.mtls):
        p, pi = e.parts[i], e.inds[mis == i]
        p.tex, p.seethru = m['imgfile'].encode('ascii'), m['seethru']
        p.bbox = np.min(pv := e.verts.pos[pi], axis=0), np.max(pv, axis=0)
        p.start, p.end, p.at, p.count = np.min(pi), np.max(pi), at, pi.size
        at += p.count
def align_up(v, a): return (v + (m := a - 1)) & ~m
def pack_array(b, o, a, ahdr_order=Ahdr.sco):
    b[o:o + a.nbytes] = a.reshape(-1).view('B')
    t = a.itemsize, len(a), o
    return [t[i] for i in ahdr_order]
def encode_e3d(e):
    ov = align_up((od := e.dt_fhdr.itemsize) + e.dt_dhdr.itemsize, 16)
    oi = ov + align_up(e.verts.nbytes, 16)
    op = oi + align_up(e.inds.nbytes, 16)
    e.b = (b := np.zeros(op + e.parts.nbytes, 'B').data)
    d, p = np.recarray(1, e.dt_dhdr, b, od), lambda a,o: pack_array(b, o, a, Ahdr.cso)
    d[0] = p(e.verts, ov), p(e.inds, oi), p(e.parts, op), e.vo, e.vf, b''
    f = np.recarray(1, e.dt_fhdr, b, 0)
    f[0] = e.fid, e.ver, hashlib.md5(b[od:]).digest(), od
    return b
def dump_e3d(e):
    print(f'{e.path} v:{len(e.verts)} i:{len(e.inds)} n:{len(e.nors)} p:{len(e.parts)}')
    print(f' {e.vo} {e.vf}')
    m = max(len(p.tex) for p in e.parts)
    for i,p in enumerate(e.parts):
        print(f' p{i} {p.tex.decode():{m}s} seethru={p.seethru} {p.start}..{p.end} @{p.at} [{p.count}]')
class Ordinals: __slots__ = 'meshes decals lights particles'.split()
class Chunk:
    __slots__ = 'pos terrain height meshes decals lights particles cluster ordinals'.split()
    def __init__(c, pos): c.pos, c.ordinals = pos, Ordinals()
    __repr__ = __str__ = lambda c: f'Chunk(pos={c.pos})'
class Elm:
    __slots__ = 'path b o terrain height meshes decals lights inside ambient particles cluster chunks idxc'.split()
    fid, ver, tsize, hsize, hstep, hmin, decal_dz = b'elmf', 0, 3, 0.5, 0.2, -2.2, np.arange(ddz := 1/32768, 0.01, ddz, 'f4')
    dt_hdr = npt('fid V4; x,y,to,ho i4; sco_meshes,sco_decals,sco_lights 3i4; inside u1; _0 V3; ambient 3f4; sco_particles 3i4; co,ver i4')
    dt_mesh = npt('apath S80; pos,rot 3f4; emit b; blend b; _0 V2; color 3f4; scale f4; _1 V20')
    dt_decal = npt('apath S80; pos,rot 3f4; _0 V24')
    dt_light = npt('pos,color 3f4; attenuation f4; _0 V12')
    dt_particle = npt('apath S80; pos 3f4; _0 V12')
    def __init__(e, path=None, buf=None, offset=0, initxy=None):
        e.path, e.b, e.o = path or '', buf, offset
        if path and not e.b and not initxy: e.b = slurp(path)
        if e.b: parse_elm(e)
        if initxy: init_elm(e, *initxy)
    def chunkify(e, terrains_per_side): chunkify_elm(e, terrains_per_side)
    def encode(e): return encode_elm(e)
    def dump(e): dump_elm(e)
    __repr__ = __str__ = lambda e: f'Elm({e.path})'
def parse_elm(e):
    h = unp(e, e.dt_hdr)
    assert h.fid == e.fid, f'invalid file id {h.fid}'
    assert h.ver == e.ver, f'file version mismatch: {h.ver} (expecting {e.ver})'
    n, w2 = (x := h.x) * (y := h.y), (w := int(e.tsize / e.hsize))*w
    e.terrain = np.frombuffer(e.b, 'B', n, h.to).reshape(y, x)
    e.height = np.frombuffer(e.b, 'B', w2*n, h.ho).reshape(w*y, w*x)
    e.meshes = unpa(e, e.dt_mesh, h.sco_meshes)
    e.decals = unpa(e, e.dt_decal, h.sco_decals)
    e.lights = unpa(e, e.dt_light, h.sco_lights)
    e.inside, e.ambient = h.inside, h.ambient
    e.particles = unpa(e, e.dt_particle, h.sco_particles)
    e.cluster = np.frombuffer(e.b, '<H', w2*n, h.co).reshape(w*y, w*x) if h.co else []
def sblocks(a, n, s): return [[a[n*i:n*i+n,n*j:n*j+n] for j in range(s[1])] for i in range(s[0])]
def sdiv(s, n): return tuple((v + n - 1)//n for v in s)
def fill_chunk(e, d, i, j, csz, ci):
    e.chunks[i,j] = (c := Chunk((j*csz, i*csz, 0)))
    for k in 'terrain height cluster'.split(): setattr(c, k, getattr(d, k)[i][j])
    for k in 'meshes decals lights particles'.split():
        setattr(c, k, getattr(e, k)[w := getattr(d, k) == ci])
        setattr(c.ordinals, k, w.nonzero()[0].astype('i4'))
def chunkify_elm(e, tps):
    csz, hps = tps * e.tsize, int(tps * e.tsize / e.hsize)
    invcsz, s, hs = 1/csz, sdiv(e.terrain.shape, tps), sdiv(e.height.shape, hps)
    e.chunks, e.idxc, maxci = np.empty(s, 'O'), (d := Chunk(None)), np.prod(s) - 1
    d.terrain, d.height = sblocks(e.terrain, tps, s), sblocks(e.height, hps, hs)
    d.cluster = sblocks(e.cluster, hps, hs) if len(e.cluster) else np.empty(hs, 'O')
    for k in 'meshes decals lights particles'.split():
        a = np.trunc(getattr(e, k).pos[:,0:2] * invcsz, dtype='i4', casting='unsafe')
        setattr(d, k, np.clip(a[:,0] + s[1]*a[:,1], 0, maxci))
    for i,j in np.ndindex(s): fill_chunk(e, d, i, j, csz, j + s[1]*i)
def encode_elm(e):
    o, asz = {'terrain':align_up(e.dt_hdr.itemsize, 16)}, lambda a: align_up(a.nbytes, 16)
    l = 'terrain height meshes decals lights particles cluster'.split()
    for i,k in enumerate(l[:-1]): o[l[i+1]] = o[k] + asz(getattr(e, k))
    n = o['cluster'] + (asz(e.cluster) if (ec := getattr(e, 'cluster', None) is not None and len(e.cluster) > 0) else 0)
    e.b, p = (b := np.zeros(n, 'B').data), lambda k: pack_array(b, o[k], getattr(e, k), Ahdr.sco)
    p('terrain'), p('height'), p('cluster') if ec else 0
    np.recarray(1, e.dt_hdr, b, 0)[0] = e.fid, *e.terrain.shape[::-1], o['terrain'], o['height'], p('meshes'), p('decals'), p('lights'), e.inside, b'', e.ambient, p('particles'), o['cluster'] if ec else 0, e.ver
    return b
def init_elm(e, x, y):
    e.terrain, e.height, e.cluster, e.inside, e.ambient = np.zeros((y, x), 'B'), np.zeros(((w := int(e.tsize / e.hsize))*y, w*x), 'B'), [], 0, (0.,)*3
    for k in 'meshes decals lights particles'.split(): setattr(e, k, np.array([], getattr(e, next(n for n in dir(e) if n.startswith('dt_') and n[3] == k[0]))).view(np.recarray))
def fxy(e, s): return ' '.join(f'{n}:{a.shape[1]}x{a.shape[0]}' if len(a := getattr(e, n)) else f'{n}:0' for n in s.split())
def fls(e, s): return ' '.join(f'{n}:{len(getattr(e, n))}' for n in s.split())
def dump_elm(e):
    print(e.path, fxy(e, 'terrain height cluster'))
    print('', fls(e, 'meshes decals lights particles'))
    print(f' inside={e.inside} ambient={e.ambient}')
class Decal:
    __slots__ = 'path type alpha_cutoff objsz imgfile imgbox imgdiv'.split()
    def __init__(d, path=None):
        d.path = path
        if path: parse_decal(d, slurp(path).tobytes().decode())
    def encode(d): return encode_decal(d)
    def dump(d): dump_decal(d)
    __repr__ = __str__ = lambda d: f'Decal({d.path})'
re_2d0line = re.compile(r'\s*(\w+)\s*[:=]\s*(\S+)')
def tup(f, t, s): return tuple(t(float(f[k])) for k in s.split())
def parse_decal(d, s):
    f = {m[1]:m[2] for l in s.splitlines() if (m := re_2d0line.match(l))}
    d.imgfile, d.type = f['texture'], f['type']
    d.alpha_cutoff = float(f.get('alpha_test', 0))
    d.objsz = tup(f, float, 'x_size y_size')
    d.imgbox = tup(f, int, 'u_start v_start u_end v_end')
    d.imgdiv = tup(f, int, 'file_x_len file_y_len')
def encode_decal(d): return ''.join(f'{t[0]}: {t[1]}\r\n' for t in (('texture', d.imgfile), ('file_x_len', d.imgdiv[0]), ('file_y_len', d.imgdiv[1]), ('x_size', d.objsz[0]), ('y_size', d.objsz[1]), ('u_start', d.imgbox[0]), ('u_end', d.imgbox[2]), ('v_start', d.imgbox[1]), ('v_end', d.imgbox[3]), ('type', d.type), (d.alpha_cutoff > 0.0)*('alpha_test', d.alpha_cutoff)) if t).encode()
def dump_decal(d):
    print(f'{d.path} {d.imgfile} type={d.type} alpha_cutoff={d.alpha_cutoff}')
    print(f' objsz:{d.objsz} imgdiv:{d.imgdiv} imgbox:{d.imgbox}')
def main():
    from sys import argv
    d = {'e3d':E3d, 'elm':Elm, '2d0':Decal}
    for p in (a := argv[1:]): d[(p[:-3] if p.endswith('.gz') else p)[-3:]](p).dump()
    if not a: print('expecting asset file')
if __name__ == '__main__': main()
