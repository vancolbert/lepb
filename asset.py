#!/usr/bin/env python
import gzip, hashlib, numpy as np, os, re, struct, sys, time
npd, sdt, exists, pj, now, rx = np.dtype, struct.Struct, os.path.exists, os.path.join, time.time, re.compile
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
    return npd(a)
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
def mkvert(vo, vf): return npd(list(filter(None, (('uv', f'2f{4 - 2*vf.huv}'), ('uv2', f'2f{4 - 2*vf.huv2}')*vo.uv2, ('nor', n := 'u2' if vf.cnor else '3f4')*vo.nor, ('tan', n)*vo.tan, ('pos', f'3f{4 - 2*vf.hpos}'), ('col', '4B')*vo.col))))
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
    if h.count: assert h.size == d.itemsize, f'Struct size mismatch: header:{h.size} != dtype:{d.itemsize}'
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
    fid, ver, ext = b'e3dx', 257, 'e3d'
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
def unp_hdr(e, dt):
    h = unp(e, dt)
    assert h.fid == (c := e.__class__).fid, f'Invalid file id {h.fid} (expecting {c.fid})'
    if hasattr(c, 'vers'): assert h.ver in c.vers, f'Invalid version {h.ver} (expecting one of {c.vers})'
    elif hasattr(c, 'ver'): assert h.ver == c.ver, f'Invalid version {h.ver} (expecting {c.ver})'
    return h
def parse_e3d(e):
    f = unp_hdr(e, e.dt_fhdr)
    assert f.md5 == hashlib.md5(e.b[f.od:]).digest(), 'MD5 mismatch'
    d, A = unp(e, e.dt_dhdr, f.od), lambda t: Ahdr(t, Ahdr.cso)
    vo, vf = Vopts(d.vo), Vfmt(d.vf)
    e.verts = unpa(e, mkvert(vo, vf), A(d.cso_verts))
    e.inds = unpa(e, npd(f'<u{4 - 2*vf.hind}'), A(d.cso_inds))
    e.parts = unpa(e, e.dt_part, A(d.cso_parts))
    e.nors, e.vo, e.vf = e.verts.nor if vo.nor else [], vo, vf
    if len(e.nors) and vf.cnor: e.nors = decode_16bit_unitvectors(e.nors)
class Geometry:
    __slots__ = 'pos uv nor mtlidx mtls'.split()
    def __init__(g): any(setattr(g, k, None) for k in g.__slots__)
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
    fid, ver, ext, tsize, hsize, hstep, hmin, decal_dz = b'elmf', 0, 'elm', 3, 0.5, 0.2, -2.2, np.arange(ddz := 1/32768, 0.01, ddz, 'f4')
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
    h = unp_hdr(e, e.dt_hdr)
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
        a = np.trunc(getattr(e, k).pos[:,0:2] * invcsz).astype('i4')
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
    __slots__, ext = 'path type alpha_cutoff objsz imgfile imgbox imgdiv'.split(), '2d0'
    def __init__(d, path=None):
        d.path = path
        if path: parse_decal(d, slurp(path).tobytes().decode())
    def encode(d): return encode_decal(d)
    def dump(d): dump_decal(d)
    __repr__ = __str__ = lambda d: f'Decal({d.path})'
re_2d0line = rx(r'\s*(\w+)\s*[:=]\s*(\S+)')
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
st_i4, st_f4, st_vec3, st_vec2, st_quat = [sdt('<'+t) for t in 'i f 3f 2f 4f'.split()]
def unp_st(c, s):
    t = s.unpack_from(c.b, c.o)
    c.o += s.size
    return t[0] if len(t) == 1 else t
def unps(c, *a): return [unp_st(c, s) for s in a]
def unp_lstr(c):
    n = unp_st(c, st_i4)
    s = c.b[c.o:c.o + n].tobytes().rstrip(b'\x00').decode()
    c.o += n
    return s
def unp_list(c, dt, n, as_array=0):
    if not n: return []
    a = np.frombuffer(c.b, dt, n, c.o)
    c.o += a.nbytes
    return a if as_array else a.tolist()
def unp_recs(c, dt, n):
    r = np.recarray(n, dt, c.b, c.o)
    c.o += r.nbytes
    return r
def pack_st(c, s, *a):
    s.pack_into(c.b, c.o, *a)
    c.o += s.size
def pack_nparr(c, a):
    c.b[c.o:c.o + a.nbytes] = a.reshape(-1).view('B')
    c.o += a.nbytes
class Cmesh:
    __slots__ = 'path ver b o submeshes'.split()
    fid, vers, ext = b'CMF\x00', (700, 1000, 1200), 'cmf'
    dt_hdr = npt('fid V4; ver i4; nsubmeshes i4')
    def __init__(c, path=None, buf=None, offset=0):
        c.path, c.ver, c.b, c.o, c.submeshes = path, 0, buf, offset, []
        if path and not c.b: c.b = slurp(path)
        if c.b: parse_cmesh(c)
    __str__ = __repr__ = lambda c: f'Cmesh({c.path} {c.ver} submeshes:{len(c.submeshes)})'
    def encode(c): return encode_cmesh(c)
    def dump(c): print(c)
def parse_cmesh(c):
    h = unp_hdr(c, c.dt_hdr)
    c.ver, c.submeshes = h.ver, [parse_submesh(c) for i in range(h.nsubmeshes)]
    return c
class Submesh:
    __slots__ = 'material verts springs faces nuv nlod weights'.split()
    dt_vert = npt('pos,nor 3f4; colv,ncolf i4; uv 2f4; wstart,wend i4; spweight f4')
    dt_face = npd('<3u4')
    dt_spring = npt('vert1,vert2 i4; coef,restlen f4')
    dt_weight = npt('vert,bone i4; weight f4')
    __str__ = __repr__ = lambda m: f'Submesh(material={m.material} verts:{len(m.verts)} faces:{len(m.faces)} weights:{len(m.weights)} springs:{len(m.springs)} nuv={m.nuv} nlod={m.nlod})'
def parse_submesh(c):
    (m := Submesh()).material, nv, nf, m.nlod, ns, nu = unps(c, *(st_i4,)*6)
    assert nu in (0,1), f'Unhandled number of UV coordinates: {nu}'
    m.verts, w, m.nuv = (v := np.recarray(nv, m.dt_vert)), np.recarray(12*nv, m.dt_weight), nu
    j, sl = 0, [st_vec3, st_vec3, st_i4, st_i4, *(st_vec2,)*nu, st_i4]
    for i in range(nv):
        nw = (t := unps(c, *sl))[-1]
        w[j:j + nw] = [(i, *unps(c, st_i4, st_f4)) for k in range(nw)]
        v[i] = *t[:4], t[4] if nu else (0,0), j, (j := j + nw), unp_st(c, st_f4) if ns else 0
    m.springs, m.faces, m.weights = unp_recs(c, m.dt_spring, ns), unp_list(c, m.dt_face, nf, as_array=1), w[:j].copy()
    return m
def encode_cmesh(c):
    c.b, c.o = (b := np.zeros((h := c.dt_hdr.itemsize) + sum(24 + len(s.verts)*(32 + s.nuv*8 + 96) + s.springs.nbytes + s.faces.nbytes for s in c.submeshes), 'B').data), h
    np.recarray(1, c.dt_hdr, b, 0)[0] = c.fid, 700, len(c.submeshes)
    for s in c.submeshes: pack_submesh(c, s)
    c.b = b[:c.o]
    return c.b
st_submesh, st_vertpre, st_influ = sdt('<IIIIII'), sdt('<3f3fII'), sdt('<If')
def pack_submesh(c, s):
    pack_st(c, st_submesh, s.material, len(s.verts), len(s.faces), s.nlod, len(s.springs), s.nuv)
    assert s.nuv in (0,1), f'Unhandled number of UV coordinate pairs: {s.nuv} (expecting 0 or 1)'
    for v in s.verts:
        pack_st(c, st_vertpre, *v.pos, *v.nor, v.colv, v.ncolf)
        if s.nuv: pack_st(c, st_vec2, *v.uv)
        pack_st(c, st_i4, v.wend - v.wstart)
        for vi,bi,w in s.weights[v.wstart:v.wend]: pack_st(c, st_influ, bi, w)
        if len(s.springs): pack_st(c, st_f4, v.spweight)
    pack_nparr(c, s.springs)
    pack_nparr(c, s.faces)
class Cskel:
    __slots__ = 'path ver b o bones'.split()
    fid, vers, ext = b'CSF\x00', (700, 1200), 'csf'
    dt_hdr = npt('fid V4; ver i4; nbones i4');
    def __init__(c, path=None, buf=None, offset=0):
        c.path, c.ver, c.b, c.o, c.bones = path, 0, buf, offset, []
        if path and not c.b: c.b = slurp(path)
        if c.b: parse_cskel(c)
    __str__ = __repr__ = lambda c: f'Cskel({c.path} {c.ver} bones:{len(c.bones)})'
    def encode(c): return encode_cskel(c)
    def dump(c): print(c)
def parse_cskel(c):
    h = unp_hdr(c, c.dt_hdr)
    c.ver, c.bones = h.ver, [parse_bone(c) for i in range(h.nbones)]
    return c
class Bone:
    __slots__ = 'name trans rot ltrans lrot parent children'.split()
    def __init__(b, n): b.name = n
    __str__ = __repr__ = lambda b: f'Bone({b.name})'
def parse_bone(c):
    b = Bone(unp_lstr(c))
    b.trans, b.rot, b.ltrans, b.lrot, b.parent, n = unps(c, st_vec3, st_quat, st_vec3, st_quat, st_i4, st_i4)
    b.children = [unp_st(c, st_i4) for i in range(n)]
    return b
def encode_cskel(c):
    c.b, c.o = (b := np.zeros((h := c.dt_hdr.itemsize) + sum(324 + len(t.children)*4 for t in c.bones), 'B').data), h
    np.recarray(1, c.dt_hdr, b, 0)[0] = c.fid, 700, len(c.bones)
    for t in c.bones: pack_bone(c, t)
    c.b = b[:c.o]
    return c.b
def pack_lstr(c, s):
    pack_st(c, st_i4, n := align_up(len(d := s.encode()), 4))
    c.b[c.o:c.o + len(d)] = d
    c.o += n
def pack_bone(c, b):
    pack_lstr(c, b.name)
    pack_st(c, st_vec3, *b.trans)
    pack_st(c, st_quat, *b.rot)
    pack_st(c, st_vec3, *b.ltrans)
    pack_st(c, st_quat, *b.lrot)
    pack_st(c, st_i4, b.parent)
    pack_st(c, st_i4, len(b.children))
    for i in b.children: pack_st(c, st_i4, i)
class Canim:
    __slots__ = 'path ver b o duration_s tracks flags'.split()
    fid, vers, ext = b'CAF\x00', (700, 1200), 'caf'
    dt_hdr = npt('fid V4; ver i4; duration_s f4; ntracks i4')
    def __init__(c, path=None, buf=None, offset=0):
        c.path, c.ver, c.b, c.o, c.duration_s, c.tracks, c.flags = path, 0, buf, offset, 0, [], 0
        if path and not c.b: c.b = slurp(path)
        if c.b: parse_canim(c)
    __str__ = __repr__ = lambda c: f'Canim({c.path} {c.ver} duration:{c.duration_s:0.3f}s tracks:{len(c.tracks)})'
    def encode(c): return encode_canim(c)
    def dump(c): print(c)
def parse_canim(c):
    h = unp_hdr(c, c.dt_hdr)
    c.ver, c.duration_s = h.ver, h.duration_s
    if c.ver == 1200: c.flags = unp_st(c, st_i4)
    pt = parse_compressed_track if c.flags else parse_track
    c.tracks = [pt(c) for i in range(h.ntracks)]
    return c
class Track:
    __slots__ = 'bone keyframes'.split()
    dt_keyframe = npt('time f4; trans 3f4; rot 4f4')
    dt_compkf = npt('time u2; trans u4; rot 3u2')
    __str__ = __repr__ = lambda t: f'Track(bone={t.bone} keyframes:{len(t.keyframes)})'
def parse_track(c):
    (t := Track()).bone, n = unps(c, st_i4, st_i4)
    t.keyframes = unp_recs(c, t.dt_keyframe, n)
    return t
def parse_compressed_track(c):
    (t := Track()).bone, n, min_trans, scale = unps(c, st_i4, st_i4, st_vec3, st_vec3)
    t.keyframes = decode_compressed_keyframes(unp_recs(c, t.dt_compkf, n), c.duration_s, min_trans, scale)
swiz = np.array(((3,0,1,2), (0,3,1,2), (0,1,3,2), (0,1,2,3)), 'u4')
def decode_compressed_keyframes(a, duration_s, min_trans, scale):
    r = np.recarray(len(a), Track.dt_keyframe)
    r.time, p, q = np.float32(duration_s * 1.5259022e-05) * a.time, a.trans, a.rot
    r.trans = min_trans + scale * np.stack((p & 0x7ff, (p >> 11) & 0x7ff, p >> 22), axis=1)
    h, s = (((q1 := q[:,1]) & 1) << 1) | ((q2 := q[:,2]) & 1), np.float32(2.15799e-05)
    x, y, z = s*q[:,0], s*(q1 & 0xfffe), s*(q2 & 0xfffe)
    np.sqrt(w := 1 - x*x - y*y - z*z, where=w > 1e-5, out=w)
    r.rot = np.stack((x, y, z, w), axis=1).ravel()[np.arange(0, 4*len(h), 4, 'u4').repeat(4) + swiz[h].ravel()].reshape(-1,4)
    return r
def encode_canim(c):
    c.b, c.o = (b := np.zeros((h := c.dt_hdr.itemsize) + sum(8 + len(t.keyframes)*32 for t in c.tracks), 'B').data), h
    np.recarray(1, c.dt_hdr, b, 0)[0] = c.fid, 700, c.duration_s, len(c.tracks)
    for t in c.tracks: pack_track(c, t)
    return b
def pack_track(c, t):
    pack_st(c, st_i4, t.bone)
    pack_st(c, st_i4, len(t.keyframes))
    pack_nparr(c, t.keyframes)
class Cmat:
    __slots__ = 'path ver b o ambient diffuse specular shininess textures'.split()
    fid, vers, ext = b'CRF\x00', (700, 1200), 'crf'
    dt_hdr = npt('fid V4; ver i4; amb,diff,spec 4u1; shine f4; ntex i4')
    def __init__(c, path=None, buf=None, offset=0):
        c.path, c.ver, c.b, c.o = path, 0, buf, offset
        if path and not c.b: c.b = slurp(path)
        if c.b: parse_cmat(c)
    __str__ = __repr__ = lambda c: f'Cmat({c.path} {c.ver} a={c.ambient} d={c.diffuse} s={c.specular} sh={c.shininess} textures:{len(c.textures)})'
    def dump(c): print(c)
def parse_cmat(c):
    h = unp_hdr(c, c.dt_hdr)
    c.ambient, c.diffuse, c.specular, c.shininess, n = unps(c, *(st_col4b,)*3, st_f4, st_i4)
    c.ver, c.textures = h.ver, [unp_lstr(c) for i in range(n)]
    return c
def chomp(s, x): return s[:-len(x)] if s.endswith(x) else s
def find_asset(ad, n): return next((p for d in ad if exists(p := pj(d, n))), n)
def mtime(p): return os.stat(p).st_mtime
re_xmlref = rx(r'<!ENTITY\s+(\w+)\s+SYSTEM\s+"([^">/]+)"')
re_xmltok = rx('|'.join(f'(?P<{w[:3]}>{w[4:]})' for w in r'cmt <!--.*?--> | tag <\w+[^>]*> | end </\w+> | ref &\w+; | wsp [\t ]+ | nwl \n | str [^<&\n]+'.split(' | ')), re.S)
re_xmlattr = rx(r'\s*(\w+)="([^"]*)"')
re_actorid = rx(b'<actor [^>]*id="(\d+)"')
class Tree:
    __slots__ = 'ref name path atime root deps'.split()
    def __init__(t, r, n): t.ref, t.name, t.path, t.atime, t.root, t.deps = r, n, None, 0, None, {r}
    __str__ = __repr__ = lambda t: f'Tree({t.ref} {t.name} path={t.path} atime={t.atime} root={t.root} deps:{len(t.deps)})'
    def fresh(t): return t.root and t.path and mtime(t.path) < t.atime
class Rect:
    __slots__ = 'x y w h'.split()
    def __init__(r, x, y, w, h): r.x, r.y, r.w, r.h = x, y, w, h
    __str__ = __repr__ = lambda r: f'Rect({r.x},{r.y} {r.w}x{r.h})'
    __getitem__ = lambda r,i: getattr(r, 'xywh'[i])
    xy = pos = property(lambda r: (r.x, r.y))
    wh = size = property(lambda r: (r.w, r.h))
    def to_uv(r): return Rect((u := 1/128)*r.x, u*r.y, u*r.w, u*r.h)
class Skin:
    __slots__ = 'tag apath priority'.split()
    rects = {a[0]:Rect(*(int(v) for v in a[1:])) for a in (w.split(',') for w in 'body,0,0,128,128 boots,0,88,39,40 cape,66,0,62,38 hair,0,0,34,48 helmet,40,74,39,14 hands,34,32,16,16 head,34,0,32,32 neck,40,48,10,26 legs,39,88,40,40 shield,50,38,39,36 arms,0,48,40,40 torso,79,74,49,54 weapon,89,38,39,36'.split())}
    def __init__(s, t, a): s.tag, s.apath, s.priority = t, a, 0
    __str__ = __repr__ = lambda s: f'Skin({s.tag} {s.apath})'
xsdeps = {'head':('hair','head'),'legs':('boots','legs'),'shirt':('arms','hands','torso','weapon')}
class Piece:
    __slots__ = 'kind id desc mesh skins glow fnam'.split()
    kinds = 'body boots cape hair head helmet hskin neck legs shield shirt weapon'.split()
    initvals = {k:int(i) for k,i in zip(kinds, '0 0 30 0 1 20 0 0 0 11 0 0'.split())}
    def __init__(p, k, i, d, m, s, g): p.kind, p.id, p.desc, p.mesh, p.skins, p.glow, p.fnam = k, i, d, m, s, g, None
    __str__ = __repr__ = lambda p: f'Piece({p.kind} {p.id} {p.desc!r} mesh={p.mesh} skins={p.skins} glow={p.glow})'
    def fmt(p): return n if (n := p.fnam) else (setattr(p, 'fnam', n := f'{p.id} {p.desc.lower()}'), n)[1]
    def get_skindeps(p): return () if not p.mesh else xsdeps.get(p.kind, (p.kind,))
hskin_ids = {'brown':0, 'normal':1, 'pale':2, 'tan':3, 'darkblue':4, 'clair':5, 'fonce':6, 'gris':7, 'medium':8, 'masque_pnj':9, 'mannequin':10, 'vert':1, 'vert clair':5, 'vert foncé':6}
def extract_text(n, t, d=None): return c.text if (c := n.find_tag(t)) and c.text else d
def extract_pieces(cmodel, a):
    cmodel.pieces = (r := {'body':{0:Piece('body', 0, 'base', (e := extract_text)(a, 'mesh'), [Skin('body', e(a, 'skin'))], None)}})
    if not (dm := a.find_tag('defaults')): return r
    for k in Piece.kinds:
        t, d = r.setdefault(k, {}), {c.attrs.get('group'):c.mesh.text for c in dm.find_all(k)}
        for n in a.find_all(k):
            p = Piece(k, int(i) if (i := (g := n.attrs.get)('id', g('number'))) is not None else None, g('type', g('color')), e(n, 'mesh', d.get(g('group'), '')), [Skin(n.tag, s)] if (s := e(n, 'skin')) else [], v if (v := e(n, 'glow')) and v != 'none' else None)
            if p.id is None and k == 'hskin': p.id = hskin_ids[p.desc]
            assert p.id is not None, f'Missing id in {n}'
            if not p.skins: p.skins = [Skin(w, e(n, w)) for w in ('arms','torso')] if n.find_tag('arms') else [Skin(w, e(n, w)) for w in ('hands','head')] if n.find_tag('hands') else [Skin(n.tag, n.text)] if n.text else []
            if k == 'weapon' and not p.mesh and p.skins: (s := p.skins[0]).tag, s.priority = 'hands', 1
            assert all(s.tag in Skin.rects for s in p.skins), f'Unexpected skin tag {p.skins} not in {Skin.rects} in {n}'
            if not p.desc: p.desc = f'{k}{p.id}'
            t[p.id] = p
        if not t.get(i := Piece.initvals[k]): t[i] = Piece(k, i, 'none', None, [], None)
    return r
class Cmodel:
    __slots__ = 'id name tree pieces'.split()
    def __init__(c, i, n, t): c.id, c.name, c.tree, c.pieces = i, n, t, None
    __str__ = __repr__ = lambda c: f'Cmodel({c.id} {c.name} {c.tree})'
class Mdlset:
    __slots__ = 'assetdirs path trees cmodels'.split()
    def __init__(m, assetdirs=None):
        m.assetdirs, m.path, m.trees, m.cmodels = (ad := ['.'] if (d := assetdirs) is None else [d] if isinstance(d, str) else d), None, {}, {}
        with open(p := find_asset(ad := m.assetdirs, pj('actor_defs', 'actor_defs.xml'))) as f: s, m.path = f.read(), p
        m.trees = {j[1]:Tree(j[1], chomp(j[2], '.xml')) for j in re_xmlref.finditer(s)}
        for t in m.trees.values():
            if exists(p := find_asset(ad, pj('actor_defs', t.name + '.xml'))) and (j := re_actorid.search(open(p, 'rb').readline())): m.cmodels[i], t.path = Cmodel(i := int(j[1]), t.name, t), p
    __str__ = __repr__ = lambda m: f'Mdlset({m.assetdirs} path={m.path} trees:{len(m.trees)} cmodels:{len(m.cmodels)})'
    def get_xmlroot(m, cmodel): return load_xmlref(m, cmodel.tree.ref)
    def setup_pieces(m, cmodel): return extract_pieces(cmodel, m.get_xmlroot(cmodel).actor)
class NodeNotFoundError(RuntimeError): pass
def throw(e, *a): raise e(*a)
class Node:
    __slots__ = 'tag attrs text children'.split()
    def __init__(n, t, a=None): n.tag, n.attrs, n.text, n.children = t, a or {}, '', []
    def find_tag(n, t, d=None): return next((c for c in n.children if c.tag == t), d)
    def __getattr__(n, t): return n.find_tag(t) or throw(NodeNotFoundError, t)
    def find_all(n, t): return [c for c in n.children if c.tag == t]
    __str__ = __repr__ = lambda n: f'Node({n.tag} {n.attrs} {n.text!r} children:{len(n.children)})'
    def dump(n, i=0):
        print(f'{" "*i}{n.tag}', ''.join(f' {k}={v!r}' for k,v in n.attrs.items()), f' {n.text!r}' if n.text else '', sep='')
        for c in n.children: c.dump(i + 1)
def load_xmlref(mdlset, ref, depth=0):
    p = find_asset(mdlset.assetdirs, pj('actor_defs', (t := mdlset.trees[ref]).name + '.xml'))
    if t.root and t.path == p and all(mdlset.trees[r].fresh() for r in t.deps): return t.root
    with open(p) as f: s, ln, c, ls, t.root, t.path, t.atime, wh = f.read(), 1, 0, 0, (o := [Node('root')])[0], p, now(), lambda: f'in {p} line {ln} column {c+1}'
    for m in re_xmltok.finditer(s):
        k, v, c = m.lastgroup, m[0], m.start() - ls
        if k == 'tag':
            o[-1].children.append(n := Node((a := v[1:-1].split(None, 1))[0], {j[1]:j[2] for j in re_xmlattr.finditer(a[1])} if len(a) > 1 else None))
            o.append(n)
        elif k == 'end': assert (n := o.pop()).tag == (e := v[2:-1]), f'Mismatched tags, expecting {n.tag!r} not {e!r} {wh()}'
        elif k == 'ref':
            assert depth < 8, 'Maximum recursion depth exceeded {wh()}'
            assert (r := v[1:-1]) in mdlset.trees, f'Unknown xml entity reference {r!r} {wh()}'
            o[-1].children.extend(load_xmlref(mdlset, r, depth+1).children)
            t.deps.update(mdlset.trees[r].deps)
        elif k == 'str': o[-1].text = v
        elif k == 'nwl': ls, ln = m.end(), ln + 1
    return t.root
class Asset:
    __slots__ = 'path cls atime r'.split()
    def __init__(a, p, c): a.path, a.cls, a.atime, a.r = p, c, 0, None
    def load(a): return a.r if a.r is not None and mtime(a.path) < a.atime else (setattr(a, 'r', a.cls(a.path)), setattr(a, 'atime', now()), a.r)[-1]
    __str__ = __repr__ = lambda a: f'Asset({a.path} {a.cls.__name__} atime={a.atime} {a.r})'
class Cache:
    __slots__ = ['d']
    def __init__(c): c.d = {}
    def obtain(c, cls, p): return (c.d.get(p) or c.d.setdefault(p, Asset(p, cls))).load()
    __str__ = __repr__ = lambda c: f'Cache({set(a.cls.__name__ for a in c.d.values())} [{len(c.d)}])'
def main():
    d = {c.ext:c for c in (E3d, Elm, Decal, Cskel, Canim, Cmesh, Cmat)}
    for p in (a := sys.argv[1:]): d[chomp(p, '.gz')[-3:]](p).dump()
    if not a: print('expecting asset file')
if __name__ == '__main__': main()
