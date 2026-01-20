import bpy, bmesh, lepb, mathutils, numpy as np, os, pathlib
from os.path import abspath, basename, exists, join as pj, split as psplit, relpath
from mathutils import Matrix, Quaternion, Vector
Path, abspath_bl = pathlib.Path, bpy.path.abspath
class Graph:
    spacing = 50, 25
    def __init__(g, nodable, pre='ShaderNode', clear=True):
        if hasattr(nodable, 'use_nodes'): nodable.use_nodes = 1
        t = getattr(nodable, 'node_tree', 0) or nodable
        g.pre, g.n, g.l, g.dims = pre, t.nodes, t.links, lepb.yuck.nodedims[pre]
        if clear: g.n.clear()
    def node(g, what, where='right', of=-2, defvals={}, pre=None, **attrs):
        r = g.n.new((g.pre if pre is None else pre) + what)
        for k,v in attrs.items(): setattr(r, k, v)
        for k,v in defvals.items(): r.inputs[k].default_value = v
        if len(g.n) < 2: return r
        v, s, pd, rd = (p := g.n[of]).location.copy(), g.spacing, g.dims.get(p.type, (150,150)), g.dims[r.type]
        if where == 'right': v.x += pd[0] + s[0]
        elif where == 'left': v.x -= rd[0] + s[0]
        elif where == 'below': v.y -= pd[1] + s[1]
        elif where == 'above': v.y += rd[1] + s[1]
        else: v = where
        r.location = v
        return r
    def edge(g, src_out=0, dst_in=0, src=-2, dst=-1):
        s, d = (i if isinstance(i, bpy.types.Node) else g.n[i] for i in (src, dst))
        g.l.new(s.outputs[src_out], d.inputs[dst_in])
def get_assetdirs(c, p=None):
    opts, prefs, ad = get_opts(c), get_prefs(c), []
    if e := opts.extra_path or prefs.default_extra_path: ad.append(Path(e))
    if p is not None: ad.extend(d for d in (Path(p).absolute()).parents if (d/'files.lst').exists())
    elif d := opts.asset_path or prefs.default_asset_path: ad.append(Path(d))
    return ad
def snip(s, prefix): return s[len(prefix):] if s.startswith(prefix) else s
def chomp(s, suffix): return s[:-len(suffix)] if s.endswith(suffix) else s
def strip_ext(s): return s[:p] if (p := s.rfind('.')) > 0 else s
imgexts = '.dds .png .jpg .bmp'.split()
def find_asset(ad, relpath, filename=None, exts=None, must_exist=1):
    if not filename: relpath, filename = a if len(a := snip(relpath, './').rsplit('/', 1)) > 1 else ('.', relpath)
    p = next((p for d in ad for e in ([''] if exts is None else exts) if (p := d/relpath/(filename+e)).exists()), None)
    if must_exist: assert p, f'Asset not found in {ad}: relpath={relpath} filename={filename} exts={exts}'
    return str(p) if p else None
def opacity_controller(g):
    g.node('BsdfTransparent', 'below')
    d = g.node('MixShader', of=-3).inputs[0].driver_add('default_value').driver
    (v := d.variables.new()).type, v.name = 'SINGLE_PROP', 'opacity'
    (t := v.targets[0]).id_type, t.id, t.data_path = 'SCENE', bpy.context.scene, 'lepb_uidata.opts.opacity'
    d.expression = f'0.01*{v.name}'
    g.edge(0, 2, -3)
    g.edge(0, 1)
def get_dirnum(d):
    t = (u := bpy.context.scene.lepb_uidata).get(k := 'dirnums') or {}
    if not (r := t.get(d)): t[d], u[k] = (r := 1 + len(t)), t
    return r
def abbreviate_path(p): return (f'{get_dirnum(a[0])}/' if len(a := p.rsplit('/', 1)) > 1 else '') + a[-1]
def mtl_e3dpart(part, ad):
    imgfile = part.tex.decode()
    if m := bpy.data.materials.get(mtlname := f'mtl-e3d-{imgfile}'): return m
    g = Graph(m := bpy.data.materials.new(mtlname))
    if not (i := bpy.data.images.get(imgname := f'img-e3d-{imgfile}')): (i := bpy.data.images.load(find_asset(ad, '3dobjects', imgfile))).name = imgname
    g.node('TexImage', image=i)
    if part.seethru or i.channels == 4:
        g.node('BsdfTransparent')
        g.node('BsdfDiffuse', 'below')
        g.edge('Color', 'Color', -3)
        g.node('MixShader')
        g.edge('Alpha', 'Fac', -4)
        g.edge('BSDF', 1, -3)
        g.edge('BSDF', 2, -2)
    else:
        g.node('BsdfDiffuse')
        g.edge('Color', 'Color')
    opacity_controller(g)
    g.node('OutputMaterial')
    g.edge(0, 'Surface')
    return m
def flat(a, dtype=None): return np.ascontiguousarray(a, dtype).reshape(-1)
def flipv(a, dtype='f4', in_place=False):
    r = a if in_place else np.array(a, dtype)
    r[:,1] *= -1
    r[:,1] += 1
    return r
def new_bpy(name, data, bdc='objects', cprops=None, lprops=None, **attrs):
    r = getattr(bpy.data, bdc).new(name, data)
    for k,v in attrs.items(): setattr(r, k, v)
    if cprops: r.id_properties_ensure().update(cprops)
    if lprops: r.lepb_props.update(lprops)
    return r
def select_none(c): any(o.select_set(0) for o in c.view_layer.objects)
def select_active(c, o):
    o.select_set(1)
    c.view_layer.objects.active = o
def select_only(c, o):
    select_none(c)
    select_active(c, o)
def linkcl(c, o, collection=None): return (o, (collection or (c.layer_collection or c.view_layer.active_layer_collection).collection).objects.link(o))[0]
def relapath(ad, p): return next(relpath(p, d) for d in ad if p.is_relative_to(d))
def msh_e3d(ad, path, name):
    if m := bpy.data.meshes.get(name): return m
    if not path: return None
    e, m, opts = lepb.asset.E3d(path), bpy.data.meshes.new(name), get_opts()
    vs, vsi = np.unique(e.verts.pos, return_inverse=1, axis=0)
    m.vertices.add(len(vs))
    m.vertices.foreach_set('co', flat(vs, 'f4'))
    m.loops.add(ni := e.inds.size)
    m.loops.foreach_set('vertex_index', flat(vsi[e.inds], 'i4'))
    m.polygons.add(nf := ni // 3)
    m.polygons.foreach_set('loop_start', np.arange(0, ni, 3, 'i4'))
    m.polygons.foreach_set('loop_total', np.full(nf, 3, 'i4'))
    if e.vo.nor: m.attributes.new(k := 'temp_custom_split_normals', 'FLOAT_VECTOR', 'CORNER').data.foreach_set('vector', flat(e.nors[e.inds], 'f4'))
    mi = np.empty(nf, 'i4')
    for i,p in enumerate(e.parts):
        m.materials.append(mtl_e3dpart(p, ad))
        mi[(j := p.at//3):j + p.count//3] = i
    m.polygons.foreach_set('material_index', mi)
    m.uv_layers.new(do_init=0).data.foreach_set('uv', flat(flipv(e.verts.uv)[e.inds], 'f4'))
    if e.vo.uv2: m.uv_layers.new(do_init=0).data.foreach_set('uv', flat(flipv(e.verts.uv2)[e.inds], 'f4'))
    if e.vo.col: m.color_attributes.new('Color', 'BYTE_COLOR', 'CORNER').data.foreach_set('color', flat(e.verts.col[e.inds], 'B'))
    m.validate(clean_customdata=0)
    if e.vo.nor:
        (c := m.attributes[k]).data.foreach_get('vector', a := np.empty(3 * len(m.loops), 'f4'))
        m.attributes.remove(c)
        m.normals_split_custom_set(a.reshape(-1, 3))
    m.update(calc_edges=1)
    if (a := opts.sharp_angle) > -1: m.set_sharp_from_angle(angle=np.radians(a))
    m.shade_smooth() if opts.shade_smooth else m.shade_flat()
    return m
def import_e3d(context, filepath):
    ad = get_assetdirs(context, p := Path(abspath_bl(filepath)).absolute())
    m = msh_e3d(ad, str(p), p.name)
    return linkcl(context, new_bpy(m.name, m, rotation_mode='YXZ', lprops=dict(group='meshes', apath=relapath(ad, p))))
def mtlinfo(m, edir):
    assert m.use_nodes, f'Expecting node tree for {m.name}'
    img, seethru = None, 0
    for n in m.node_tree.nodes:
        if n.type == 'TEX_IMAGE': img = n.image
        elif n.type == 'BSDF_TRANSPARENT': seethru = 1
    assert img, f'Expecting a texture image in {m.name}'
    d, f = psplit(abspath_bl(img.filepath, library=img.library))
    return {'imgfile':f, 'imgdir':d, 'seethru':img.channels == 4 and seethru}
def write_with_backup(p, d):
    if exists(p) and (b := next((b for i in range(1000) if not exists(b := f'{p}.{i:03d}.bak')), None)):
        with open(b, 'wb') as o, open(p, 'rb') as i: o.write(i.read())
    with open(p, 'wb') as f: f.write(d)
def triangulate(m): return (m, (b := bmesh.new()).from_mesh(m), bmesh.ops.triangulate(b, faces=b.faces[:]), b.to_mesh(m), b.free(), m.calc_loop_triangles(), m.calc_tangents())[0]
def export_e3d(c, filepath):
    m, g = triangulate((o := c.object.evaluated_get(c.evaluated_depsgraph_get())).to_mesh()), lepb.asset.Geometry()
    nv, nf = len(m.vertices), len(m.loop_triangles)
    m.vertices.foreach_get('co', vs := np.empty(3*nv, 'f4'))
    m.loop_triangles.foreach_get('vertices', inds := np.empty(ni := 3*nf, 'i4'))
    g.pos, g.nor, g.uv, g.mtlidx = vs.reshape(-1, 3)[inds].reshape(-1), np.empty(3*ni, 'f4'), np.empty(2*ni, 'f4'), np.empty(nf, 'i4')
    m.corner_normals.foreach_get('vector', g.nor)
    m.uv_layers[0].data.foreach_get('uv', g.uv)
    flipv(g.uv.reshape(-1, 2), in_place=1)
    m.loop_triangles.foreach_get('material_index', g.mtlidx)
    fp = psplit(p := abspath_bl(filepath))
    g.mtls = [mtlinfo(m, fp[0]) for m in m.materials]
    write_with_backup(p, lepb.asset.E3d(p, geom=g).encode())
    o.to_mesh_clear()
def mtl_decal(d, ad):
    if m := bpy.data.materials.get(n := f'mtl-decal-{d.imgfile}'): return m
    g = Graph(m := bpy.data.materials.new(n))
    if not (i := bpy.data.images.get(imgname := f'img-decal-{d.imgfile}')): (i := bpy.data.images.load(find_asset(ad, pj('2dobjects', 'ground'), d.imgfile))).name = imgname
    g.node('TexImage', image=i)
    if i.channels == 4:
        g.node('BsdfTransparent')
        g.node('BsdfDiffuse', 'below')
        g.edge('Color', 'Color', -3)
        g.node('MixShader')
        g.edge('Alpha', 'Fac', -4)
        g.edge('BSDF', 1, -3)
        g.edge('BSDF', 2, -2)
    else:
        g.node('BsdfDiffuse')
        g.edge('Color', 'Color')
    opacity_controller(g)
    g.node('OutputMaterial')
    g.edge(0, 'Surface')
    m['alpha_cutoff'] = d.alpha_cutoff
    return m
def msh_decal(ad, path, name):
    if m := bpy.data.meshes.get(n := 'msh-decal-'+name): return m
    if not path: return None
    d, m = lepb.asset.Decal(path), bpy.data.meshes.new(n)
    x, y = (0.5*v for v in d.objsz)
    m.from_pydata([(-x,-y,0),(x,-y,0),(x,y,0),(-x,y,0)],[],[(0,1,2,3)])
    m.materials.append(mtl_decal(d, ad))
    m.uv_layers.new(do_init=0).data.foreach_set('uv', flat((np.array(d.imgbox).reshape(-1,2) / d.imgdiv).reshape(-1)[[0,1,2,1,2,3,0,3]], 'f4'))
    return m
def import_2d0(context, filepath):
    ad = get_assetdirs(context, p := Path(abspath_bl(filepath)).absolute())
    m = msh_decal(ad, str(p), p.name)
    return linkcl(context, new_bpy(m.name, m, lprops=dict(group='decals', apath=relapath(ad, p))))
def export_2d0(context, filepath):
    m, d = (o := context.object).data, lepb.asset.Decal()
    d.alpha_cutoff = (mtl := m.materials[0]).get('alpha_cutoff', 0)
    m.vertices.foreach_get('co', vs := np.empty(3 * (nv := len(m.vertices)), 'f4'))
    d.imgfile = psplit(abspath_bl((img := next(n.image for n in mtl.node_tree.nodes if n.type == 'TEX_IMAGE')).filepath, library=img.library))[-1]
    d.objsz, d.imgdiv, d.type = (-2 * vs[:2]).tolist(), [v - 1 for v in img.size], 'ground'
    m.uv_layers[0].data.foreach_get('uv', uv := np.empty(2 * nv, 'f4'))
    d.imgbox = (uv[[0,1,2,5]].reshape(-1,2) * d.imgdiv).astype('i4').reshape(-1).tolist()
    write_with_backup(filepath, d.encode())
def colorramp(r, interp, elems):
    r.interpolation, r.color_mode, l = interp, 'RGB', r.elements
    while len(l) > len(elems): l.remove(l[-1])
    while len(l) < len(elems): l.new(0)
    for i,e in enumerate(elems): l[i].position, l[i].color = e[0], (*e[1],1)
def mtl_htile(l, n):
    if m := bpy.data.materials.get(n): return m
    g, b, u, e = Graph(m := bpy.data.materials.new(n)), 255, 1/256, l.e
    m.preview_render_type = 'FLAT'
    g.node('ObjectInfo')
    g.node('SeparateXYZ')
    g.edge('Location', 'Vector')
    f = g.node('Value', 'below', label='map z').outputs[0].driver_add('default_value')
    (v := f.driver.variables.new()).type = 'TRANSFORMS'
    (t := v.targets[0]).id, t.data_path, t.transform_space, t.transform_type = l.maproot, 'location', 'WORLD_SPACE', 'LOC_Z'
    f.driver.expression = v.name
    g.node('Math', of=-3, operation='SUBTRACT')
    g.edge('Z', 0, -3)
    g.edge(0, 1)
    g.node('MapRange', defvals={'From Min':e.hmin, 'From Max':e.hmin + b*e.hstep})
    g.edge()
    colorramp(g.node('ValToRGB').color_ramp, 'EASE', [(0,[1,0,0]), (18*u,[0,1,0]), (48*u,[0,0,1]), (1,[1,1,1])])
    g.edge('Result', 'Fac')
    colorramp(g.node('ValToRGB', 'below').color_ramp, 'CONSTANT', [(0,(.1,)*3), (u,(1,)*3), (256*u,(.1,)*3)])
    g.edge('Result', 'Fac', -3)
    g.node('Mix', of=-3, data_type='RGBA', blend_type='MULTIPLY', defvals={'Factor':1})
    g.edge('Color', 'A', -3)
    g.edge('Color', 'B')
    g.node('BsdfDiffuse')
    g.edge('Result', 'Color')
    g.node('OutputMaterial')
    g.edge('BSDF', 'Surface')
    return m
def msh_htile(l, n):
    if not (m := bpy.data.meshes.get(n)):
        m, r = bpy.data.meshes.new(n), 0.48*l.e.hsize
        m.from_pydata([(-r,-r,0),(r,-r,0),(r,r,0),(-r,r,0)],[],[(0,1,2,3)])
        m.materials.append(mtl_htile(l, n))
    return m
class State:
    __slots__ = 'loaders evdata mdlset acache dirs'.split()
    def __init__(s): s.loaders, s.evdata, s.mdlset, s.acache, s.dirs = {}, {}, None, lepb.asset.Cache(), {}
def get_state():
    if not (r := getattr(s := bpy.types.Scene, k := 'lepb_state', None)): setattr(s, k, r := State())
    return r
k_mouse = 'mouse_x mouse_y'.split()
def copy_evdata(e): get_state().evdata.update((k, getattr(e, k)) for k in k_mouse)
def get_mouse(): return [d[k] for k in k_mouse] if (d := get_state().evdata) else [0,0]
def mksuf(l, s): return l.suffix if s is None else s
class Loader:
    __slots__ = 'e ad curcl maproot mrn tag suffix msgs'.split()
    def __init__(l, context, e, tag):
        l.e, l.ad, l.curcl, l.maproot, l.mrn, l.tag, l.suffix, l.msgs = e, get_assetdirs(context, e.path), context.scene.collection, None, None, tag, '-'+tag, []
        get_state().loaders[tag] = l
    def cl(l, name, parent=None, suffix=None):
        if not (r := bpy.data.collections.get(n := name + mksuf(l, suffix))): (parent or l.curcl).children.link(r := bpy.data.collections.new(n))
        l.curcl = r
        return r
    def obj(l, name, data, suffix=None, cprops={}, **attrs):
        if not (r := bpy.data.objects.get(n := name + mksuf(l, suffix))): l.curcl.objects.link(r := new_bpy(n, data, cprops=cprops, **attrs))
        return r
    def err(l, m): l.msgs.append(({'ERROR'}, m))
def obj_hgrid(l, name, pos, h):
    m, r = bpy.data.meshes.new(name), 0.5*(e := l.e).hsize
    xy = np.meshgrid(*[np.linspace(r, n*r*2 - r, n, dtype='f4') for n in h.shape[::-1]])
    m.vertices.add(h.size)
    m.vertices.foreach_set('co', np.stack((*xy, h.astype('f4')*e.hstep + e.hmin), axis=-1).ravel())
    m.update()
    o = l.obj(name, m, parent=l.maproot, location=pos, instance_type='VERTS', hide_render=1, hide_viewport=1, lprops=dict(group='height'))
    l.obj('htile-'+name, msh_htile(l, f'htile{l.suffix}'), parent=o, hide_render=1).hide_set(1)
    return o
def gen_grid(n, m, d, z=0, dv='f4', di='i4'):
    xy = np.meshgrid(*[np.linspace(0, d*k, k+1, dtype=dv) for k in (m, n)])
    verts = np.stack((*xy, np.full((n+1, m+1), z, dv)), -1)
    inds = (np.arange(0, n*m+n-m, m+1, di).repeat(m) + np.tile(np.arange(m, dtype=di), n)).repeat(4) + np.tile(np.array([0,1,m+2,m+1], di), n*m)
    uvs = np.tile(np.array([0,0,1,0,1,1,0,1], dv), n*m)
    return verts, inds, uvs
def mtl_terrain(b, ad):
    if m := bpy.data.materials.get(n := f'terrain-tile{b}'): return m
    g = Graph(m := bpy.data.materials.new(n))
    m.preview_render_type = 'FLAT'
    if b == 255: g.node('BsdfTransparent')
    else:
        if not (i := bpy.data.images.get(n)): (i := bpy.data.images.load(find_asset(ad, '3dobjects', f'tile{b}', imgexts))).name = n
        g.node('TexImage', image=i)
        g.node('BsdfDiffuse')
        g.edge('Color', 'Color')
    opacity_controller(g)
    g.node('OutputMaterial')
    g.edge(0, 'Surface')
    return m
def remove_handler(k, h):
    l, n = getattr(bpy.app.handlers, k), h.__name__
    for o in [f for f in l if f.__name__ == n]: l.remove(o)
    return l
def install_handler(k, h): remove_handler(k, h).append(bpy.app.handlers.persistent(h))
def _is_water(b): return (b == 0) | ((230 < b) & (b < 255))
tbl_water = _is_water(np.arange(256))
def ndg_water():
    if r := bpy.data.node_groups.get(n := 'elm-lower-water'): return r
    g = Graph(r := bpy.data.node_groups.new(n, 'GeometryNodeTree'), pre='GeometryNode')
    for k in ('INPUT', 'OUTPUT'): r.interface.new_socket('Geometry', socket_type='NodeSocketGeometry', in_out=k)
    g.node('NodeGroupInput', pre='')
    g.node('InputNamedAttribute', 'below', data_type='BOOLEAN', defvals={'Name':'water'})
    g.node('SetPosition', of=-3, defvals={'Offset':(0,0,-0.25)})
    g.edge(src=-3)
    g.edge(dst_in='Selection')
    g.node('NodeGroupOutput', pre='')
    g.edge()
    return r
def hdl_deps(scene, depsgraph):
    if (a := getattr(c := bpy.context, 'active_operator', None)) and a.bl_idname == 'OBJECT_OT_material_slot_assign': any(update_water(o) for o in c.selected_objects if o.mode == 'EDIT' and o.type == 'MESH' and o.data.attributes.get('water'))
def tilenum_fn(p, d=None): return int(v) if (f := Path(p).name).startswith('tile') and (i := f.find('.', 4)) != -1 and (v := f[4:i]).isdigit() else d
def tilenum_mtl(m): return next((b for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and (b := tilenum_fn(n.image.filepath)) is not None), 255)
def terrain_tilenums(m):
    m.polygons.foreach_get('material_index', mi := np.empty(len(m.polygons), 'i4'))
    return np.fromiter((tilenum_mtl(t) for t in m.materials), 'B')[mi]
def update_water(o):
    if o.get(k := 'lepb-updating-water'): return
    o[k] = 1
    bpy.ops.object.mode_set()
    (m := o.data).attributes['water'].data.foreach_set('value', tbl_water[terrain_tilenums(m)])
    bpy.ops.object.mode_set(mode='EDIT')
    del o[k]
def obj_terrain(l, name, pos, t):
    m, d = bpy.data.meshes.new(name), (e := l.e).tsize
    verts, inds, uvs = gen_grid(*t.shape, e.tsize)
    vs = verts.reshape(-1, 3)[inds]
    m.vertices.add(nv := len(vs))
    m.vertices.foreach_set('co', flat(vs, 'f4'))
    m.loops.add(nv)
    m.loops.foreach_set('vertex_index', np.arange(nv, dtype='i4'))
    m.polygons.add(nf := t.size)
    m.polygons.foreach_set('loop_start', np.arange(0, nv, 4, 'i4'))
    m.polygons.foreach_set('loop_total', np.full(nf, 4, 'i4'))
    m.attributes.new('water', 'BOOLEAN', 'FACE').data.foreach_set('value', flat(tbl_water[t], '?'))
    tu, ti = np.unique(t, return_inverse=1)
    for b in tu: m.materials.append(mtl_terrain(b, l.ad))
    m.polygons.foreach_set('material_index', flat(ti, 'i4'))
    m.uv_layers.new(do_init=0).data.foreach_set('uv', flat(uvs, 'f4'))
    m.update(calc_edges=1)
    (o := l.obj(name, m, location=pos, parent=l.maproot, lprops=dict(group='terrain'))).modifiers.new('lower-water', 'NODES').node_group = ndg_water()
    return o
def chunkname(ij): return f'chunk{ij[0]},{ij[1]}'
def assetpath(l, p):
    b = Path(p)
    a = [t for d in l.ad if (t := d/b).exists()]
    if not a: l.err(f'asset {p} not found in {l.ad}')
    return str(a[0]) if a else None, b.name
np_strings_decode = np.char.decode if np.__version__[0] < '2' else np.strings.decode
def load_meshes(l, a, oi):
    pu, pi, du = *(u := np.unique(a.apath, return_inverse=1)), np_strings_decode(u[0])
    lp, ar, ah, ac = [assetpath(l, p) for p in du], np.radians(a.rot), a.blend == 20, np.where((ec := a.scale) < 0.0625, 1, ec).repeat(3).reshape(-1,3)
    for i,m in enumerate(a):
        p, n, h, o = *lp[pi[i]], bool(ah[i])*'-placeholder', oi[i].item()
        l.obj(f'mesh{o}{h}-{n}', None if h else msh_e3d(l.ad, p, n), parent=l.maproot, location=m.pos, rotation_euler=ar[i], rotation_mode='YXZ', scale=ac[i], lprops=dict(group='meshes', ordinal=o, apath=du[pi[i]], blend=m.blend, emit_light=bool(m.emit), emit_color=m.color))
def load_decals(l, a, oi):
    pu, pi, du = *(u := np.unique(a.apath, return_inverse=1)), np_strings_decode(u[0])
    lp, ar, ap, dz, ddz, dzmax = [assetpath(l, p) for p in du], np.radians(a.rot), a.pos, np.take(l.e.decal_dz, oi, mode='wrap') if (use_dz := get_opts().use_dz) else np.zeros(a.size, 'f4'), l.e.ddz, l.e.decal_dz[-1]
    if use_dz: (ap := ap.copy())[:,2] += dz
    for i,d in enumerate(a):
        p, n, o = *lp[pi[i]], oi[i].item()
        l.obj(f'decal{o}-{n}', msh_decal(l.ad, p, n), parent=l.maproot, location=ap[i], rotation_euler=ar[i], lprops=dict(group='decals', ordinal=o, apath=du[pi[i]], dz=dz[i].item(), dz_added=use_dz))
light_energy_multiplier = 100
def load_lights(l, a, oi):
    ac, ae = a.color * np.reciprocal(m := np.max(a.color, axis=1)).reshape(-1,1), m*light_energy_multiplier
    for i,t in enumerate(a): l.obj(n := f'light{(o := oi[i].item())}', new_bpy(n+l.suffix, 'POINT', bdc='lights', color=ac[i], energy=ae[i], use_shadow=0, use_soft_falloff=1, shadow_soft_size=2), parent=l.maproot, location=t.pos, show_in_front=1, lprops=dict(group='lights', ordinal=o))
def load_particles(l, a, oi):
    dp = np_strings_decode(a.apath)
    for i,t in enumerate(a): l.obj(f'particles{(o := oi[i].item())}', None, parent=l.maproot, location=t.pos, empty_display_type='SPHERE', lprops=dict(group='particles', ordinal=o, apath=dp[i]))
def load_chunk(l, ij):
    c, cc = l.e.chunks[ij], l.cl(n := chunkname(ij))
    obj_hgrid(l, f'height-{n}', c.pos, c.height)
    obj_terrain(l, f'terrain-{n}', c.pos, c.terrain)
    l.cl(f'meshes-{n}', parent=cc)
    load_meshes(l, c.meshes, c.ordinals.meshes)
    l.cl(f'decals-{n}', parent=cc)
    load_decals(l, c.decals, c.ordinals.decals)
    l.cl(f'lights-{n}', parent=cc)
    load_lights(l, c.lights, c.ordinals.lights)
    l.cl(f'particles-{n}', parent=cc)
    load_particles(l, c.particles, c.ordinals.particles)
    cc.objects[cc.name]['loaded'] = True
def get_loader(c, r):
    if not (l := get_state().loaders.get(r['tag'])): init_map(l := Loader(c, lepb.asset.Elm(p := r['epath'], initxy=r.get('initxy') if not exists(p) else None), r['tag']), r['tps'])
    return l
def init_map(l, tps):
    (e := l.e).chunkify(tps)
    cl_map = l.cl('map')
    l.maproot, l.mrn = (r := l.obj('map', None, empty_display_type='ARROWS', empty_display_size=5, cprops=dict(epath=e.path, tag=l.tag, tps=tps, inside=bool(e.inside), ambient=e.ambient))), r.name
    r.id_properties_ui('inside').update(description='No day/night lighting.')
    r.id_properties_ui('ambient').update(subtype='COLOR', description='Ambient light color if "inside" is set.', min=0, max=1)
    for ij,c in np.ndenumerate(e.chunks):
        cc = l.cl(n := chunkname(ij), parent=cl_map)
        l.obj(n, None, location=c.pos, parent=r, empty_display_type='CUBE', cprops=dict(ij=ij, loaded=False))
def mktag(c=None, p=None): return psplit(p)[-1] if p else f'new.{next_tagnum(c):03}.elm'
def in_map(l, ij): return 0 <= ij[0] < (s := l.e.chunks.shape)[0] and 0 <= ij[1] < s[1]
def init_chunks(l, opts): any(load_chunk(l, ij) for ij in (tuple(int(v) for v in s.split(',', 2)) for s in opts.chunks_to_load.split(';') if s) if in_map(l, ij))
def import_elm(c, p):
    init_map(l := Loader(c, lepb.asset.Elm(p), mktag(c, p)), (opts := get_opts(c)).chunk_size)
    init_chunks(l, opts)
    return l
def obj_elm(c, x, y):
    assert (d := (opts := get_opts(c)).asset_path or get_prefs(c).default_asset_path), 'Asset path must be set in lepb panel or addon prefs'
    init_map(l := Loader(c, lepb.asset.Elm(path=pj(d, t := mktag(c)), initxy=(x, y)), t), opts.chunk_size)
    l.maproot['initxy'] = x, y
    init_chunks(l, opts)
    return l.maproot
def is_map(o): return o.name.startswith('map') and o.get('tag') is not None
def find_map(c):
    if (o := c.object) and is_map(o): return o
    if o and (p := o.parent) and is_map(p): return p
    if len(l := list(get_state().loaders.values())) == 1 and (n := l[0].mrn) and (r := bpy.data.objects.get(n)): return r
    for g in (c.selected_objects, bpy.data.objects):
        if len(l := [o for o in g if is_map(o)]) == 1: return l[0]
    return None
def trunci(a): return np.trunc(a).astype('i4')
def gather_terrain(e, a, lc):
    loc = np.fromiter((o.location.yx for o in a), '2f4', a.size)
    dim = np.fromiter((o.dimensions.yx for o in a), '2f4', a.size)
    icsz = (itsz := 1 / e.tsize) / (tps := int(np.max(dim) * itsz))
    if (t := getattr(e, 'terrain', None)) is None: t = np.full(trunci(np.max(loc + dim, 0) * itsz), 255, 'B')
    if not t.flags.writeable: t = t.copy()
    n, ij, sa = tps, trunci(loc * icsz), trunci(dim * itsz)
    for o,(i,j),s in zip(a, ij, sa): t[n*i:n*i+n,n*j:n*j+n] = terrain_tilenums(o.data).reshape(s)
    e.terrain = t
def gather_height(e, a, lc):
    loc = np.fromiter((o.location.yx for o in a), '2f4', a.size)
    dim = 0.5 + np.fromiter((o.dimensions.yx for o in a), '2f4', a.size)
    hpt = e.tsize * (ihsz := 1 / e.hsize)
    icsz = ihsz / (hps := int(np.max(dim) * ihsz))
    hs = tuple(int(v * hpt) for v in e.terrain.shape)
    if (h := getattr(e, 'height', None)) is None: h = np.full(hs, 0, 'B')
    assert hs == h.shape, f'Height map shape {h.shape} must match terrain, expecting {hs}'
    if not h.flags.writeable: h = h.copy()
    n, ij, sa, hmin, ihstep = hps, trunci(loc * icsz), trunci(dim * ihsz), e.hmin, 1 / e.hstep
    for o,(i,j),s in zip(a, ij, sa):
        o.data.vertices.foreach_get('co', v := np.empty(3 * np.prod(s), 'f4'))
        h[n*i:n*i+n,n*j:n*j+n] = np.clip((v[2::3] - hmin) * ihstep, 0, 255).astype('B').reshape(s)
    e.height = h
def idxs(a): return a.nonzero()[0]
def grparr(e, k, dt, a, lc):
    if (r := getattr(e, k, None)) is None: r = np.zeros(a.size, dt)
    if not r.flags.writeable: r = r.copy()
    return r if isinstance(r, np.recarray) else r.view(np.recarray), np.ones(len(r), '?') if (d := getattr(e, 'idxc', None)) is None or len(da := getattr(d, k)) < 1 else lc[da]
def gather_meshes(e, a, lc):
    m, loaded = grparr(e, 'meshes', e.dt_mesh, a, lc)
    ia = np.fromiter((o.lepb_props.ordinal for o in a), 'i4', a.size)
    (extant := np.zeros(m.size, '?'))[ia[(0 <= ia) & (ia < m.size)]] = 1
    m.blend[loaded & ~extant] = 20
    iu, ii, ic = np.unique(ia, return_inverse=1, return_counts=1)
    a_reps = a[repd := ic[ii] > 1]
    i_reps, dupd = idxs(repd), np.fromiter((o.name[-3:].isdigit() for o in a_reps), '?', a_reps.size)
    ia[i_reps[dupd]] = -1
    i_free, j_free, c_extra = idxs(free := m.blend == 20), 0, m.size
    if (n := a.size + np.sum(~loaded) - m.size) > 0: m = np.concat((m, np.zeros(n, m.dtype))).view(np.recarray)
    (used := np.zeros(m.size, '?'))[idxs(~loaded & ~free)] = 1
    for j in range(ia.size):
        if used[i := ia[j]] or i < 0 or i >= m.size: ia[j] = -1
        else: used[i] = 1
    for j in range(ia.size):
        if (i := ia[j]) != -1: continue
        if j_free < i_free.size:
            ia[j] = i_free[j_free]
            j_free += 1
        else:
            ia[j] = c_extra
            c_extra += 1
    ra = np.degrees(np.fromiter((o.rotation_euler for o in a), '3f4', a.size))
    for o,i,r in zip(a, ia, ra):
        m[i] = (p := o.lepb_props).apath, o.location, r, p.emit_light, p.blend, b'', p.emit_color, 1, b''
        p.ordinal = i
    e.meshes = m[:k] if (k := next((n for n in range(m.size,0,-1) if m[n-1].blend != 20), 0)) < m.size else m
def gather_decals(e, a, lc):
    d, loaded = grparr(e, 'decals', e.dt_decal, a, lc)
    ra = np.degrees(np.fromiter((o.rotation_euler for o in a), '3f4', a.size))
    added, dz = np.fromiter((o.lepb_props.dz_added for o in a), '?', a.size), np.fromiter((o.lepb_props.dz for o in a), 'f4', a.size)
    la = np.fromiter((o.location for o in a), '3f4', a.size)
    la[:,2] -= added * dz
    e.decals = np.concat((d[~loaded], np.fromiter(((o.lepb_props.apath, l, r, b'') for o,l,r in zip(a, la, ra)), d.dtype, a.size))).view(np.recarray)
def gather_lights(e, a, lc):
    l, loaded = grparr(e, 'lights', e.dt_light, a, lc)
    ma = (1/light_energy_multiplier) * np.fromiter((o.data.energy for o in a), 'f4', a.size)
    ca = ma.reshape(-1, 1) * np.fromiter((o.data.color for o in a), '3f4', a.size)
    e.lights = np.concat((l[~loaded], np.fromiter(((o.location, c, 0, b'') for o,c in zip(a, ca)), l.dtype, a.size))).view(np.recarray)
def gather_particles(e, a, lc):
    p, loaded = grparr(e, 'particles', e.dt_particle, a, lc)
    e.particles = np.concat((p[~loaded], np.fromiter(((o.lepb_props.apath, o.location, b'') for o in a), p.dtype, a.size))).view(np.recarray)
def export_elm(c, filepath):
    e, g = get_loader(c, r := find_map(c)).e, globals()
    e.inside, e.ambient = r['inside'], r['ambient'].to_list()
    a = np.fromiter((o for o in bpy.data.objects if o.parent == r), 'O')
    t = np.fromiter((ord(o.lepb_props.group[0]) for o in a), 'B', a.size)
    s = ec.shape if (ec := getattr(e, 'chunks', None)) is not None else (0,0)
    li = np.fromiter((c[0]*s[1]+c[1] for o in a[t == ord('n')] if (c := o.get('ij')) and o['loaded']), 'i4')
    (lc := np.zeros(np.prod(s), '?'))[li] = 1
    for k in 'terrain height meshes decals lights particles'.split(): g['gather_'+k](e, a[t == ord(k[0])], lc)
    write_with_backup(filepath, e.encode())
    r['epath'] = abspath(filepath)
    e.chunkify(r['tps'])
def get_cmesh(p): return get_state().acache.obtain(lepb.asset.Cmesh, p)
def get_cskel(p): return get_state().acache.obtain(lepb.asset.Cskel, p)
def get_canim(p): return get_state().acache.obtain(lepb.asset.Canim, p)
def mtl_cmodel_rects():
    if m := bpy.data.materials.get(mtlname := 'mtl-cmodel-rects'): return m
    g = Graph(m := bpy.data.materials.new(mtlname))
    if not (img := bpy.data.images.get(imgname := 'img-cmodel-rects')):
        img = bpy.data.images.new(imgname, *(s := (rects := lepb.asset.Skin.rects)['body'].size))
        a, u, v = np.empty((f := s[1], s[0], 4), 'f4'), 1 / s[0], 1 / f
        for x,y,w,h in rects.values(): a[f-y-h:f-y,x:x+w] = 0.2 + 0.8*x*u, 0.2 + 0.8*y*v, 0.2, 1
        img.pixels.foreach_set(a.ravel())
        img.update()
    g.node('TexImage', image=img)
    g.node('BsdfDiffuse')
    g.edge('Color', 'Color')
    g.node('OutputMaterial')
    g.edge(0, 'Surface')
    return m
def mtl_piece(p, skin, ad):
    ap = abbreviate_path(fp := find_asset(ad, strip_ext(skin.apath), exts=imgexts))
    if m := bpy.data.materials.get(mtlname := f'mtl-piece-{p.kind}-{skin.tag}-{ap}'): return m
    g = Graph(m := bpy.data.materials.new(mtlname))
    if not (i := bpy.data.images.get(imgname := 'img-'+ap)): (i := bpy.data.images.load(fp)).name = imgname
    g.node('TexCoord')
    r = lepb.asset.Skin.rects[skin.tag].to_uv()
    g.node('Mapping', vector_type='TEXTURE', defvals={'Location':(r.x, 1 - r.y, 0), 'Scale':(r.w, r.h, 1)})
    g.edge('UV', 'Vector')
    g.node('TexImage', image=i)
    g.edge()
    if i.channels == 4:
        g.node('BsdfTransparent')
        g.node('BsdfDiffuse', 'below')
        g.edge('Color', 'Color', -3)
        g.node('MixShader')
        g.edge('Alpha', 'Fac', -4)
        g.edge('BSDF', 1, -3)
        g.edge('BSDF', 2, -2)
    else:
        g.node('BsdfDiffuse')
        g.edge('Color', 'Color')
    g.node('OutputMaterial')
    g.edge(0, 'Surface')
    return m
def msh_cmesh(cmesh):
    if m := bpy.data.meshes.get(n := 'msh-cmesh-'+abbreviate_path(cmesh.path)): return m
    vs = np.empty((nv := sum(len(s.verts) for s in cmesh.submeshes), 3), 'f4')
    inds = np.empty((nf := sum(len(s.faces) for s in cmesh.submeshes), 3), 'i4')
    uvs, nors, ov, oi = np.zeros((nv, 2), 'f4'), np.empty((nv, 3), 'f4'), 0, 0
    for s in cmesh.submeshes:
        vs[ov:ov + (nsv := len(s.verts))] = s.verts.pos
        inds[oi:oi + (nsf := len(s.faces))] = ov + s.faces
        assert s.nuv in (0,1), f'Unhandled number of uv coordinate pairs per vertex: {s.nuv}'
        if s.nuv: uvs[ov:ov + nsv] = s.verts.uv
        nors[ov:ov + nsv] = s.verts.nor
        ov, oi = ov + nsv, oi + nsf
    m = bpy.data.meshes.new(n)
    m.vertices.add(nv)
    m.vertices.foreach_set('co', flat(vs, 'f4'))
    m.loops.add(ni := inds.size)
    m.loops.foreach_set('vertex_index', flat(inds, 'i4'))
    m.polygons.add(nf)
    m.polygons.foreach_set('loop_start', np.arange(0, ni, 3, 'i4'))
    m.polygons.foreach_set('loop_total', np.full(nf, 3, 'i4'))
    m.polygons.foreach_set('material_index', np.full(nf, 0, 'i4'))
    m.uv_layers.new(do_init=0).data.foreach_set('uv', flat(uvs[inds], 'f4'))
    m.attributes.new(k := 'temp_custom_split_normals', 'FLOAT_VECTOR', 'CORNER').data.foreach_set('vector', flat(nors[inds], 'f4'))
    m.validate(clean_customdata=0)
    (c := m.attributes[k]).data.foreach_get('vector', a := np.empty(3*len(m.loops), 'f4'))
    m.attributes.remove(c)
    m.normals_split_custom_set(a.reshape(-1,3))
    m.update(calc_edges=1)
    return m
def import_cmf(c, p, r): return rigup_piece(r, o) if (o := linkcl(c, new_bpy(basename(p), msh_cmesh(get_cmesh(p)), cprops=dict(cmesh_path=p)))) and r else o
def gather_bones(o):
    if not o or o.type != 'ARMATURE': return [], range(999)
    l, r = [], [None]*len(d := o.data.bones)
    for i,b in enumerate(d):
        if b.name.startswith('_'): continue
        r[i] = len(l)
        l.append(b)
    return l, r
def export_cmf(c, p):
    cmesh = lepb.asset.Cmesh()
    for o in [s for s in c.selected_objects if s.type == 'MESH']:
        m = triangulate((e := o.evaluated_get(c.evaluated_depsgraph_get())).to_mesh())
        nv, nf, vgi = len(m.vertices), len(m.loop_triangles), gather_bones(o.parent)[1]
        m.vertices.foreach_get('co', vs := np.empty(3*nv, 'f4'))
        m.loop_triangles.foreach_get('vertices', inds := np.empty(ni := 3*nf, 'i4'))
        m.corner_normals.foreach_get('vector', nors := np.empty(3*ni, 'f4'))
        m.uv_layers[0].data.foreach_get('uv', uvs := np.empty(2*ni, 'f4'))
        vs, nors, uvs = vs.reshape(-1,3), nors.reshape(-1,3), uvs.reshape(-1,2)
        h = (np.hstack((vs[inds], nors, uvs)) * 1e5).astype('i4')
        ua, ix, ia = np.unique(h, axis=0, return_index=1, return_inverse=1)
        v, w, n = np.empty(len(ix), (s := lepb.asset.Submesh()).dt_vert).view(np.recarray), np.empty(sum(len(m.vertices[i].groups) for i in inds[ix]), s.dt_weight).view(np.recarray), 0
        for j,k in enumerate(ix):
            v[j] = vs[i := inds[k]], nors[k], 0, 0, uvs[k], n, (we := n + len(l := m.vertices[i].groups)), 0
            w[n:(n := we)] = [(j, vgi[g.group], g.weight) for g in l]
        s.material, s.verts, s.springs, s.faces, s.nuv, s.nlod, s.weights = 0, v, np.recarray(0, s.dt_spring), ia.astype('u4').reshape(-1,3), 1, 0, w
        cmesh.submeshes.append(s)
        e.to_mesh_clear()
    write_with_backup(p, cmesh.encode())
def cquat_to_bpy(q): return Quaternion((q[3], -q[0], -q[1], -q[2]))
def bquat_to_cal3d(q): return (-q[1], -q[2], -q[3], q[0])
def clamp(a, v, b): return a if v < a else b if v > b else v
def fill_rig_with_cskel(c, r, cskel, minbl=0.1, maxbl=1):
    bmat = lambda i, d={}, l=cskel.bones: d.get(i) or d.setdefault(i, (bmat(b.parent, d) if (b := l[i]).parent >= 0 else Matrix()) @ Matrix.Translation(b.trans) @ cquat_to_bpy(b.rot).to_matrix().to_4x4())
    select_active(c, r)
    bpy.ops.object.mode_set(mode='EDIT')
    for i,b in enumerate(cskel.bones):
        m, e = bmat(i), r.data.edit_bones.new(b.name)
        e.head, e.tail = m.to_translation(), m @ Vector((0,1,0))
        e.align_roll(m @ Vector((0,0,1)) - e.head)
        if b.parent >= 0: e.parent = r.data.edit_bones[b.parent]
        e.length = l[0] if (l := [d for j in b.children if (d := e.vector.dot(bmat(j).to_translation() - e.head)) > 0]) and len(b.children) == 1 else clamp(minbl, max(l), maxbl) if l else clamp(minbl, e.parent.length, maxbl) if e.parent else maxbl
    bpy.ops.object.mode_set()
    r.select_set(0)
    return r
def import_csf(c, p): return fill_rig_with_cskel(c, linkcl(c, new_bpy(basename(p), bpy.data.armatures.new('arma-'+abbreviate_path(p)), cprops=dict(cskel_path=p))), get_cskel(p))
def export_csf(c, p):
    cskel, bl = lepb.asset.Cskel(), gather_bones(c.object)[0]
    n, amat = {a.name:i for i,a in enumerate(bl)}, lambda i,d={}: d.get(i) or d.setdefault(i, (amat(b.parent, d) if (b := cskel.bones[i]).parent >= 0 else Matrix()) @ Matrix.Translation(b.trans) @ bl[i].matrix.to_4x4())
    for i,a in enumerate(bl):
        b = lepb.asset.Bone(a.name)
        b.parent, b.children = n[a.parent.name] if a.parent else -1, [n[t.name] for t in a.children]
        cskel.bones.append(b)
        b.trans = ap.matrix_local.to_quaternion().conjugated() @ (a.head_local - ap.head_local) if (ap := a.parent) else a.head_local
        b.rot = bquat_to_cal3d(-a.matrix.to_quaternion())
        t, r, s = amat(i).inverted_safe().decompose()
        b.ltrans, b.lrot = t, bquat_to_cal3d(-r)
    write_with_backup(p, cskel.encode())
def get_fps(c): return c.scene.render.fps * c.scene.render.fps_base
def extract_pose_bone_keyframes(i, d, k, a):
    for j in range(a.shape[-1]):
        p = d[(dp := f'pose.bones[{i}].{k}', j)].keyframe_points
        assert len(a) == len(p), 'Expecting {len(a)} keyframe points for {dp}[{j}], not {len(p)}'
        p.foreach_get('co', t := np.empty(len(a)*2, 'f4'))
        a[:,j] = t.reshape(-1,2)[:,1]
def export_caf(c, p):
    canim, cskel, action = lepb.asset.Canim(), get_cskel((o := c.object)['cskel_path']), (d := o.animation_data).action or d.nla_tracks.active.strips[0].action
    canim.duration_s = ((fr := action.frame_range)[1] - fr[0]) * (spf := 1 / get_fps(c))
    fcs = action.layers[0].strips[0].channelbag(action.slots[0]).fcurves
    d, nk = {(f.data_path, f.array_index):f for f in fcs}, len(kfp := fcs[0].keyframe_points)
    kfp.foreach_get('co', a := np.empty(nk*2, 'f4'))
    ts = (a.reshape(-1, 2)[:,0] - 1) * spf
    for i,b in enumerate(cskel.bones):
        t, am = lepb.asset.Track(), np.array((m := o.data.bones[i].matrix), 'f4')
        t.bone, t.keyframes = i, (kf := np.empty(nk, t.dt_keyframe).view(np.recarray))
        kf.time, mq = ts, m.to_quaternion()
        extract_pose_bone_keyframes(i, d, 'location', (ta := np.empty(nk, '3f4')))
        kf.trans = (am @ ta.reshape(-1,3,1)).reshape(-1,3) + b.trans
        extract_pose_bone_keyframes(i, d, 'rotation_quaternion', (qa := np.empty(nk, '4f4')))
        flip_antipodals(ra := np.fromiter((bquat_to_cal3d(mq @ Quaternion(q)) for q in qa), '4f4', nk))
        kf.rot = ra
        canim.tracks.append(t)
    write_with_backup(p, canim.encode())
def assign_skins(o, m, skins):
    if len(skins) < 1 or not m.uv_layers: return
    if len(skins) < 2: return m.polygons.foreach_set('material_index', np.ones(len(m.polygons), 'i4'))
    m.polygons.foreach_get('material_index', mi := np.empty(nf := len(m.polygons), 'i4'))
    m.uv_layers[0].data.foreach_get('uv', uvs := np.empty(2*len(m.loops), 'f4'))
    t, rects = uvs.reshape(-1, 2), lepb.asset.Skin.rects
    for i,s in enumerate(skins):
        r = rects[s.tag].to_uv()
        u0, v0, u1, v1, tu, tv = r.x, 1 - r.y - r.h, r.x + r.w, 1 - r.y, t[:,0], t[:,1]
        mi[((u0 <= tu) & (tu <= u1) & (v0 <= tv) & (tv <= v1)).reshape(-1, 3).all(axis=1)] = i + 1
    m.polygons.foreach_set('material_index', mi)
def img_stitch(cm, cms, ad):
    sw, sh = (scale := 4) * (rs := (rects := lepb.asset.Skin.rects)['body'].size)[0], scale*rs[1]
    if not (img := bpy.data.images.get(imgname := f'img-stitch-{cm.id}_{cm.name}')): img = bpy.data.images.new(imgname, sw, sh)
    (sd := set(rects)).discard('body')
    a = np.zeros((sh, sw, 4), 'f4')
    for skin in find_skins(cm, cms, sd):
        if not (p := find_asset(ad, strip_ext(skin.apath), exts=imgexts, must_exist=0)): continue
        if not (i := bpy.data.images.get('img-'+abbreviate_path(p))): continue
        r, w, h = rects[skin.tag], i.size[0], i.size[1]
        assert w == (ew := r.w*scale) and h == (eh := r.h*scale), f'Expecting image size {ew}x{eh} not {w}x{h} for {p}'
        i.pixels.foreach_get(ia := np.empty(h*w*4, 'f4'))
        x, y = r.x*scale, r.y*scale
        a[sh-y-h:sh-y, x:x+w] = ia.reshape(h, w, 4)
    img.pixels.foreach_set(a.ravel())
    img.update()
    return img
def find_skins(cm, cms, sd):
    d = {t:[] for t in sd}
    any(d[s.tag].append(s) for k,i in cms.items() for s in cm.pieces[k][i].skins if s.tag in sd)
    return [max(l, key=lambda s:s.priority) for l in d.values() if l]
def reset_skin_materials(m, p, skins, ad, keep_material_indices=0):
    if keep_material_indices: m.polygons.foreach_get('material_index', mi := np.empty(len(m.polygons), 'i4'))
    m.materials.clear()
    m.materials.append(mtl_cmodel_rects())
    for s in skins: m.materials.append(mtl_piece(p, s, ad))
    if keep_material_indices: m.polygons.foreach_set('material_index', mi)
def obj_piece(c, cm, p, ad, cms):
    if not p.mesh or not (w := find_asset(ad, p.mesh, must_exist=p.kind != 'body')): return
    m, sd = msh_cmesh(cmesh := get_cmesh(w)), p.get_skindeps()
    o = linkcl(c, new_bpy(f'{cm.id}_{cm.name}-{p.kind}_{p.id}_{p.desc}', m, cprops=dict(cmesh_path=cmesh.path, piece_kind=p.kind, piece_id=p.id, cmodel_id=cm.id)))
    skins = find_skins(cm, cms, sd)
    reset_skin_materials(m, p, skins, ad)
    assign_skins(o, m, skins)
    return o
def rigup_piece(r, o):
    o.modifiers.new('mdfr_arma-'+o.name, 'ARMATURE').object, o.parent, cmesh, cskel = r, r, get_cmesh(o['cmesh_path']), get_cskel(r['cskel_path'])
    for i,b in enumerate(cskel.bones):
        g, ov = o.vertex_groups.new(name=b.name), 0
        for s in cmesh.submeshes:
            for w in np.unique(aw := (a := s.weights[s.weights.bone == i]).weight): g.add((a.vert[aw == w] + ov).tolist(), w, 'REPLACE')
            ov += len(s.verts)
    return o
def make_fcurves(d, dp, fa, va, interp, gn):
    for i in range(va.shape[-1]):
        f = d.new(dp, index=i, group_name=gn)
        (p := f.keyframe_points).add(len(a := np.stack((fa, va[:,i]), axis=1)))
        p.foreach_set('co', flat(a, 'f4'))
        p.foreach_set('interpolation', np.full(len(a), interp, 'u1'))
        f.update()
def import_cmodel(c, ms, cm):
    name, actor, ps, ad, iv = f'{cm.id}_{cm.name}', ms.get_xmlroot(cm).actor, ms.setup_pieces(cm), get_assetdirs(c), lepb.asset.Piece.initvals
    cms = {k:d[iv[k]].id for k,d in ps.items()}
    objs = [o for k,i in cms.items() if (o := obj_piece(c, cm, ps[k][i], ad, cms))]
    minbl = 0.4*(maxbl := 0.2*max(max(o.dimensions) for o in objs))
    cskel = get_cskel(find_asset(ad, actor.skeleton.text))
    r = linkcl(c, new_bpy('rig-'+name, bpy.data.armatures.new('arma-'+name), cprops=dict(cmodel_state=cms, cmodel_id=cm.id, cskel_path=cskel.path)))
    fill_rig_with_cskel(c, r, cskel, minbl, maxbl)
    for o in objs: rigup_piece(r, o)
    if (al := [(c.tag,c) for c in t.children] if (t := actor.find_tag('frames')) else []) and len(ps) > 1: al.extend((f'{c.tag}_{k[5:]}',c) for k,t in ms.trees.items() if k.startswith('anim_') for c in t.root.children if c.tag.startswith('CAL_'))
    fill_nla_with_canims(c, r, [(find_asset(ad, (a := n.text.split(None, 1))[0]), tag[4:], 1e-3 * int(n.attrs.get('duration', 0))) for tag,n in al])
    return r, objs
def flip_antipodals(a): a[1:] *= np.multiply.accumulate(np.sign(np.sum(a[1:] * a[:-1], axis=1))).reshape(-1,1)
def fill_nla_with_canims(c, r, ptd):
    if not r.animation_data: r.animation_data_create()
    cskel, fps, lin = get_cskel(r['cskel_path']), get_fps(c), bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items['LINEAR'].value
    brinv = lambda i, d={}, l=r.data.bones: d.get(i) or d.setdefault(i, l[i].matrix.to_quaternion().conjugated())
    for path, track_name, duration in reversed(ptd):
        canim, action_name = get_canim(path), f'canim-{abbreviate_path(cskel.path)}-{abbreviate_path(path)}'
        if not (action := bpy.data.actions.get(action_name)):
            slot = (action := bpy.data.actions.new(action_name)).slots.new('OBJECT', 'Object Transforms')
            fcs = action.layers.new('Layer').strips.new(type='KEYFRAME').channelbag(slot, ensure=1).fcurves
            for ct in sorted(canim.tracks, key=lambda t:t.bone):
                b, rinv = cskel.bones[bi := ct.bone], brinv(bi)
                fa, dtrans, gn = (kf := ct.keyframes).time * fps + 1, kf.trans - b.trans, f'{bi} {b.name}'
                ta = np.fromiter((rinv @ Vector(d) for d in dtrans), '3f4', len(dtrans))
                make_fcurves(fcs, f'pose.bones[{bi}].location', fa, ta, lin, gn)
                flip_antipodals(ra := np.fromiter((rinv @ cquat_to_bpy(r) for r in kf.rot), '4f4', len(kf)))
                make_fcurves(fcs, f'pose.bones[{bi}].rotation_quaternion', fa, ra, lin, gn)
        (t := r.animation_data.nla_tracks.new()).name, t.mute = track_name, 1
        t.strips.new(track_name, 1, action).scale = duration / canim.duration_s or 1
def set_hideviewrend(o, v): o.hide_viewport, o.hide_render = v, v
def find_piece(p, l): return next((o for o in l if o.get('piece_kind') == p.kind and o.get('piece_id') == p.id), None)
def swap_piece(c, r, k, i):
    cm, cms, ad = (ms := get_mdlset(c)).cmodels[r['cmodel_id']], r['cmodel_state'], get_assetdirs(c)
    new_p, old_p, ol = cm.pieces[k][i], cm.pieces[k][cms[k]], [o for o in r.children if o.get('piece_kind')]
    if old_o := find_piece(old_p, ol): set_hideviewrend(old_o, 1)
    cms[k] = new_p.id
    if not (new_o := find_piece(new_p, ol)): new_o = obj_piece(c, cm, new_p, ad, cms)
    if new_o:
        if not new_o.parent: rigup_piece(r, new_o)
        set_hideviewrend(new_o, 0)
    changed = set(s.tag for s in old_p.skins + new_p.skins)
    for o in ol:
        if o.hide_viewport: continue
        sd = (op := cm.pieces[o['piece_kind']][o['piece_id']]).get_skindeps()
        if changed.intersection(sd): reset_skin_materials(o.data, op, find_skins(cm, cms, sd), ad, keep_material_indices=1)
from bpy.types import AddonPreferences, FileHandler, Menu, Operator, OperatorFileListElement, Panel, PropertyGroup, UIList
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper, poll_file_object_drop
from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
from mathutils.geometry import intersect_line_plane
from bl_math import lerp
class LEPB_prefs(AddonPreferences):
    bl_idname = __package__
    default_asset_path: StringProperty(name='Default Asset Path', description='Game asset directory to use if none is set in the addon panel.', default='', maxlen=2048, subtype='DIR_PATH')
    default_extra_path: StringProperty(name='Default Extra Path', description='Extra asset directory to use if none is set in the addon panel.', default='', maxlen=2048, subtype='DIR_PATH')
    def draw(self, context):
        self.layout.prop(self, 'default_asset_path', text='Default Asset Path')
        self.layout.prop(self, 'default_extra_path', text='Default Extra Path')
def get_prefs(c=None): return (c or bpy.context).preferences.addons[__package__].preferences
class MeshImportOpts:
    sharp_angle: IntProperty(name='Edge Sharp Angle', description='Mark edges "sharp" based on the angle in degrees between adjacent faces. -1 disables.', default=45, min=-1, max=180)
    shade_smooth: BoolProperty(name='Shade Smooth', description='Set "shade smooth" on imported mesh objects.', default=1)
class MapImportOpts:
    chunk_size: IntProperty(name='Chunk Size', description='Number of terrain tiles per chunk side. Each terrain is 3x3 world units (meters), or 6x6 height tiles (walkable positions in game). Only applied when creating or importing a map.', default=8, min=1, max=1024)
    chunks_to_load: StringProperty(name='Chunks to load', description='Semi-colon separated list of i,j row-column chunk coordinates to load automatically after map import.', default='0,0')
class LEPB_opts(PropertyGroup, MeshImportOpts, MapImportOpts):
    asset_path: StringProperty(name='Asset Path', description='The base directory used to find game data files such as images textures when loading assets that contain relative path references. Should contain the file "files.lst".', default='', maxlen=2048, subtype='DIR_PATH')
    extra_path: StringProperty(name='Extra Asset Directory', description='Additional directory to search in first when loading referenced assets. For example downloaded client assets in the game config subdirectory "updates/1_9_5_0". Should contain the file "files.lst".', default='', maxlen=2048, subtype='DIR_PATH')
    minute: IntProperty(name='Game Minute', description='Time of day in game. Hours 0 to 3 are daylight, 3 to 6 are night.', default=60, min=0, max=359)
    use_dz: BoolProperty(name='Decal Z Offset', description='Mimick the way the game client offsets the z locations of decals to reduce z-fighting. Adjusting the near and far clip planes depending on the camera position may be more reliable. Only applied during load.', default=1)
    cur_tagnum: IntProperty(default=0, options={'HIDDEN'})
    opacity: IntProperty(name='Map Opacity', description='Modify the transparency of all terrain, mesh and decal map objects in the 3D viewport to make height map tiles more easily visible. Applied when viewport shading is material preview or rendered.', default=100, min=0, max=100, step=1, subtype='PERCENTAGE')
def get_opts(c=None): return (c or bpy.context).scene.lepb_uidata.opts
def next_tagnum(c=None):
    (o := get_opts(c)).cur_tagnum += 1
    return o.cur_tagnum
def annots(o): return ((r := {}), any(r.update(getattr(c, '__annotations__', {})) for c in type(o).mro()))[0]
def update_opts(c, src): return ((o := get_opts(c)), any(setattr(o, k, v) for k in annots(o).keys() if (v := getattr(src, k, None)) is not None))[0]
def sync_opts(c, dst): return ((o := get_opts(c)), any(setattr(dst, k, v) for k in annots(dst).keys() if (v := getattr(o, k, None)) is not None))[0]
class LEPB_props(PropertyGroup):
    group: EnumProperty(name='group', items=[('none', 'None', '', 0), ('height', 'Height Tile Grid', '', 1), ('terrain', 'Terrain Tile Grid', '', 2), ('meshes', 'Mesh 3D Objects', '', 3), ('decals', 'Decal 2D Objects', '', 4), ('lights', 'Lights', '', 5), ('particles', 'Particle Systems', '', 6)], default=0)
    apath: StringProperty(name='apath', default='')
    ordinal: IntProperty(name='ordinal', description='Index in the ELM mesh array, used as the map object ID in the client-server protocol. Automatically assigned during export.', default=-1, min=-1)
    blend: IntProperty(name='blend', description='A value of 0 means opaque, 1 transparent, and 20 for invisible placeholders (objects previously deleted but kept to preserve array order).', default=0, min=0, max=20)
    emit_light: BoolProperty(name='emit_light', description='Whether a mesh 3D object emits light.', default=0)
    emit_color: FloatVectorProperty(name='emit_color', subtype='COLOR')
    dz: FloatProperty(name='dz', default=0, min=0, max=lepb.asset.Elm.decal_dz[-1], step=lepb.asset.Elm.ddz, precision=9)
    dz_added: BoolProperty(name='dz_added', description='Whether a decal 2D object z position was offset.', default=0)
    def update(p, d): any(setattr(p, k, v) for k,v in d.items())
class Importer(Operator, ImportHelper):
    files: CollectionProperty(name='Files', type=OperatorFileListElement)
    directory: StringProperty(name='Directory', subtype='DIR_PATH')
    parent: StringProperty(options={'HIDDEN'})
    spread: BoolProperty(options={'HIDDEN'})
    poll = classmethod(lambda _,c: c.mode == 'OBJECT')
    def execute(s, c):
        l, p, opts = c.scene.cursor.location.copy(), bpy.data.objects.get(s.parent), update_opts(c, s)
        select_none(c)
        for f in s.files:
            if (o := s.do_import(c, pj(s.directory, f.name))):
                if s.spread and hasattr(o, 'location'): o.location, l.x = l, l.x + o.dimensions.x * 1.1
                if p: o.parent = p
                o.select_set(1)
        if s.files and o: select_active(c, o)
        return {'FINISHED'}
class MapAssetImporter(Importer):
    parent: StringProperty(name='Map', description='The imported map asset will be parented to the given map root object, so that it will be included in the ELM file when the map is exported.', default='', search=lambda _,c,text: (o.name for o in bpy.data.objects if o.get('tag') and text in o.name))
    spread: BoolProperty(name='Spread along X-axis', default=1, description='When importing multiple objects at once, displace each of them on the x-axis so they are spread out on a line, and not all on top of each other.')
    def invoke(s, c, e):
        if not s.parent and (r := find_map(c)): s.parent = r.name
        return super().invoke(c, e)
class LEPB_OT_import_e3d(MapAssetImporter, MeshImportOpts):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.import_e3d', 'Import E3D Mesh', {'REGISTER','UNDO'}, 'Import a 3D object mesh asset.', '.e3d'
    filter_glob: StringProperty(default='*.e3d;*.e3d.gz', options={'HIDDEN'})
    def do_import(s, c, p): return import_e3d(c, p)
class LEPB_OT_export_e3d(Operator, ExportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.export_e3d', 'Export E3D Mesh', {'PRESET'}, 'Export the active object as a 3D object mesh asset.', '.e3d'
    filter_glob: StringProperty(default='*.e3d', options={'HIDDEN'})
    poll = classmethod(lambda _,c: (o := c.object) and o.type == 'MESH' and c.mode == 'OBJECT')
    def execute(s, c): return ({'FINISHED'}, export_e3d(c, s.filepath))[0]
class LEPB_OT_import_2d0(MapAssetImporter):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.import_2d0', 'Import 2D0 Decal', {'REGISTER','UNDO'}, 'Import a 2D object decal asset.', '.2d0'
    filter_glob: StringProperty(default='*.2d0', options={'HIDDEN'})
    def do_import(s, c, p): return import_2d0(c, p)
class LEPB_OT_export_2d0(Operator, ExportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.export_2d0', 'Export 2D0 Decal', {'PRESET'}, 'Export the active object as a 2D decal asset.', '.2d0'
    filter_glob: StringProperty(default='*.2d0', options={'HIDDEN'})
    poll = classmethod(lambda _,c: (o := c.object) and o.type == 'MESH' and c.mode == 'OBJECT')
    def execute(s, c): return ({'FINISHED'}, export_2d0(c, s.filepath))[0]
class LEPB_OT_import_elm(Operator, ImportHelper, MapImportOpts):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.import_elm', 'Import ELM Map', {'REGISTER','UNDO'}, 'Import a map and all of its assets.', '.elm'
    filter_glob: StringProperty(default='*.elm;*.elm.gz', options={'HIDDEN'})
    def invoke(s, c, e): return (sync_opts(c, s), super().invoke(c, e))[1]
    def execute(s, c):
        opts, l = update_opts(c, s), import_elm(c, s.filepath)
        for t,m in l.msgs: s.report(t, m)
        select_only(c, l.maproot)
        return {'FINISHED'}
class LEPB_OT_export_elm(Operator, ExportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.export_elm', 'Export ELM Map', {'REGISTER','UNDO','PRESET'}, 'Export map rooted at the active object (an empty object shown as xyz axes and named "map...").', '.elm'
    filter_glob: StringProperty(default='*.elm', options={'HIDDEN'})
    poll = classmethod(lambda _,c: find_map(c))
    def execute(s, c): return ({'FINISHED'}, export_elm(c, s.filepath))[0]
class LEPB_OT_import_cmf(Importer):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.import_cmf', 'Import Cal3D Mesh', {'REGISTER','UNDO'}, 'Import a Cal3D mesh asset. This is just the vertex data, without corresponding textures, skeletons, or animations.', '.cmf'
    filter_glob: StringProperty(default='*.cmf', options={'HIDDEN'})
    parent: StringProperty(name='Armature', description='Parent to an existing Cal3D skeleton. Alsoadds an armature modifier and vertex groups based on bone weights in the CMF file.', default='', search=lambda _,c,t: (o.name for o in bpy.data.objects if o.get('cskel_path') and t in o.name))
    def do_import(s, c, p): return import_cmf(c, p, bpy.data.objects.get(s.parent))
class LEPB_OT_export_cmf(Operator, ExportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.export_cmf', 'Export Cal3D Mesh', {'PRESET'}, 'Export selected mesh objects as submeshes of a Cal3D mesh asset.', '.cmf'
    filter_glob: StringProperty(default='*.cmf', options={'HIDDEN'})
    poll = classmethod(lambda _,c: c.mode == 'OBJECT' and any(o for o in c.selected_objects if o.type == 'MESH'))
    def execute(s, c): return ({'FINISHED'}, export_cmf(c, s.filepath))[0]
class LEPB_OT_import_csf(Importer):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.import_csf', 'Import Cal3D Skeleton', {'REGISTER','UNDO'}, 'Import a Cal3D skeleton.', '.csf'
    filter_glob: StringProperty(default='*.csf', options={'HIDDEN'})
    def do_import(s, c, p): return import_csf(c, p)
class LEPB_OT_export_csf(Operator, ExportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.export_csf', 'Export Cal3D Skeleton', {'PRESET'}, 'Export the active armature as a Cal3D skeleton.', '.csf'
    filter_glob: StringProperty(default='*.csf', options={'HIDDEN'})
    poll = classmethod(lambda _,c: c.mode == 'OBJECT' and (o := c.object) and o.type == 'ARMATURE')
    def execute(s, c): return ({'FINISHED'}, export_csf(c, s.filepath))[0]
class LEPB_OT_import_caf(Operator, ImportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.import_caf', 'Import Cal3D Animation', {'REGISTER','UNDO'}, 'Import Cal3D animations as NLA tracks for a Cal3D skeleton.', '.caf'
    files: CollectionProperty(name='Files', type=OperatorFileListElement)
    directory: StringProperty(name='Directory', subtype='DIR_PATH')
    filter_glob: StringProperty(default='*.caf', options={'HIDDEN'})
    duration: FloatProperty(name='Duration', subtype='TIME_ABSOLUTE', min=0, unit='TIME_ABSOLUTE', description='Playback duration in seconds. If zero the value in the file is used.')
    poll = classmethod(lambda _,c: c.mode in ('OBJECT','POSE') and (o := c.object) and o.type == 'ARMATURE' and o.get('cskel_path'))
    def execute(s, c): return ({'FINISHED'}, fill_nla_with_canims(c, c.object, [(abspath(pj(s.directory, f.name)), chomp(f.name, '.caf'), s.duration) for f in s.files]))[0]
class LEPB_OT_export_caf(Operator, ExportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.export_caf', 'Export Cal3D Animation', {'PRESET'}, 'Export the active object\'s current action as a Cal3D animation.', '.caf'
    filter_glob: StringProperty(default='*.caf', options={'HIDDEN'})
    poll = classmethod(lambda _,c: (o := c.object) and c.mode in ('OBJECT','POSE') and (d := o.animation_data) and (d.action or ((t := d.nla_tracks.active) and len(t.strips))))
    def execute(s, c): return ({'FINISHED'}, export_caf(c, s.filepath))[0]
class LEPB_OT_new_map(Operator, MapImportOpts):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.new_map', 'New Map', {'REGISTER','UNDO'}, 'Create an empty map with terrain and height grids.'
    tx: IntProperty(name='X-Axis Terrain Count', description='Number of terrain tiles along the x-axis.', default=32, min=1, max=65536)
    ty: IntProperty(name='Y-Axis Terrain Count', description='Number of terrain tiles along the y-axis.', default=32, min=1, max=65536)
    poll = classmethod(lambda _,c: c.mode == 'OBJECT')
    def invoke(s, c, e): return (sync_opts(c, s), c.window_manager.invoke_props_dialog(s))[1]
    def execute(s, c): return ({'FINISHED'}, update_opts(c, s), select_only(c, obj_elm(c, s.tx, s.ty)))[0]
class LEPB_OT_load_chunks(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.load_chunks', 'Load Chunks', {'REGISTER', 'UNDO'}, 'Load contents of all selected chunks (cube empties named "chunkI,J...").'
    poll = classmethod(lambda _,c: next((o for o in c.selected_objects if o.get('ij') and not o['loaded']), None))
    def execute(self, c):
        for o in c.selected_objects:
            if (ij := o.get('ij')) and not o['loaded']: load_chunk(get_loader(c, o.parent), tuple(ij))
        return {'FINISHED'}
def sethhvp(c, v):
    if (l := [o for o in c.selected_objects if o.name.startswith('height')]):
        for o in l: o.hide_viewport = v
    elif (o := c.object) and (r := o.parent or find_map(c)) and (tps := r.get('tps')) and len(a := np.fromiter((o.location.yx for o in c.selected_objects), '2f4', len(c.selected_objects))):
        icsz, d, suf = 1 / (tps * lepb.asset.Elm.tsize), bpy.data.objects, get_loader(c, r).suffix
        for ij in np.unique((a * icsz).astype('i4'), axis=0):
            if (o := d.get(f'height-{chunkname(ij)}{suf}')): o.hide_viewport = v
class LEPB_OT_hide_height(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.hide_height', 'Hide Height Tiles', {'REGISTER', 'UNDO'}, 'Hide height maps.'
    def execute(s, c): return ({'FINISHED'}, sethhvp(c, 1))[0]
class LEPB_OT_show_height(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.show_height', 'Show Height Tiles', {'REGISTER', 'UNDO'}, 'Show height maps in the viewport for loaded chunks containing all selected objects.'
    def execute(s, c): return ({'FINISHED'}, sethhvp(c, 0))[0]
class LEPB_OT_update_lighting(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.update_lighting', 'Update Lighting', {'REGISTER', 'UNDO'}, 'Change world lighting based on the "Game Minute" setting and the custom properties "inside" and "ambient" of the current map.'
    poll = classmethod(lambda _,c: find_map(c))
    def execute(self, c):
        n, co, a = 'Sun', c.scene.collection.objects, (r := find_map(c))['ambient']
        if r['inside']:
            if (o := co.get(n)): o.data.energy = 0
        else:
            a, m, emax, ang = (0.05087609,)*3, get_opts(c).minute, 5, np.pi/2
            if not (o := co.get(n)): co.link(o := new_bpy(n, new_bpy(n, 'SUN', bdc='lights', energy=emax), location=(0,0,20)))
            o.rotation_euler = 0, lerp(ang, -ang, m/180), 0
            o.data.energy = lerp(0, emax, m/90) if 0 <= m < 90 else lerp(emax, 0, (m-90)/90) if 90 <= m < 180 else 0
        c.scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = *a, 1
        return {'FINISHED'}
class LEPB_OT_toggle_relationshiplines(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.toggle_relationshiplines', 'Toggle Relationship Lines', {'REGISTER'}, 'Show or hide all dotted child-parent relationship lines in the 3D viewport overlay.'
    def execute(self, context):
        for s in (s for a in context.screen.areas for s in a.spaces if s.type == 'VIEW_3D' and hasattr(s, 'overlay')): s.overlay.show_relationship_lines ^= 1
        return {'FINISHED'}
class LEPB_OT_add_terrain_materials(Operator, ImportHelper):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.add_terrain_materials', 'Add Terrain Materials', {'REGISTER', 'UNDO'}, 'Add multiple terrain tile images as materials to selected terrain objects.'
    files: CollectionProperty(name='Files', type=OperatorFileListElement)
    directory: StringProperty(name='Directory', subtype='DIR_PATH')
    filter_glob: StringProperty(name='Select', description=f'Select image files from the "3dobjects" asset sub-directory named tileN + EXT where N is a number from 0 to 255 and EXT is in {imgexts}.', default=';'.join('*'+e for e in imgexts))
    poll = classmethod(lambda _,c: next((o for o in c.selected_objects if o.type == 'MESH' and o.name.startswith('terrain')), None))
    def execute(s, c):
        ad = get_assetdirs(c, s.directory)
        l = [mtl_terrain(tilenum_fn(f.name), ad) for f in s.files]
        any(o.data.materials.append(m) for o in c.selected_objects for m in l if o.type == 'MESH' and o.name.startswith('terrain') and not o.material_slots.get(m.name))
        return {'FINISHED'}
class LEPB_OT_refresh_evdata(Operator):
    bl_idname, bl_label, bl_options = 'lepb.refresh_evdata', 'Refresh Event Data', {'INTERNAL'}
    def invoke(self, context, event): return ({'FINISHED'}, copy_evdata(event))[0]
def tmr_refresh():
    if l := [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D' and len(a.spaces) and a.spaces[0].show_region_ui]:
        bpy.ops.lepb.refresh_evdata('INVOKE_DEFAULT')
        for a in l: a.tag_redraw()
    return 0.1
def in_rect(r, p): return (x,y) if 0 <= (x := p[0] - r.x) <= r.width and 0 <= (y := p[1] - r.y) <= r.height else None
def allsane(a): return np.all(np.abs(a) < 1e6)
class LEPB_PT_addon(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type = 'LEPB_PT_addon', 'lepb', 'lepb', 'VIEW_3D', 'UI'
    def draw(s, c): pass
class LEPB_PT_import(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id, bl_options = 'LEPB_PT_import', 'Import', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon', {'DEFAULT_CLOSED'}
    def draw(s, c):
        r = (l := s.layout).row()
        for k in 'e3d 2d0 elm cmf csf caf'.split(): r.operator('lepb.import_'+k, text=k.upper())
        l.prop(o := get_opts(c), 'asset_path', text='Assets')
        l.prop(o, 'extra_path', text='Extra')
        l.prop(o, 'chunk_size')
        l.prop(o, 'use_dz')
        t = c.scene.lepb_uidata.get(w := 'dirnums') or {}
        h, b = l.panel('LEPB_PT_dirnums', default_closed=1)
        h.label(text='Abbreviated Directories')
        if b: any(b.label(text=f'{v} {k}') for k,v in t.items())
class LEPB_PT_export(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id, bl_options = 'LEPB_PT_export', 'Export', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon', {'DEFAULT_CLOSED'}
    def draw(s, c):
        r = s.layout.row()
        for k in 'e3d 2d0 elm cmf csf caf'.split(): r.operator('lepb.export_'+k, text=k.upper())
class LEPB_PT_maps(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id, bl_options = 'LEPB_PT_maps', 'Maps', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon', {'DEFAULT_CLOSED'}
    def draw(self, c):
        (l := self.layout).operator(LEPB_OT_load_chunks.bl_idname, text='Load Chunk')
        l.operator(LEPB_OT_add_terrain_materials.bl_idname, text='Add Terrain Materials')
        l.operator(LEPB_OT_toggle_relationshiplines.bl_idname, text='Toggle Relationship Lines')
        l.label(text='Height Map Tiles:', icon='MESH_GRID')
        (r := l.row()).operator(LEPB_OT_show_height.bl_idname, text='Show')
        r.operator(LEPB_OT_hide_height.bl_idname, text='Hide')
        l.prop(opts := get_opts(c), 'opacity')
        if o := find_map(c): (r := l.row()).prop(o, '["inside"]', text='Inside'), r.prop(o, '["ambient"]', text='Ambient')
        l.prop(opts, 'minute')
        l.operator(LEPB_OT_update_lighting.bl_idname, text='Update Lighting')
        m, w = get_mouse(), None
        for g in (g for a in c.screen.areas for g in a.regions if a.type == 'VIEW_3D' and g.type == 'WINDOW' and (v := in_rect(g, m))):
            if (p := intersect_line_plane(o := region_2d_to_origin_3d(g, g.data, v), o + 1e5*region_2d_to_vector_3d(g, g.data, v), (0,0,0), (0,0,1))) and allsane(p.xy): w = np.floor(2*p.xy).astype('i4').tolist()
        l.label(text=f'Mouse {w}')
        l.label(text=f'Object {np.floor(2*o.location.xy).astype("i4").tolist() if (o := c.object) else None}')
class LEPB_PT_elm(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id, bl_options = 'LEPB_PT_elm', 'ELM Data', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon', {'DEFAULT_CLOSED'}
    poll = classmethod(lambda _,c: c.object)
    def draw(self, c):
        (l := self.layout).prop(p := c.object.lepb_props, 'group', text='')
        if (g := p.group) in ('meshes', 'decals', 'particles'): l.prop(p, 'apath', text='')
        if g == 'meshes': (r := l.row()).prop(p, 'ordinal', text='ID'), r.prop(p, 'blend', text='Blend'), (r := l.row()).prop(p, 'emit_light', text='Emit'), r.prop(p, 'emit_color', text='')
        if g == 'decals': (r := l.row()).prop(p, 'dz_added', text='Offset Z'), r.prop(p, 'dz', text='')
def update_model_scale(s, c):
    i, v = s.id, (s.scale if s.use_scale else 1.0,)*3
    for o in bpy.data.objects:
        if o.get('cmodel_id') == i and o.type == 'ARMATURE': o.scale = v
def update_model_mesh_scale(s, c):
    i, v = s.id, (s.mesh_scale if s.use_mesh_scale else 1.0,)*3
    for o in bpy.data.objects:
        if o.get('cmodel_id') == i and o.type == 'MESH': o.scale = v
def update_model_bone_scale(s, c):
    i, v = s.id, (n := (s.bone_scale if s.use_bone_scale else 1.0)) / s.cur_bone_scale
    s.cur_bone_scale = n
    if not (l := [o for o in bpy.data.objects if o.get('cmodel_id') == i and o.type == 'ARMATURE']): return
    old_active = c.active_object
    for o in l:
        was_selected = o.select_get()
        select_active(c, o)
        bpy.ops.object.mode_set(mode='EDIT')
        for e in o.data.edit_bones:
            e.head *= v
            e.tail *= v
        bpy.ops.object.mode_set()
        if not was_selected: o.select_set(0)
    if old_active: select_active(c, o)
def setup_model(c, mdlset, cmodel, m):
    a, m.cur_bone_scale = mdlset.get_xmlroot(cmodel).actor, 1.0
    if (f := (next((float(t.text) for w in ('scale', 'actor_scale') if (t := a.find_tag(w))), 1.0))) != 1.0: m.scale, m.use_scale = f, 1
    if (t := a.find_tag('mesh_scale')) and (f := float(t.text)) != 1.0: m.mesh_scale, m.use_mesh_scale = f, 1
    if (t := a.find_tag('bone_scale')) and (f := float(t.text)) != 1.0: m.bone_scale, m.use_bone_scale = f, 1
class LEPB_model(PropertyGroup):
    id: IntProperty()
    name: StringProperty()
    selected: BoolProperty()
    use_scale: BoolProperty(name='Enable Model Scale', update=update_model_scale)
    scale: FloatProperty(name='Scale', default=1.0, min=0.01, max=100, step=0.05, update=update_model_scale, description='Scale applied to all rigs imported with this model.')
    use_mesh_scale: BoolProperty(name='Enable Mesh Scale', update=update_model_mesh_scale)
    mesh_scale: FloatProperty(name='Mesh Scale', default=1.0, min=0.01, max=100, step=0.05, update=update_model_mesh_scale, description='Scale applied to all mesh objects imported with this model.')
    use_bone_scale: BoolProperty(name='Enable Bone Scale', update=update_model_bone_scale)
    bone_scale: FloatProperty(name='Bone Scale', default=1.0, min=0.01, max=100, step=0.05, update=update_model_bone_scale, description='Multiplies all bone positions for all rigs imported with this model.')
    cur_bone_scale: FloatProperty(default=1.0)
class LEPB_uidata(PropertyGroup):
    opts: PointerProperty(type=LEPB_opts)
    models: CollectionProperty(type=LEPB_model)
    cur_model: IntProperty()
def get_mdlset(c): return m if (m := (s := get_state()).mdlset) else (setattr(s, 'mdlset', m := lepb.asset.Mdlset(get_assetdirs(c))), m)[1]
class LEPB_OT_refresh_models(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.refresh_models', 'Refresh Models', {'REGISTER'}, 'Scan asset paths for actor xml files and load model names and IDs.'
    def execute(s, c):
        assert (ad := get_assetdirs(c)), 'Missing asset path in addon options or preferences'
        (d := (u := c.scene.lepb_uidata).models).clear()
        u.cur_model, get_state().mdlset = 0, (ms := lepb.asset.Mdlset(ad))
        for m in sorted(ms.cmodels.values(), key=lambda m:m.id): (i := d.add()).id, i.name = m.id, m.name
        return {'FINISHED'}
class LEPB_OT_import_model(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.import_model', 'Import Model', {'REGISTER', 'UNDO'}, 'Import Cal3D models from definitions in actor_defs xml files.'
    poll = classmethod(lambda _,c: any(m.selected for m in c.scene.lepb_uidata.models))
    def execute(s, c):
        (ms := get_mdlset(c)).assetdirs, l, a, u = get_assetdirs(c), c.scene.cursor.location.copy(), [], c.scene.lepb_uidata
        select_none(c)
        for i,m in enumerate(u.models):
            if not m.selected: continue
            r, objs = import_cmodel(c, ms, cm := ms.cmodels[m.id])
            r.location, m.selected = l, 0
            setup_model(c, ms, cm, m)
            l.x += max(1, int(0.5 * r.dimensions.x))
            a.append((r, i))
        for r,i in a: r.select_set(1)
        select_active(c, a[-1][0])
        u.cur_model = a[-1][1]
        return {'FINISHED'}
class LEPB_UL_models(UIList):
    def draw_item(self, context, layout, data, m, icon, active_data, active_propname, index, flt_flag):
        if self.layout_type == 'GRID': return layout.prop(m, 'selected', text=f'{m.id} {m.name}', toggle=1, translate=0)
        (s := layout.split(factor=0.1)).prop(m, 'selected', text='')
        s.label(text=f'{m.id} {m.name}', translate=0)
class LEPB_PT_models(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id = 'LEPB_PT_models', 'Models', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon'
    def draw(self, c):
        (r := (l := self.layout).row()).operator(LEPB_OT_refresh_models.bl_idname, text='Refresh')
        r.label(text=str(len((u := c.scene.lepb_uidata).models) or ''))
        l.template_list('LEPB_UL_models', '', u := c.scene.lepb_uidata, 'models', u, 'cur_model', item_dyntip_propname='name')
        (r := l.row()).operator(LEPB_OT_import_model.bl_idname, text='Import')
        r.label(text=str(sum(m.selected for m in u.models)))
        ms, i, n = get_mdlset(c), u.cur_model, len(u.models)
        if ms and 0 <= i < n and (cm := ms.cmodels.get((m := u.models[i]).id)):
            l.label(text=cm.tree.path, translate=0)
            if cm.tree.root:
                for k in 'scale mesh_scale bone_scale'.split():
                    (r := l.row()).prop(m, w := 'use_'+k, text='')
                    (s := r.row()).enabled = getattr(m, w)
                    s.prop(m, k)
def find_cmodel_rig(o): return o if o and o.get('cmodel_state') else o.parent if o and o.parent and o.parent.get('cmodel_state') else None
class LEPB_OT_swap_piece(Operator):
    bl_property, bl_idname, bl_label, bl_options, bl_description = 'piece_id', 'lepb.swap_piece', 'Swap Model Piece', {'REGISTER', 'UNDO'}, 'Change a body part or equipment for an imported Cal3D model.'
    piece_kind: StringProperty()
    cmodel_id: IntProperty()
    rig_name: StringProperty()
    piece_id: EnumProperty(items=lambda s,c:[(str(p.id), p.fmt(), p.desc, p.id) for p in sorted(d.values(), key=lambda t:t.id)] if (cm := get_mdlset(c).cmodels.get(s.cmodel_id)) and (d := cm.pieces.get(s.piece_kind)) else [])
    poll = classmethod(lambda _,c: find_cmodel_rig(c.object))
    def execute(s, c): return ({'FINISHED'}, swap_piece(c, bpy.data.objects[s.rig_name], s.piece_kind, int(s.piece_id)))[0]
    def invoke(s, c, e): return ({'FINISHED'}, c.window_manager.invoke_search_popup(s))[0]
class LEPB_OT_stitch_skins(Operator):
    bl_idname, bl_label, bl_options, bl_description = 'lepb.stitch_skins', 'Stitch Skins', {'REGISTER', 'UNDO'}, 'Stitch together all of the active model\'s piece skins into one image.'
    poll = classmethod(lambda _,c: find_cmodel_rig(c.object))
    def execute(s, c):
        ms, cms, i = get_mdlset(c), (r := find_cmodel_rig(c.object))['cmodel_state'], r['cmodel_id']
        img = img_stitch(ms.cmodels[i], cms, get_assetdirs(c))
        s.report({'INFO'}, f'Updated {img.name} from skins in {r.name}')
        return {'FINISHED'}
class LEPB_PT_pieces(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id = 'LEPB_PT_pieces', 'Model Pieces', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon'
    def draw(self, c):
        if not (rig := find_cmodel_rig(c.object)): return self.layout.label(text='no model rig or pieces selected')
        l, ms, i, cms = self.layout, get_mdlset(c), rig['cmodel_id'], rig['cmodel_state']
        if not (p := ms.cmodels[i].pieces) or len(p) < 2: return l.label(text='no pieces')
        l.operator(LEPB_OT_stitch_skins.bl_idname)
        (col := (row := l.row()).column(align=1)).alignment = 'LEFT'
        for k,v in (kv := [t for t in cms.items() if t[0] != 'body']): col.label(text=k)
        col = row.column(align=1)
        for k,v in kv: (o := col.operator(LEPB_OT_swap_piece.bl_idname, text=p[k][v].fmt(), translate=0)).piece_kind, o.cmodel_id, o.rig_name, o.piece_id = k, i, rig.name, str(v)
class LEPB_MT_add(Menu):
    bl_label, bl_idname = 'lepb', 'LEPB_MT_add'
    def draw(self, context):
        (o := self.layout.operator)(LEPB_OT_import_e3d.bl_idname, text='Mesh 3d-object (e3d)', icon='MESH_CUBE')
        o(LEPB_OT_import_2d0.bl_idname, text='Decal 2d-object (2d0)', icon='MESH_PLANE')
        o(LEPB_OT_import_elm.bl_idname, text='Import Map (elm)', icon='IMPORT')
        o(LEPB_OT_new_map.bl_idname, text='New Empty Map', icon='MESH_GRID')
        o(LEPB_OT_import_cmf.bl_idname, text='Import Cal3D Mesh (cmf)', icon='MESH_CUBE')
        o(LEPB_OT_import_csf.bl_idname, text='Import Cal3D Skeleton (csf)', icon='ARMATURE_DATA')
        o(LEPB_OT_import_caf.bl_idname, text='Import Cal3D Animation (caf)', icon='ACTION')
class FileHandlerWithPoll(FileHandler): poll = classmethod(lambda _c,: poll_file_object_drop(c))
class LEPB_FH_e3d(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_e3d', 'E3D Mesh', 'lepb.import_e3d', 'lepb.export_e3d', '.e3d'
class LEPB_FH_2d0(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_2d0', '2D0 Decal', 'lepb.import_2d0', 'lepb.export_2d0', '.2d0'
class LEPB_FH_elm(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_elm', 'ELM Map', 'lepb.import_elm', 'lepb.export_elm', '.elm;.elm.gz'
class LEPB_FH_cmf(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_cmf', 'Cal3D Mesh', 'lepb.import_cmf', 'lepb.export_cmf', '.cmf'
class LEPB_FH_csf(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_csf', 'Cal3D Skeleton', 'lepb.import_csf', 'lepb.export_csf', '.csf'
class LEPB_FH_caf(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_caf', 'Cal3D Animation', 'lepb.import_caf', 'lepb.export_caf', '.caf'
def mfadd(s, c): s.layout.menu(LEPB_MT_add.bl_idname)
def mfimp_e3d(s, c): s.layout.operator(LEPB_OT_import_e3d.bl_idname, text='LE mesh 3d-object (.e3d)')
def mfexp_e3d(s, c): s.layout.operator(LEPB_OT_export_e3d.bl_idname, text='LE mesh 3d-object (.e3d)')
def mfimp_2d0(s, c): s.layout.operator(LEPB_OT_import_2d0.bl_idname, text='LE decal 2d-object (.2d0)')
def mfexp_2d0(s, c): s.layout.operator(LEPB_OT_export_2d0.bl_idname, text='LE decal 2d-object (.2d0)')
def mfimp_elm(s, c): s.layout.operator(LEPB_OT_import_elm.bl_idname, text='LE map (.elm)')
def mfexp_elm(s, c): s.layout.operator(LEPB_OT_export_elm.bl_idname, text='LE map (.elm)')
def mfimp_csf(s, c): s.layout.operator(LEPB_OT_import_csf.bl_idname, text='Cal3D Skeleton (.csf)')
def mfexp_csf(s, c): s.layout.operator(LEPB_OT_export_csf.bl_idname, text='Cal3D Skeleton (.csf)')
def mfimp_cmf(s, c): s.layout.operator(LEPB_OT_import_cmf.bl_idname, text='Cal3D Mesh (.cmf)')
def mfexp_cmf(s, c): s.layout.operator(LEPB_OT_export_cmf.bl_idname, text='Cal3D Mesh (.cmf)')
def mfimp_caf(s, c): s.layout.operator(LEPB_OT_import_caf.bl_idname, text='Cal3D Animation (.caf)')
def mfexp_caf(s, c): s.layout.operator(LEPB_OT_export_caf.bl_idname, text='Cal3D Animation (.caf)')
classes = [v for k,v in locals().items() if k.startswith('LEPB_')]
mfimps = [v for k,v in locals().items() if k.startswith('mfimp_')]
mfexps = [v for k,v in locals().items() if k.startswith('mfexp_')]
def register():
    for c in classes: bpy.utils.register_class(c)
    bpy.types.Scene.lepb_uidata = PointerProperty(type=LEPB_uidata)
    bpy.types.Object.lepb_props = PointerProperty(type=LEPB_props)
    bpy.types.VIEW3D_MT_add.prepend(mfadd)
    for f in mfimps: bpy.types.TOPBAR_MT_file_import.append(f)
    for f in mfexps: bpy.types.TOPBAR_MT_file_export.append(f)
    bpy.app.timers.register(tmr_refresh, persistent=1)
    install_handler('depsgraph_update_post', hdl_deps)
def unregister():
    remove_handler('depsgraph_update_post', hdl_deps)
    if bpy.app.timers.is_registered(tmr_refresh): bpy.app.timers.unregister(tmr_refresh)
    for f in mfexps: bpy.types.TOPBAR_MT_file_export.remove(f)
    for f in mfimps: bpy.types.TOPBAR_MT_file_import.remove(f)
    bpy.types.VIEW3D_MT_add.remove(mfadd)
    del bpy.types.Object.lepb_props
    del bpy.types.Scene.lepb_uidata
    for c in classes: bpy.utils.unregister_class(c)
