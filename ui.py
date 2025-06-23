import bpy, bmesh, numpy as np
from os.path import exists, split as psplit, join as pjoin, relpath, abspath
from bpy.path import abspath as abspath_bl
from pathlib import Path
from .asset import E3d, Decal, Elm, Geometry
from .yuck import nodedims
class Graph:
    spacing = 50, 25
    def __init__(g, nodable, pre='ShaderNode', clear=True):
        if hasattr(nodable, 'use_nodes'): nodable.use_nodes = 1
        t = getattr(nodable, 'node_tree', 0) or nodable
        g.pre, g.n, g.l, g.dims = pre, t.nodes, t.links, nodedims[pre]
        if clear: g.n.clear()
    def node(g, what, where='right', of=-2, defvals={}, pre=None, **attrs):
        r = g.n.new((g.pre if pre is None else pre) + what)
        for k,v in attrs.items(): setattr(r, k, v)
        for k,v in defvals.items(): r.inputs[k].default_value = v
        if len(g.n) < 2: return r
        v, s, pd, rd = (p := g.n[of]).location.copy(), g.spacing, g.dims[p.type], g.dims[r.type]
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
def assetdirs(c, p): return ([Path(ea)] if (ea := get_opts(c).extra_assets) else []) + [d for d in (Path(p).absolute()).parents if (d/'files.lst').exists()]
def find_asset(ad, subdir, b, exts=None):
    p = next((p for d in ad for e in ([''] if exts is None else exts) if (p := d/subdir/(b+e)).exists()), None)
    assert p, f'asset {b} {exts} not found in {ad} {subdir}'
    return str(p)
def opacity_controller(g):
    g.node('BsdfTransparent', 'below')
    f = g.node('MixShader', of=-3).inputs[0].driver_add('default_value')
    (v := f.driver.variables.new()).type, v.name = 'SINGLE_PROP', 'opacity'
    (t := v.targets[0]).id_type, t.id, t.data_path = 'SCENE', bpy.context.scene, 'lepb_opts.opacity'
    f.driver.expression = f'0.01*{v.name}'
    g.edge(0, 2, -3)
    g.edge(0, 1)
def mtl_e3dpart(part, ad):
    imgfile = part.tex.decode()
    if m := bpy.data.materials.get(mtlname := f'e3d-mtl-{imgfile}'): return m
    g = Graph(m := bpy.data.materials.new(mtlname))
    if not (i := bpy.data.images.get(imgname := f'e3d-img-{imgfile}')): (i := bpy.data.images.load(find_asset(ad, '3dobjects', imgfile))).name = imgname
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
def new_bpy(name, data, bdc='objects', cprops={}, lprops={}, **attrs):
    r = getattr(bpy.data, bdc).new(name, data)
    for k,v in attrs.items(): setattr(r, k, v)
    for k,v in cprops.items(): r[k] = v
    for k,v in lprops.items(): setattr(r.lepb_props, k, v)
    return r
def select_none(c): any(o.select_set(0) for o in c.view_layer.objects)
def select_active(c, o):
    o.select_set(1)
    c.view_layer.objects.active = o
def select_only(c, o):
    select_none(c)
    select_active(c, o)
def linkcl(c, o, collection=None): (collection or (c.layer_collection or c.view_layer.active_layer_collection).collection).objects.link(o)
def relapath(ad, p): return next(relpath(p, d) for d in ad if p.is_relative_to(d))
def import_e3d(context, filepath):
    ad = assetdirs(context, p := Path(abspath_bl(filepath)).absolute())
    m = msh_e3d(ad, str(p), p.name)
    linkcl(context, o := new_bpy(m.name, m, rotation_mode='YXZ', lprops=dict(group='meshes', apath=relapath(ad, p))))
    return o
def msh_e3d(ad, path, name):
    if m := bpy.data.meshes.get(name): return m
    if not path: return None
    e, m, opts = E3d(path), bpy.data.meshes.new(name), get_opts()
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
def mtlinfo(m, edir):
    assert m.use_nodes, f'expecting node tree for {m.name}'
    img, seethru = None, 0
    for n in m.node_tree.nodes:
        if n.type == 'TEX_IMAGE': img = n.image
        elif n.type == 'BSDF_TRANSPARENT': seethru = 1
    assert img, f'expecting a texture image in {m.name}'
    d, f = psplit(abspath_bl(img.filepath, library=img.library))
    return {'imgfile':f, 'imgdir':d, 'seethru':img.channels == 4 and seethru}
def write_with_backup(p, d):
    if exists(p) and (b := next((b for i in range(1000) if not exists(b := f'{p}.{i:03d}.bak')), None)):
        with open(b, 'wb') as o, open(p, 'rb') as i: o.write(i.read())
    with open(p, 'wb') as f: f.write(d)
def triangulate(m): return (m, (b := bmesh.new()).from_mesh(m), bmesh.ops.triangulate(b, faces=b.faces[:]), b.to_mesh(m), b.free())[0]
def export_e3d(c, filepath):
    m, g = triangulate((o := c.object.evaluated_get(c.evaluated_depsgraph_get())).to_mesh()), Geometry()
    m.calc_loop_triangles(), m.calc_tangents()
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
    write_with_backup(p, E3d(p, geom=g).encode())
    o.to_mesh_clear()
def mtl_decal(d, ad):
    if m := bpy.data.materials.get(n := f'decal-mtl-{d.imgfile}'): return m
    g = Graph(m := bpy.data.materials.new(n))
    if not (i := bpy.data.images.get(imgname := f'decal-img-{d.imgfile}')): (i := bpy.data.images.load(find_asset(ad, '2dobjects/ground', d.imgfile))).name = imgname
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
    if m := bpy.data.meshes.get(name): return m
    if not path: return None
    d, m = Decal(path), bpy.data.meshes.new(name)
    x, y = (0.5*v for v in d.objsz)
    m.from_pydata([(-x,-y,0),(x,-y,0),(x,y,0),(-x,y,0)],[],[(0,1,2,3)])
    m.materials.append(mtl_decal(d, ad))
    m.uv_layers.new(do_init=0).data.foreach_set('uv', flat((np.array(d.imgbox).reshape(-1,2) / d.imgdiv).reshape(-1)[[0,1,2,1,2,3,0,3]], 'f4'))
    return m
def import_2d0(context, filepath):
    ad = assetdirs(context, p := Path(abspath_bl(filepath)).absolute())
    m = msh_decal(ad, str(p), p.name)
    linkcl(context, o := new_bpy(m.name, m, lprops=dict(group='decals', apath=relapath(ad, p))))
    return o
def export_2d0(context, filepath):
    m, d = (o := context.object).data, Decal()
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
    __slots__ = 'loaders evdata'.split()
    def __init__(s): s.loaders, s.evdata = {}, {}
def get_state():
    if not (r := getattr(s := bpy.types.Scene, k := 'lepb', None)): setattr(s, k, r := State())
    return r
k_mouse = 'mouse_x mouse_y'.split()
def copy_evdata(e): get_state().evdata.update((k, getattr(e, k)) for k in k_mouse)
def get_mouse(): return [d[k] for k in k_mouse] if (d := get_state().evdata) else [0,0]
def mksuf(l, s): return l.suffix if s is None else s
class Loader:
    __slots__ = 'e ad curcl maproot mrn tag suffix msgs'.split()
    def __init__(l, context, e, tag):
        l.e, l.ad, l.curcl, l.maproot, l.mrn, l.tag, l.suffix, l.msgs = e, assetdirs(context, e.path), context.scene.collection, None, None, tag, '-'+tag, []
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
imgexts = 'dds png jpg bmp'.split()
def mtl_terrain(b, ad):
    if m := bpy.data.materials.get(n := f'terrain-tile{b}'): return m
    g = Graph(m := bpy.data.materials.new(n))
    m.preview_render_type = 'FLAT'
    if b == 255: g.node('BsdfTransparent')
    else:
        if not (i := bpy.data.images.get(n)): (i := bpy.data.images.load(find_asset(ad, '3dobjects', f'tile{b}.', imgexts))).name = n
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
def tilenum_fn(p, default=None): return int(v) if (f := Path(p).name).startswith('tile') and (d := f.find('.', 4)) != -1 and (v := f[4:d]).isdigit() else default
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
def load_meshes(l, a, oi):
    pu, pi, du = *(u := np.unique(a.apath, return_inverse=1)), np.strings.decode(u[0])
    lp, ar, ah, ac = [assetpath(l, p) for p in du], np.radians(a.rot), a.blend == 20, np.where((ec := a.scale) < 0.0625, 1, ec).repeat(3).reshape(-1,3)
    for i,m in enumerate(a):
        p, n, h, o = *lp[pi[i]], ah[i]*'-placeholder', oi[i].item()
        l.obj(f'mesh{o}{h}-{n}', None if h else msh_e3d(l.ad, p, n), parent=l.maproot, location=m.pos, rotation_euler=ar[i], rotation_mode='YXZ', scale=ac[i], lprops=dict(group='meshes', ordinal=o, apath=du[pi[i]], blend=m.blend, emit_light=bool(m.emit), emit_color=m.color))
def load_decals(l, a, oi):
    pu, pi, du = *(u := np.unique(a.apath, return_inverse=1)), np.strings.decode(u[0])
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
    dp = np.strings.decode(a.apath)
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
    if not (l := get_state().loaders.get(r['tag'])): init_map(l := Loader(c, Elm(p := r['epath'], initxy=r.get('initxy') if not exists(p) else None), r['tag']), r['tps'])
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
    init_map(l := Loader(c, Elm(p), mktag(c, p)), (opts := get_opts(c)).chunk_size)
    init_chunks(l, opts)
    return l
def obj_elm(c, x, y):
    assert (d := (opts := get_opts(c)).asset_path or get_prefs(c).default_asset_path), 'asset path must be set in lepb panel or addon prefs'
    init_map(l := Loader(c, Elm(path=pjoin(d, t := mktag(c)), initxy=(x, y)), t), opts.chunk_size)
    l.maproot['initxy'] = x, y
    init_chunks(l, opts)
    return l.maproot
def is_map(o): return o.name.startswith('map') and o.get('tag') is not None
def find_map(c):
    if (o := c.object) and is_map(o): return o
    if o and (p := o.parent) and is_map(p): return p
    if len(l := list(get_state().loaders.values())) == 1 and (r := bpy.data.objects.get(l[0].mrn)): return r
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
    assert hs == h.shape, f'height map shape {h.shape} must match terrain, expecting {hs}'
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
from bpy.types import AddonPreferences, FileHandler, Menu, Operator, OperatorFileListElement, Panel, PropertyGroup
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper, poll_file_object_drop
from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
from mathutils.geometry import intersect_line_plane
from bl_math import lerp
class LEPB_prefs(AddonPreferences):
    bl_idname = __package__
    default_asset_path: StringProperty(name='Default Asset Path', description='Game asset directory to use if none is set in the addon panel.', default='', maxlen=2048, subtype='DIR_PATH')
    def draw(self, context): self.layout.prop(self, 'default_asset_path', text='Default Asset Path')
def get_prefs(c=None): return (c or bpy.context).preferences.addons[__package__].preferences
class MeshImportOpts:
    sharp_angle: IntProperty(name='Edge Sharp Angle', description='Mark edges "sharp" based on the angle in degrees between adjacent faces. -1 disables.', default=45, min=-1, max=180)
    shade_smooth: BoolProperty(name='Shade Smooth', description='Set "shade smooth" on imported mesh objects.', default=1)
class MapImportOpts:
    chunk_size: IntProperty(name='Chunk Size', description='Number of terrain tiles per chunk side. Each terrain is 3x3 world units (meters), or 6x6 height tiles (walkable positions in game). Only applied when creating or importing a map.', default=8, min=1, max=1024)
    chunks_to_load: StringProperty(name='Chunks to load', description='Semi-colon separated list of i,j row-column chunk coordinates to load automatically after map import.', default='0,0')
class LEPB_opts(PropertyGroup, MeshImportOpts, MapImportOpts):
    asset_path: StringProperty(name='Asset Path', description='The base directory used to find game data files such as images textures when loading assets that contain relative path references. Should contain the file "files.lst".', default='', maxlen=2048, subtype='DIR_PATH')
    extra_assets: StringProperty(name='Extra Asset Directory', description='Additional directory to search in first when loading referenced assets. For example downloaded client assets in the game config subdirectory "updates/1_9_5_0". Should contain the file "files.lst".', default='', maxlen=2048, subtype='DIR_PATH')
    minute: IntProperty(name='Game Minute', description='Time of day in game. Hours 0 to 3 are daylight, 3 to 6 are night.', default=60, min=0, max=359)
    use_dz: BoolProperty(name='Decal Z Offset', description='Mimick the way the game client offsets the z locations of decals to reduce z-fighting. Adjusting the near and far clip planes depending on the camera position may be more reliable. Only applied during load.', default=1)
    cur_tagnum: IntProperty(default=0, options={'HIDDEN'})
    opacity: IntProperty(name='Map Opacity', description='Modify the transparency of all terrain, mesh and decal map objects in the 3D viewport to make height map tiles more easily visible. Applied when viewport shading is material preview or rendered.', default=100, min=0, max=100, step=1, subtype='PERCENTAGE')
def get_opts(c=None): return (c or bpy.context).scene.lepb_opts
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
    dz: FloatProperty(name='dz', default=0, min=0, max=Elm.decal_dz[-1], step=Elm.ddz, precision=9)
    dz_added: BoolProperty(name='dz_added', description='Whether a decal 2D object z position was offset.', default=0)
class Importer(Operator, ImportHelper):
    files: CollectionProperty(name='Files', type=OperatorFileListElement)
    directory: StringProperty(name='Directory', subtype='DIR_PATH')
    parent_map: StringProperty(name='Parent to Map', description='The imported asset will be parented to the given map root object, so that it will be included in the ELM file when the map is exported.', default='', search=lambda _,c,text: (o.name for o in bpy.data.objects if o.get('tag') and text in o.name))
    poll = classmethod(lambda _,c: c.mode == 'OBJECT')
    def invoke(s, c, e):
        if not s.parent_map and (r := find_map(c)): s.parent_map = r.name
        return super().invoke(c, e)
    def execute(s, c):
        l, p, opts = c.scene.cursor.location.copy(), bpy.data.objects.get(s.parent_map), update_opts(c, s)
        select_none(c)
        for f in s.files:
            (o := s.do_import(c, pjoin(s.directory, f.name))).location, o.parent = l, p
            l.x += o.dimensions.x*1.1
            o.select_set(1)
        if s.files: select_active(c, o)
        return {'FINISHED'}
class LEPB_OT_import_e3d(Importer, MeshImportOpts):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.import_e3d', 'Import E3D Mesh', {'REGISTER','UNDO'}, 'Import a 3D object mesh asset.', '.e3d'
    filter_glob: StringProperty(default='*.e3d;*.e3d.gz', options={'HIDDEN'})
    def do_import(s, c, p): return import_e3d(c, p)
class LEPB_OT_export_e3d(Operator, ExportHelper):
    bl_idname, bl_label, bl_options, bl_description, filename_ext = 'lepb.export_e3d', 'Export E3D Mesh', {'PRESET'}, 'Export the active object as a 3D object mesh asset.', '.e3d'
    filter_glob: StringProperty(default='*.e3d', options={'HIDDEN'})
    poll = classmethod(lambda _,c: (o := c.object) and o.type == 'MESH' and c.mode == 'OBJECT')
    def execute(s, c): return ({'FINISHED'}, export_e3d(c, s.filepath))[0]
class LEPB_OT_import_2d0(Importer):
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
        icsz, d, suf = 1 / (tps * Elm.tsize), bpy.data.objects, get_loader(c, r).suffix
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
    filter_glob: StringProperty(name='Select', description=f'Select image files from the "3dobjects" asset sub-directory named "tileN.EXT" where N is a number from 0 to 255 and EXT is in {imgexts}.', default=';'.join('*.'+e for e in imgexts))
    poll = classmethod(lambda _,c: next((o for o in c.selected_objects if o.type == 'MESH' and o.name.startswith('terrain')), None))
    def execute(s, c):
        ad = assetdirs(c, s.directory)
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
class LEPB_PT_import(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id = 'LEPB_PT_import', 'Import', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon'
    def draw(self, c):
        (r := (l := self.layout).row()).operator(LEPB_OT_import_e3d.bl_idname, text='E3D')
        r.operator(LEPB_OT_import_2d0.bl_idname, text='2D0')
        r.operator(LEPB_OT_import_elm.bl_idname, text='ELM')
        l.label(text='Game Asset Directory:')
        l.prop(o := get_opts(c), 'asset_path', text='')
        l.label(text='Extra Asset Directory:')
        l.prop(o, 'extra_assets', text='')
        l.prop(o, 'chunk_size')
        l.prop(o, 'use_dz')
class LEPB_PT_export(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id = 'LEPB_PT_export', 'Export', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon'
    def draw(self, c):
        (r := (l := self.layout).row()).operator(LEPB_OT_export_e3d.bl_idname, text='E3D')
        r.operator(LEPB_OT_export_2d0.bl_idname, text='2D0')
        r.operator(LEPB_OT_export_elm.bl_idname, text='ELM')
class LEPB_PT_elm(Panel):
    bl_idname, bl_label, bl_category, bl_space_type, bl_region_type, bl_parent_id = 'LEPB_PT_elm', 'ELM Data', 'lepb', 'VIEW_3D', 'UI', 'LEPB_PT_addon'
    poll = classmethod(lambda _,c: c.object)
    def draw(self, c):
        (l := self.layout).prop(p := c.object.lepb_props, 'group', text='')
        if (g := p.group) in ('meshes', 'decals', 'particles'): l.prop(p, 'apath', text='')
        if g == 'meshes': (r := l.row()).prop(p, 'ordinal', text='ID'), r.prop(p, 'blend', text='Blend'), (r := l.row()).prop(p, 'emit_light', text='Emit'), r.prop(p, 'emit_color', text='')
        if g == 'decals': (r := l.row()).prop(p, 'dz_added', text='Offset Z'), r.prop(p, 'dz', text='')
class LEPB_MT_add(Menu):
    bl_label, bl_idname = 'lepb', 'LEPB_MT_add'
    def draw(self, context):
        (l := self.layout).operator(LEPB_OT_import_e3d.bl_idname, text='Mesh 3d-object (e3d)', icon='MESH_CUBE')
        l.operator(LEPB_OT_import_2d0.bl_idname, text='Decal 2d-object (2d0)', icon='MESH_PLANE')
        l.operator(LEPB_OT_import_elm.bl_idname, text='Import Map (elm)', icon='IMPORT')
        l.operator(LEPB_OT_new_map.bl_idname, text='New Empty Map', icon='MESH_GRID')
class FileHandlerWithPoll(FileHandler): poll = classmethod(lambda _c,: poll_file_object_drop(c))
class LEPB_FH_e3d(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_e3d', 'E3D Mesh', 'lepb.import_e3d', 'lepb.export_e3d', '.e3d'
class LEPB_FH_2d0(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_2d0', '2D0 Decal', 'lepb.import_2d0', 'lepb.export_2d0', '.2d0'
class LEPB_FH_elm(FileHandlerWithPoll): bl_idname, bl_label, bl_import_operator, bl_export_operator, bl_file_extensions = 'LEPB_FH_elm', 'ELM Map', 'lepb.import_elm', 'lepb.export_elm', '.elm;.elm.gz'
def mfadd(s, c): s.layout.menu(LEPB_MT_add.bl_idname)
def mfimp_e3d(s, c): s.layout.operator(LEPB_OT_import_e3d.bl_idname, text='LE mesh 3d-object (.e3d)')
def mfexp_e3d(s, c): s.layout.operator(LEPB_OT_export_e3d.bl_idname, text='LE mesh 3d-object (.e3d)')
def mfimp_2d0(s, c): s.layout.operator(LEPB_OT_import_2d0.bl_idname, text='LE decal 2d-object (.2d0)')
def mfexp_2d0(s, c): s.layout.operator(LEPB_OT_export_2d0.bl_idname, text='LE decal 2d-object (.2d0)')
def mfimp_elm(s, c): s.layout.operator(LEPB_OT_import_elm.bl_idname, text='LE map (.elm)')
def mfexp_elm(s, c): s.layout.operator(LEPB_OT_export_elm.bl_idname, text='LE map (.elm)')
classes = [v for k,v in locals().items() if k.startswith('LEPB_')]
mfimps = [v for k,v in locals().items() if k.startswith('mfimp_')]
mfexps = [v for k,v in locals().items() if k.startswith('mfexp_')]
def register():
    for c in classes: bpy.utils.register_class(c)
    bpy.types.Scene.lepb_opts = PointerProperty(type=LEPB_opts)
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
    del bpy.types.Scene.lepb_opts
    for c in classes: bpy.utils.unregister_class(c)
