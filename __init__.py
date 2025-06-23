bl_info = {'name':'lepb','author':'vancolbert','version':(1,0,0),'blender':(4,0,0),'location':'File > Import-Export','description':'Import and export Landes Eternelles game assets','tracker_url':'https://github.com/vancolbert/lepb/issues/','category':'Import-Export'}
if 'bpy' in locals():
    import sys, importlib
    for n in (m := sys.modules.copy()):
        if n.startswith('lepb'): print('Reloading:', importlib.reload(m[n]))
try: import bpy
except ImportError: bpy = None
from . import asset, yuck
if bpy: from . import ui
def register(): ui.register()
def unregister(): ui.unregister()
