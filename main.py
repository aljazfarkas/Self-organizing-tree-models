import bpy
import bmesh
from mathutils import *
from math import cos, sin, radians, degrees, isclose

SCALE_RATIO = 9/10
ENERGY_RATIO = 2/3
SIDE_NODE_ANGLE = 30
BASE_ROTATION = 90
SOURCE_ENERGY = 40

COUNTER = 0

class Internode:
    def __init__(self, cyl):
        self.id = 0
        self.cyl = cyl
        self.next_internode = None
        self.side_internode = None
        self.parent_node = None
        self.base_rotation = 0
        self.energy = 0

#spremeni pivot za rotacijo in lokacijo objekta
def set_origin(ob, global_origin=Vector()):
    mw = ob.matrix_world
    o = mw.inverted() @ Vector(global_origin)
    ob.data.transform(Matrix.Translation(-o))
    mw.translation = global_origin
    
def growNode(parent, is_side_node=False):
    global COUNTER 
    if(parent.energy > 10):      
        bpy.ops.mesh.primitive_cylinder_add()
        cyl = bpy.context.object
        cyl.name = "Node"
        
        node = Internode(cyl)
        node.id = COUNTER
        COUNTER += 1
        node.energy = parent.energy * ENERGY_RATIO
        node.parent_node = parent

        if (is_side_node == False or node.id == 3):
            node.base_rotation = parent.base_rotation + BASE_ROTATION
        else: 
            node.base_rotation = parent.base_rotation
            
        #vnaprej določim angle za internode
        if(is_side_node == True):
            angle = radians(SIDE_NODE_ANGLE)
        else:
            angle = 0
            
        #dimensions
        cyl.dimensions = (parent.cyl.dimensions.x * SCALE_RATIO,
                            parent.cyl.dimensions.y * SCALE_RATIO,
                            parent.cyl.dimensions.z * SCALE_RATIO )
        bpy.ops.transform.translate()
        
        #coefficient je koeficient, ki pove, v kateri angle se bo nagibal internode
        if(angle == 0):
            coefficient = 0
        if(angle < 0):
            coefficient = -1
        if(angle > 0):
            coefficient = 1
        
        #spremenim pivot cilindra - pri rotation_euler.y = 0  in rotation_euler.x = 0 spodaj na sredini
        #pri rotation_euler.y > 0 in rotation_euler.x = 0 spodaj na levitilen
        #pri rotation_euler.y < 0 in rotation_euler.x = 0 spodaj na desni
        offset_x = coefficient * cyl.dimensions.x/2
        offset_z = cyl.dimensions.z/2
        set_origin(cyl, Vector((cyl.location.x - offset_x,
                                cyl.location.y, 
                                cyl.location.z - offset_z)))
         
        #location - offset_x in offset_y obratne kotne funkcije
        offset_x =  sin(parent.cyl.delta_rotation_euler.y) * parent.cyl.dimensions.z * cos(radians(node.base_rotation)) + \
                    cos(parent.cyl.delta_rotation_euler.y) * -parent.cyl.dimensions.x/2 * coefficient * cos(radians(node.base_rotation))
        
        offset_y = sin(parent.cyl.delta_rotation_euler.y) * parent.cyl.dimensions.z * sin(radians(node.base_rotation)) + \
                    cos(parent.cyl.delta_rotation_euler.y) * -parent.cyl.dimensions.x/2 * coefficient * sin(radians(node.base_rotation))   
        
        offset_z = cos(parent.cyl.delta_rotation_euler.y) * parent.cyl.dimensions.z + \
                    sin(parent.cyl.delta_rotation_euler.y) * parent.cyl.dimensions.x/2 * coefficient
        
        cyl.location = (parent.cyl.location.x + offset_x,
                        parent.cyl.location.y + offset_y,
                        parent.cyl.location.z + offset_z)
        bpy.ops.transform.translate()
        
        #rotation
        cyl.delta_rotation_euler = Euler((parent.cyl.delta_rotation_euler.x, 
                                    parent.cyl.delta_rotation_euler.y + angle,
                                    radians(node.base_rotation)),'XYZ')          
        bpy.ops.transform.translate()
        
        #pivot spremenim nazaj - na središče spodnje ploskve
        offset_x = cos(cyl.delta_rotation_euler.y) * cyl.dimensions.x/2 * coefficient * cos(radians(node.base_rotation))
        offset_y = cos(cyl.delta_rotation_euler.y) * cyl.dimensions.x/2 * coefficient * sin(radians(node.base_rotation))
        offset_z = sin(cyl.delta_rotation_euler.y) * -cyl.dimensions.x/2 * coefficient
        
        set_origin(cyl, Vector((cyl.location.x + offset_x,
                                cyl.location.y + offset_y, 
                                cyl.location.z + offset_z)))
        
        if(is_side_node == False):
            node.next_internode = growNode(node)
            node.side_internode = growNode(node, is_side_node=True)
        else:
            node.side_internode = growNode(node, is_side_node=True)
        
        print(f"{node.id} {cyl.location}")
#            if (node.id == 3):
#                node.side_internode = growNode(node, is_side_node=True)
    
        return node
    else:
        return None
     

print("Generating tree...")

bpy.ops.mesh.primitive_cylinder_add()
cyl = bpy.context.object
cyl.name = "Stem"


cyl.dimensions = (4,4,6)
bpy.ops.transform.translate()

cyl.location = (0, 0, cyl.dimensions.z/2)
bpy.ops.transform.translate()

cyl.delta_rotation_euler = Euler((0,0,radians(BASE_ROTATION)),'XYZ')
bpy.ops.transform.translate() 

#nastavim pivot
set_origin(cyl, Vector((cyl.location.x, cyl.location.y, cyl.location.z - cyl.dimensions.z/2)))

stem = Internode(cyl)
stem.id = COUNTER
COUNTER += 1
stem.energy = SOURCE_ENERGY
stem.base_rotation = BASE_ROTATION
#stem.next_internode = growNode(stem)
stem.side_internode = growNode(stem, is_side_node=True)

print("Generated tree")