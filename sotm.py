import cProfile

import math
import sys
import numpy
import scipy.spatial
import random
from mathutils import Vector, Matrix
import bpy
from timeit import default_timer as timer
from datetime import timedelta
import os
import tracemalloc

from bpy.props import IntProperty, FloatProperty, PointerProperty
from bpy.utils import register_class, unregister_class
from bpy.types import PropertyGroup, Operator, Panel


bl_info = {
    "name": "Generate tree",
    "blender": (3, 1, 0),
    "location": "View 3D > Add > Mesh > New Object",
    "description": "Generate a tree using self-organizing tree models algorithm",
    "category": "Add Mesh",
}


class GenerateTreeOperator(bpy.types.Operator):
    bl_idname = "object.generate_tree"
    bl_label = "Generate Tree"    
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        generateTree(
            scene.generate_tree.number_of_attraction_points,
            scene.generate_tree.attraction_points_width,
            scene.generate_tree.attraction_points_height,
            scene.generate_tree.internode_length,
            scene.generate_tree.minimal_internode_diameter,
            scene.generate_tree.internode_diameter_coefficient,
            scene.generate_tree.radius_of_influence,
            scene.generate_tree.kill_distance,
            scene.generate_tree.perception_angle,
            scene.generate_tree.branching_angle,
            scene.generate_tree.coefficient_of_proportionality,
            scene.generate_tree.lambda_coefficient,
            scene.generate_tree.weight,
            scene.generate_tree.number_of_iterations,
            scene.generate_tree.max_number_of_buds
        )
        
        # cProfile.runctx("""generateTree(
            # scene.generate_tree.number_of_attraction_points,
            # scene.generate_tree.attraction_points_width,
            # scene.generate_tree.attraction_points_height,
            # scene.generate_tree.internode_length,
            # scene.generate_tree.minimal_internode_diameter,
            # scene.generate_tree.internode_diameter_coefficient,
            # scene.generate_tree.radius_of_influence,
            # scene.generate_tree.kill_distance,
            # scene.generate_tree.perception_angle,
            # scene.generate_tree.branching_angle,
            # scene.generate_tree.coefficient_of_proportionality,
            # scene.generate_tree.lambda_coefficient,
            # scene.generate_tree.weight,
            # scene.generate_tree.number_of_iterations,
            # scene.generate_tree.max_number_of_buds
        # )""", globals(), locals(), sort="cumtime")


        return {'FINISHED'}


class GenerateTreeSettings(PropertyGroup):
    number_of_attraction_points: IntProperty(
        name="Number of attraction points to generate",
        description="",
        min=1,
        default=60000,
    )

    attraction_points_width: IntProperty(
        name="Attraction points width",
        description="Width of the generated attraction points ellipsoid",
        min=1,
        default=200,
    )

    attraction_points_height: IntProperty(
        name="Attraction points height",
        description="Height of the generated attraction points ellipsoid",
        min=1,
        default=400,
    )

    internode_length: FloatProperty(
        name="Internode length",
        description="Internode length",
        min=1.0, max=100.0,
        default=8.0,
    )

    minimal_internode_diameter: FloatProperty(
        name="Minimal internode diameter",
        description="Internode diameter of terminal internodees",
        min=1.0, max=100.0,
        default=1.0,
    )

    internode_diameter_coefficient: FloatProperty(
        name="Internode diameter coefficient",
        description="Usually between 2 and 3",
        min=1.0,
        default=3.0,
    )

    radius_of_influence: FloatProperty(
        name="Radius of influence",
        description="At this distance (in internode lengths) attraction points start to influnce the internode closest to it - Usually between 4 and 6 internode lengths",
        min=1.0,
        default=6.0,
    )

    kill_distance: FloatProperty(
        name="Kill distance",
        description="Attraction point is removed when there is at least one tree internode closer than kill distance (in internode lengths) - Usually 2 internode lengths",
        min=1.0,
        default=2.0,
    )

    perception_angle: FloatProperty(
        name="Perception angle",
        description="We ignore attraction points that are outside conical perception volume with this angle - tipically 90 degrees",
        min=0.0, max=360.0,
        default=90.0,
    )

    branching_angle: FloatProperty(
        name="Branching angle",
        description="An angle, around which a new internode is looking for attraction points",
        min=0.0, max=360.0,
        default=120.0,
    )

    coefficient_of_proportionality: FloatProperty(
        name="Coefficient of proportionality",
        description="Coefficient of proportionality when calculating bud fate",
        min=1.0,
        default=3.0,
    )

    lambda_coefficient: FloatProperty(
        name="Lambda",
        description="Whether resource allocation is biased towards main axis (lambda_coefficient > 0.5), not biased (lambda_coefficient == 0.5), or biased toward the lateral internode (lambda_coefficient < 0.5)",
        min=0.0, max=1.0,
        default=0.57,
    )

    weight: FloatProperty(
        name="Weight",
        description="The weight which is applied to a new internode (tropism)",
        default=0.0,
    )

    number_of_iterations: IntProperty(
        name="Number of iterations",
        description="Number of iterations for tree generation",
        min=1,
        default=13,
    )

    max_number_of_buds: IntProperty(
        name="Max number of buds",
        description="Max number of buds for tree generation",
        min=1,
        default=10000,
    )


class GenerateTreePanel(Panel):
    bl_idname = "VIEW3D_PT_generate_tree"
    bl_label = "Generate Tree"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        scene = context.scene
        layout = self.layout

        # Prikažemo polja, s katerimo spreminjamo uporabniške parametre preko kazalca
        layout.prop(scene.generate_tree, "number_of_attraction_points")
        layout.prop(scene.generate_tree, "attraction_points_width")
        layout.prop(scene.generate_tree, "attraction_points_height")
        layout.prop(scene.generate_tree, "internode_length")
        layout.prop(scene.generate_tree, "minimal_internode_diameter")
        layout.prop(scene.generate_tree, "internode_diameter_coefficient")
        layout.prop(scene.generate_tree, "radius_of_influence")
        layout.prop(scene.generate_tree, "kill_distance")
        layout.prop(scene.generate_tree, "perception_angle")
        layout.prop(scene.generate_tree, "branching_angle")
        layout.prop(scene.generate_tree, "coefficient_of_proportionality")
        layout.prop(scene.generate_tree, "lambda_coefficient")
        layout.prop(scene.generate_tree, "weight")
        layout.prop(scene.generate_tree, "number_of_iterations")
        layout.prop(scene.generate_tree, "max_number_of_buds")

        # Panelu dodamo gumb, s katerim zaženemo upravljalec
        layout.operator(GenerateTreeOperator.bl_idname)



def register():
    register_class(GenerateTreeSettings)
    register_class(GenerateTreeOperator)
    register_class(GenerateTreePanel)

    bpy.types.Scene.generate_tree = PointerProperty(type=GenerateTreeSettings)


def unregister():
    unregister_class(GenerateTreeSettings)
    unregister_class(GenerateTreeOperator)
    unregister_class(GenerateTreePanel)


# SPACE COLONIZATION ALGORITHM
NUMBER_OF_ATTRACTION_POINTS = 10000
ATTRACTION_POINTS_WIDTH = 100
ATTRACTION_POINTS_HEIGHT = 200

INTERNODE_LENGTH = 5

# The diameter of terminal INTERNODEes
MINIMAL_INTERNODE_DIAMETER = 1

# Usually between 2 and 3
INTERNODE_DIAMETER_COEFFIENT = 3

NUM_VERTICES = 12

# radius of influence - when attraction point influences the tree node
# that is closest to it
# - expressed as multiple of INTERNODE_LENGTH
# - tipically 4-6 internode lengths
RADIUS_OF_INFLUENCE = 6
# RADIUS_OF_INFLUENCE *= INTERNODE_LENGTH

# kill distance - attraction point is removed when there is at least one tree node
# closer than kill distance
# - expressed as multiple of INTERNODE_LENGTH
# - tipically 2 internode lengths
KILL_DISTANCE = 2
# KILL_DISTANCE *= INTERNODE_LENGTH

# Adjustment for SOTM (self-organizing tree models)
# We ignore the points that are outside conical perception volume
# We divide by two, as it can only go PERCEPTION_ANGLE/2 to every side
# - tipically 90 degrees
PERCEPTION_ANGLE = 90
# PERCEPTION_ANGLE /= 2


# SOTM - an angle, at which a new branch sprouts
BRANCHING_ANGLE = 90

# SOTM - Bud fate: Coefficient of proportionality
ALPHA = 2

# SOTM - Bud fate: whether resource allocation is biased towards main axis (lambda_coefficient > 0.5),
# not biased (lambda_coefficient == 0.5), or biased toward the lateral internode (lambda_coefficient < 0.5)
LAMBDA_COEFFICIENT = 0.50

# SOTM - number of iterations
NUMBER_OF_ITERATIONS = 2

MAX_NUMBER_OF_BUDS = 100000

WEIGHT = 0


def resetScene():
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
        else:
            obj.select_set(False)
    bpy.ops.object.delete()


def create_mesh(vertices, edges, faces):
    mesh = bpy.data.meshes.new('tree')
    # print("create_mesh\n  vertices {}\n  edges {}\n  faces {}".format(vertices, edges, faces))
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate()
    mesh.update()
    return mesh

# We generate attraction_points uniformly distributed in an ellipsoid
# by generating points in a cube and then throwing out points
# not in the ellipsoid
#   numberOfBuds: number of attraction_points generated in the ellipsis
#   r: radius of the cube
#   rz: vertical radius of the cube
#   p: center point of the cube


def ellipsoid(number_of_attraction_points, r, rz):
    attraction_points = []
    r2 = r*r
    z2 = rz*rz
    p = Vector((0, 0, rz))
    if rz > r:
        r = rz
    while True:
        x = (random.random()*2-1)*r
        y = (random.random()*2-1)*r
        z = (random.random()*2-1)*r
        if x*x/r2+y*y/r2+z*z/z2 <= 1:
            pos = p + Vector((x, y, z))
            attraction_points.append(AttractionPoint(pos))
        if (len(attraction_points) >= number_of_attraction_points):
            break
    return attraction_points


def getDistance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)


def colorMesh(mesh):
    mymat = bpy.data.materials.new("mat")
    # We change the tree to brown
    mymat.diffuse_color = (0.40, 0.29, 0.16, 1.0)
    mesh.materials.append(mymat)


def findClosest(curr, buds):
    closest = None
    closestIndex = None
    closestDistance = 9999

    for i in range(len(buds)):
        distance = getDistance(curr.pos, buds[i].pos)
        if(distance < closestDistance):
            closest = buds[i]
            closestIndex = i
            closestDistance = distance

    del buds[closestIndex]
    return closest


def vector2Array(vector):
    return [pos for pos in vector]


def create_vertices(trans_mat, rotation_mat, diameter, is_end, display=False):
    v = []
    radius = diameter / 2
    if is_end:
        # Če je to končni internodij, zapremo cilinder
        return [trans_mat @ rotation_mat @ Vector((0, 0, 0))] * NUM_VERTICES

    angle = 0.0
    inc = 2*math.pi/NUM_VERTICES
    for i in range(0, NUM_VERTICES):
        vertex = Vector(
            (radius * math.cos(angle), radius * math.sin(angle), 0))
        vertex = trans_mat @ rotation_mat @ vertex
        v.append(vertex)
        if display:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=vertex)
        angle += inc

    return v


def connect(quads, last_indices, new_indices):
    for i in range(0, NUM_VERTICES):
        quads.append([last_indices[i], last_indices[i - 1],
                      new_indices[i - 1], new_indices[i]])


def normalizedAverageOfVectors(origin, buds):
    averaged_vector = Vector((0, 0, 0))
    for bud in buds:
        subVector = bud.pos - origin
        averaged_vector += (subVector / subVector.magnitude)

    normalized_averaged_vector = averaged_vector / averaged_vector.magnitude
    return normalized_averaged_vector


def getDistance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if vector == Vector((0, 0, 0)):
        return vector
    return vector / numpy.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return numpy.arcsin(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))


def prepare_internode_candidate(current_internode, attraction_points_in_range, is_lateral_internode=False):
    random_number_x = random.choice([-1, 0, 1])
    random_number_y = random.choice(
        [-1, 0, 1]) if random_number_x != 0 else random.choice([-1, 1])
    # Stranska veja raste pod naključnim kotom okoli glavne osi
    rotation_matrix = Matrix.Rotation(
        math.radians(BRANCHING_ANGLE) * (1 if is_lateral_internode else 0), 4, (random_number_x, random_number_y, 0))

    internode_vector = (current_internode.bud2.pos -
                     current_internode.bud1.pos) @ rotation_matrix

    # Pridobimo samo atrakcijske točke v spoznavni prostornini popka
    attraction_points_in_cone = [point for point in attraction_points_in_range if math.degrees(
        internode_vector.angle(point.pos - current_internode.bud1.pos)) < PERCEPTION_ANGLE]

    if is_lateral_internode == False:
        if len(attraction_points_in_cone) > 0 and len(current_internode.children) == 0:
            # Če je set atrakcijskih točk v spoznavni prostornini popka prazen,
            # popek več nima prostora za rast (Q=0)
            current_internode.space_for_growth = True
        else:
            return []

        # Povprečje normaliziranih vektorjev proti vsem atrakcijskim točkam
        normalized_averaged_vector = normalizedAverageOfVectors(
            current_internode.bud2.pos, attraction_points_in_cone)

        current_internode.optimal_growth_direction = normalized_averaged_vector
        return [attraction_point for attraction_point in attraction_points_in_range
                if attraction_point not in attraction_points_in_cone]
    else:
        if len(attraction_points_in_cone) == 0:
            return []

        # Povprečje normaliziranih vektorjev proti vsem atrakcijskim točkam
        normalized_averaged_vector = normalizedAverageOfVectors(
            current_internode.bud2.pos, attraction_points_in_cone)

        current_internode.lateral_internode_optimal_growth_direction = normalized_averaged_vector
        return


def grow_internode_candidate(internode, all_internodes, all_internode_pos, is_lateral_internode=False):
    optimal_growth_direction = None
    new_internode_pos = None

    if (is_lateral_internode == False):
        optimal_growth_direction = internode.optimal_growth_direction
    else:
        optimal_growth_direction = internode.lateral_internode_optimal_growth_direction

    is_new_internode_stem = False
    if (is_lateral_internode == False and internode.is_stem == True):
        is_new_internode_stem = True

    if is_lateral_internode == False and is_new_internode_stem == False:
        tropism = Vector(optimal_growth_direction) - Vector((0, 0, WEIGHT))
        tropism /= tropism.magnitude
        new_internode_pos = internode.bud2.pos + \
            tropism * INTERNODE_LENGTH * internode.l
    else:
        new_internode_pos = internode.bud2.pos + \
            Vector(optimal_growth_direction) * INTERNODE_LENGTH * internode.l

    # Shranjujemo vsako koordinato novega popka, in preverjamo, da nimamo duplikatov
    new_internode_pos_tuple = tuple(new_internode_pos)
    if new_internode_pos_tuple in all_internode_pos:
        return False
    else:
        all_internode_pos.add(new_internode_pos_tuple)

    new_internode = Internode(bud1=internode.bud2, bud2=Bud(
        new_internode_pos), parent=internode, is_stem=is_new_internode_stem)
    internode.children.append(new_internode)

    internode.space_for_growth = False

    if is_lateral_internode == False:
        new_internode.n = internode.n - 1
    else:
        new_internode.n = 0

    all_internodes.append(new_internode)
    return True


class Bud:
    def __init__(self, pos):
        self.pos = pos


class AttractionPoint:
    def __init__(self, pos):
        self.pos = pos


class Internode:
    def __init__(self, bud1, bud2, parent, is_stem=False):
        self.bud1 = bud1
        self.bud2 = bud2
        self.parent = parent
        self.children = []
        self.is_stem = is_stem
        self.optimal_growth_direction = None
        self.lateral_internode_optimal_growth_direction = None
        self.space_for_growth = False
        self.Q = 0
        self.v = 0
        self.n = 0
        self.l = 1
        self.d = 0


def generateTree(
    number_of_attraction_points,
    attraction_points_width,
    attraction_points_height,
    internode_length,
    minimal_internode_diameter,
    internode_diameter_coefficient,
    radius_of_influence,
    kill_distance,
    perception_angle,
    branching_angle,
    coefficient_of_proportionality,
    lambda_coefficient,
    weight,
    number_of_iterations,
    max_number_of_buds
):
    start_generation_time = timer()
    tracemalloc.start()

    global NUMBER_OF_ATTRACTION_POINTS
    global ATTRACTION_POINTS_WIDTH
    global ATTRACTION_POINTS_HEIGHT
    global INTERNODE_LENGTH
    global MINIMAL_INTERNODE_DIAMETER
    global INTERNODE_DIAMETER_COEFFIENT
    global RADIUS_OF_INFLUENCE
    global KILL_DISTANCE
    global PERCEPTION_ANGLE
    global BRANCHING_ANGLE
    global ALPHA
    global LAMBDA_COEFFICIENT
    global WEIGHT
    global NUMBER_OF_ITERATIONS
    global MAX_NUMBER_OF_BUDS

    NUMBER_OF_ATTRACTION_POINTS = number_of_attraction_points
    ATTRACTION_POINTS_WIDTH = attraction_points_width
    ATTRACTION_POINTS_HEIGHT = attraction_points_height

    INTERNODE_LENGTH = internode_length
    MINIMAL_INTERNODE_DIAMETER = minimal_internode_diameter
    INTERNODE_DIAMETER_COEFFIENT = internode_diameter_coefficient

    RADIUS_OF_INFLUENCE = radius_of_influence
    RADIUS_OF_INFLUENCE *= INTERNODE_LENGTH

    KILL_DISTANCE = kill_distance
    KILL_DISTANCE *= INTERNODE_LENGTH

    PERCEPTION_ANGLE = perception_angle
    PERCEPTION_ANGLE /= 2

    BRANCHING_ANGLE = branching_angle

    ALPHA = coefficient_of_proportionality
    LAMBDA_COEFFICIENT = lambda_coefficient

    WEIGHT = weight

    NUMBER_OF_ITERATIONS = number_of_iterations

    MAX_NUMBER_OF_BUDS = max_number_of_buds

    resetScene()

    # Using space colonization algorithm
    # Buds are the generated attraction points
    attraction_points = []
    # attraction_points = [
    #     Bud(Vector((0, 0, 10))),
    #     Bud(Vector((-10, 0, 20))),
    #     Bud(Vector((-20, 0, 15))),
    #     Bud(Vector((0, -15, 50))),
    #     Bud(Vector((0, 25, 25))),
    #     Bud(Vector((0, 5, 30))),
    #     # Bud(Vector((15, 0, 10))),
    #     Bud(Vector((0, 35, 20))),
    # #     Bud(Vector((10, 0, 35))),
    # #     Bud(Vector((-20, 0, 30))),
    # ]
    attraction_points = ellipsoid(NUMBER_OF_ATTRACTION_POINTS, r=ATTRACTION_POINTS_WIDTH,
                                  rz=ATTRACTION_POINTS_HEIGHT)

    for attraction_point in attraction_points:
        # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=attraction_point.pos)
        pass

    # We create the stem
    root = Bud(Vector((0, 0, 0)))
    # We seet root so that it faces up, but for a very small number
    root_up = Bud(Vector((0, 0, sys.float_info.epsilon)))

    rootInternode = Internode(root, root_up, None, is_stem=True)
    rootInternode.n = 1

    internodes = [rootInternode]

    # A lateral internode can get same position as a main internode
    # In this case, we don't grow the internode
    # Here, we store positions of all internodes
    all_internode_pos = set()

    # We calculate how far away each internode is from root,
    # and store it in this dict
    # Dictionary is ordered, first are terminal branhches, on the end is root internode
    internode_order_dict = {}

    print("\n*************\nSTART\n*************")

    #########################################
    # MODIFIED SPACE COLONIZATION ALGORITHM #
    #########################################
    def getInternodeGrowthCandidates(internodes):
        return [internode for internode in internodes if internode.space_for_growth == True and internode.n > 0]
    
    iteration_count = 1
    last_iteration_start = timer()

    for i in range(NUMBER_OF_ITERATIONS):
        print(f"iteration {iteration_count}: {timedelta(seconds=timer()-last_iteration_start)}")
        last_iteration_start = timer()
        iteration_count += 1
        print(f"number of buds: {len(all_internode_pos)}")

        if len(all_internode_pos) >= MAX_NUMBER_OF_BUDS:
            break
        else:
            if len(attraction_points) == 0:
                print("Error: Every attraction points was reached")
                break

            while True:
                internodes_kdtree = scipy.spatial.cKDTree(
                    [internode.bud2.pos[:] for internode in internodes])
                points = [attraction_point.pos[:] for attraction_point in attraction_points]

                # closest_bud_distances - seznam razdalj od atrakcijskih točk do nabližjega popka
                # closest_buds - indeks popka, ki je najbližji atrakcijski točki
                closest_bud_distances, closest_buds = internodes_kdtree.query(
                    points,
                    distance_upper_bound=RADIUS_OF_INFLUENCE,
                    workers=-1
                )

                # Dodelimo atrakcijske točke najbližjim popkom
                assigned_attraction_points = [[] for i in internodes]
                assigned_attraction_point_distances = [[] for i in internodes]
                is_empty = True

                attraction_points_to_remove = []
                for i in range(len(closest_buds)):
                    # Razdalja je inf, če je atrakcijska točka oddaljena več kot RADIUS_OF_INFLUENCE od popka
                    # Izločimo atrakcijske točke, ki so od pripadajočega popka oddaljene za manj kot KILL_DISTANCE
                    if (closest_bud_distances[i] != numpy.inf):
                        if closest_bud_distances[i] < KILL_DISTANCE:
                            attraction_points_to_remove.append(
                                attraction_points[i])
                        else:
                            assigned_attraction_points[closest_buds[i]].append(
                                attraction_points[i])
                            assigned_attraction_point_distances[closest_buds[i]].append(
                                closest_bud_distances[i])
                            is_empty = False
                
                attraction_points = list((set(attraction_points) - set(attraction_points_to_remove)))

                if is_empty:
                    print("Napaka: Ni več atrakcijskih točk v dosegu")
                    break

                for idx, attraction_points_in_range in enumerate(assigned_attraction_points):
                    current_internode = internodes[idx]

                    # Pripravimo glavno os in dobimo preostale atrakcijske točke v dosegu
                    attraction_points_in_range = prepare_internode_candidate(
                        current_internode, attraction_points_in_range)
                    if len(attraction_points_in_range) == 0:
                        if internodes == [rootInternode]:
                            break
                    else:
                        prepare_internode_candidate(
                            current_internode, attraction_points_in_range, is_lateral_internode=True)

                if len(internode_growth_candidates := getInternodeGrowthCandidates(internodes)) > 0 or internodes == [rootInternode]:
                    should_break = False
                    for internode in internode_growth_candidates:
                        if len(all_internode_pos) < MAX_NUMBER_OF_BUDS:
                            grow_internode_candidate(
                                internode, internodes, all_internode_pos, is_lateral_internode=False)

                            if internode.lateral_internode_optimal_growth_direction != None:
                                grow_internode_candidate(
                                    internode, internodes, all_internode_pos, is_lateral_internode=True)
                        else:
                            should_break = True
                            break
                    if should_break:
                        break
                else:
                    break

                if len(internode_growth_candidates) == 0 and internodes == [rootInternode]:
                    print("Error: Root internode can not grow")
                    return

        stack = []
        internode_order_dict = {}

        for internode in internodes:
            internode_order_dict[internode] = 1
            if len(internode.children) == 0:
                stack.append(internode)

        while len(stack) > 0:
            internode = stack.pop()

            for child in internode.children:
                internode_order_dict[internode] += internode_order_dict[child]

            if internode.parent != None:
                stack.append(internode.parent)

        # Sortiramo internode_order_dict - terminalni internodiji na zadnjem mestu, koreninski internodij pa na prvem
        internode_order_dict = {k: v for k, v in sorted(
            internode_order_dict.items(), key=lambda item: item[1], reverse=False)}

        # for internode in internode_order_dict:
        #     print(f"{internode.bud2.pos}, {internode_order_dict[internode]}, children={[internode.bud2.pos[0] for internode in internode.children]}")

        ###########################
        # CALCULATION OF BUD FATE #
        ###########################
        # We store the terminal buds, so we perform the algorithm for Extended BH model
        # quicker - in the first pass, information about the amount of light Q that
        # reaches the buds flows basipetally (from terminal buds to the root)
        # and the cumulative values are stored within the internodes (internodes)

        ### FIRST PASS ####

        for internode in internode_order_dict:
            if len(internode.children) == 0 and (internode.space_for_growth or internode == rootInternode):
                internode.Q = 1
                if len(internodes) > 1 and internode == rootInternode:
                    internode.Q = 0
            else:
                internode.Q = 0
            # Pripravimo se za drugi prehod, tako da resetiramo vrednosti indernodija
            internode.v = 0
            internode.n = 0
            internode.l = 1

        for internode in internode_order_dict:
            if len(internode.children) == 0:
                pass
            elif len(internode.children) == 1:
                internode.Q += internode.children[0].Q
            else:
                for child in internode.children:
                    internode.Q += child.Q

            # print(f"{internode.bud2.pos}: Q={internode.Q}")
        # print('************')

        # for internode in internodes:
        # print('-------')

        ### SECOND PASS ###

        v_base = ALPHA * rootInternode.Q
        rootInternode.v = v_base
        rootInternode.n = math.floor(rootInternode.v)

        for internode in reversed(internode_order_dict):
            if len(internode.children) > 0:
                main_axis = internode.children[0]
                lateral_internode = internode.children[1] if len(
                    internode.children) > 1 else None

                # Otrok z indeksom 0 je nadaljevanje glavne osi
                Q_m = main_axis.Q
                Q_l = lateral_internode.Q if lateral_internode != None else 0

                if Q_m == 0 and Q_l == 0:
                    continue

                v_m = internode.v * (LAMBDA_COEFFICIENT * Q_m) / \
                    (LAMBDA_COEFFICIENT * Q_m + (1 - LAMBDA_COEFFICIENT) * Q_l)
                v_l = internode.v * ((1 - LAMBDA_COEFFICIENT) * Q_l) / \
                    (LAMBDA_COEFFICIENT * Q_m + (1 - LAMBDA_COEFFICIENT) * Q_l)

                main_axis.v = v_m
                main_axis.n = math.floor(main_axis.v)
                if main_axis.n == 0:
                    main_axis.l = 1
                else:
                    main_axis.l = main_axis.v / main_axis.n

                if lateral_internode != None:
                    lateral_internode.v = v_l
                    lateral_internode.n = math.floor(lateral_internode.v)
                    if lateral_internode.n == 0:
                        lateral_internode.l = 1
                    else:
                        lateral_internode.l = lateral_internode.v / lateral_internode.n

            # print(f"{internode.bud2.pos[0]}: Q={internode.Q} v={internode.v} n={internode.n} l={internode.l} children={[internode.pos[0] for internode in internode.children]}")

        # print('------------')

    # # TEST
    # internode1 = Internode(rootInternode.bud2, Bud(Vector((2, 0, 4))), rootInternode)
    # internode3 = Internode(internode1.bud2, Bud(Vector((3, 0, 5))), internode1)
    # internode9 = Internode(internode3.bud2, Bud(Vector((5, 0, 6))), internode3)
    # internode8 = Internode(internode9.bud2, Bud(Vector((7, 0, 7))), internode9)
    # # # internode4 = Internode(internode1.bud2, Bud(Vector((-4, 5, 9))), internode1)
    # # # internode5 = Internode(internode4.bud2, Bud(Vector((-4, 7, 12))), internode4)

    # rootInternode.children.append(internode1)
    # internode1.children.append(internode3)
    # internode3.children.append(internode9)
    # internode9.children.append(internode8)
    # # # internode4.children.append(internode5)
    # # # internode7.children.append(internode8)

    # internodees.append(internode1)
    # internodees.append(internode3)
    # internodees.append(internode9)
    # internodees.append(internode8)

    ##################################
    # CALCULATION OF BRANCH DIAMETER #
    ##################################

    for internode in internode_order_dict:
        if len(internode.children) == 0:
            internode.d = MINIMAL_INTERNODE_DIAMETER
        else:
            internode.d = 0
            if len(internode.children) == 2:
                internode.d += (internode.children[0].d ** INTERNODE_DIAMETER_COEFFIENT +
                                internode.children[1].d ** INTERNODE_DIAMETER_COEFFIENT) ** (1/INTERNODE_DIAMETER_COEFFIENT)
            elif len(internode.children) == 1:
                internode.d += internode.children[0].d

    # for internode in internodes:
    #     print(
    #         f"{internode.bud2.pos}: d={internode.d}")
    #     pass

    # for internode in internodes:
    #     print(
    #         f"{internode.bud2.pos}: stem={internode.is_stem}")
    #     pass
    ############################
    # MODELING OF THE INTERNODES #
    ############################

    faces = []
    vertices = []
    stack = []

    for child in rootInternode.children:
        stack.append((rootInternode, Matrix.Identity(4),  None))

    while len(stack) > 0:
        parent, parent_rotation, last_indices = stack.pop()

        for idx, internode in enumerate(parent.children):
            if idx > 0:
                parent, parent_rotation, last_indices = stack.pop()

            b1 = Matrix.Translation(parent.bud1.pos)
            start_index = len(vertices)

            v1 = create_vertices(b1, parent_rotation,
                                 diameter=parent.d, is_end=False)
            vertices.extend(v1)

            if parent.bud1 == root:
                last_indices = range(start_index, start_index + len(v1))

            b2 = Matrix.Translation(internode.bud1.pos)

            # We calculate the angle between the vector it's X axis
            x_angle = angle_between(
                parent.bud1.pos - parent.bud2.pos, (1, 0, 0))
            # We calculate the angle between the vector it's Y axis
            y_angle = angle_between(
                parent.bud1.pos - parent.bud2.pos, (0, 1, 0))

            # We don't prettify angle for lateral internodes
            if parent != None and len(parent.children) > 1 and internode == parent.children[1]:
                x_angle = angle_between(
                    internode.bud1.pos - internode.bud2.pos, (1, 0, 0))
                y_angle = angle_between(
                    internode.bud1.pos - internode.bud2.pos, (0, 1, 0))
                x_angle = -x_angle
                y_angle = y_angle
            else:
                x_angle = -x_angle/2
                y_angle = y_angle/2

            rotation_x = Matrix.Rotation(x_angle, 4, 'Y')
            rotation_y = Matrix.Rotation(y_angle, 4, 'X')
            rotation = rotation_x @ rotation_y

            v2 = create_vertices(b2, parent_rotation @
                                 rotation, diameter=internode.d, is_end=False)

            start_index = len(vertices)
            vertices.extend(v2)
            new_indices = range(start_index, start_index + len(v2))

            connect(faces, last_indices, new_indices)

            last_indices = new_indices

            if len(internode.children) == 0:
                b3 = Matrix.Translation(internode.bud2.pos)

                v3 = create_vertices(b3, rotation, diameter=0, is_end=True)

                start_index = len(vertices)
                vertices.extend(v3)
                new_indices_2 = range(start_index, start_index + len(v3))

                connect(faces, new_indices, new_indices_2)

            for idx, child in enumerate(internode.children):
                stack.insert(0, (internode, rotation, last_indices))

    mesh = create_mesh(vertices, [], faces)
    colorMesh(mesh)
    obj = bpy.data.objects.new('tree', mesh)

    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # We apply Subdivision Surface to the mesh, to get a smooth looking tree
    # https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/subdivision_surface.html
    # https://docs.blender.org/api/current/bpy.types.SubsurfModifier.html#bpy.types.SubsurfModifier
    # m = obj.modifiers.get("My SubDiv") or obj.modifiers.new(
    #     'My SubDiv', 'SUBSURF')
    # m.levels = 3
    # m.render_levels = 3
    # m.quality = 2

    for i, internode in enumerate(internodes):
        # print(f"{i} - pos1: {internode.bud1.pos}, pos2: {internode.bud2.pos}")
        pass


    def display_top(snapshot, key_type='lineno', limit=3):
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = snapshot.statistics(key_type)

        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))

    print("Number of internodes generated: ", len(all_internode_pos))
    print(f"generation time: {timedelta(seconds=timer()-start_generation_time)}")

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    print("*************\nEND\n*************")
    os.system('say "your program has finished"')


# def generate():
if __name__ == "__main__":
    register()
    # generateTree(
    #     NUMBER_OF_ATTRACTION_POINTS,
    #     ATTRACTION_POINTS_WIDTH,
    #     ATTRACTION_POINTS_HEIGHT,
    #     INTERNODE_LENGTH,
    #     MINIMAL_INTERNODE_DIAMETER,
    #     INTERNODE_DIAMETER_COEFFIENT,
    #     RADIUS_OF_INFLUENCE,
    #     KILL_DISTANCE,
    #     PERCEPTION_ANGLE,
    #     BRANCHING_ANGLE,
    #     ALPHA,
    #     LAMBDA_COEFFICIENT,
    #     WEIGHT,
    #     NUMBER_OF_ITERATIONS,
    #     MAX_NUMBER_OF_BUDS
    # )

# cProfile.run('generate()')
