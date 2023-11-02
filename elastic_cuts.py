from dolfin import *
from mshr import *
from matplotlib import cm, pyplot as pp
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import numpy as np, sys, os
import networkx as nx
import random as rnd
import skfmm  # fast marching for distances (scikit-fmm)


def _main():
    # ---make a timestamped folder to spam images
    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")
    rd = os.path.join(os.path.dirname(__file__), './results/' + dt_string)
    if not os.path.isdir(rd): os.makedirs(rd)
    # P1.---MESH GENERATION
    # --rectangle in parallel
    Nx, Ny = [100, 50]
    lx1, lx2 = [-0.5, 0.5]
    ly1, ly2 = [0.0, 0.5]
    XX, YY = np.meshgrid(np.linspace(lx1, lx2, Nx + 1), np.linspace(ly1, ly2, Ny + 1))
    dist_mat = np.zeros((Ny + 1, Nx + 1))
    sdbdy_mat = np.zeros((Ny + 1, Nx + 1))
    mesh_rect = RectangleMesh(Point(lx1, ly1), Point(lx2, ly2), Nx, Ny, 'crossed')
    Vr = FunctionSpace(mesh_rect, 'CG', 1)
    Vrvec = VectorFunctionSpace(mesh_rect, 'CG', 1)
    Vrmat = TensorFunctionSpace(mesh_rect, 'CG', 1)
    gdim = mesh_rect.geometry().dim()
    dofsVr_max = (Nx + 1) * (Ny + 1) + Nx * Ny
    dofsVr = Vr.tabulate_dof_coordinates().reshape((-1, gdim))
    pxr, pyr = [((-lx1 + dofsVr[:, 0]) / (lx2 - lx1)) * 2 * Nx, ((-ly1 + dofsVr[:, 1]) / (ly2 - ly1)) * 2 * Ny]
    dist_rect_fn = Function(Vr)
    # --main FEM mesh
    # circles
    circle1 = Circle(Point(0, 0), 5)
    circle2 = Circle(Point(-2.5, 0), 1.5)
    circle3 = Circle(Point(2.5, 0), 1.5)
    circle4 = Circle(Point(0, 2.5), 1.5)
    circle5 = Circle(Point(0, -2.5), 1.5)
    # domain = circle1 - circle2 - circle3 - circle4 - circle5
    # mesh = generate_mesh(domain, 50)
    # polygon
    # domain_vertices = [Point(1.0, 0.0), Point(0.25, 0.125),Point(0.0, 1.0),Point(-0.25, 0.125),
    #                   Point(-1.0, 0.0), Point(-0.25, -0.125),Point(0.0, -1.0),Point(0.25, -0.125)]
    # domain_vertices = [Point(lx1, ly1), Point(lx2, ly1), Point(lx2, ly2), Point(lx1, ly2)]
    # scaffold_vertices = [Point(lx1, ly1), Point(lx2, ly1), Point(lx2, ly1 + 1.0/16.0*(ly2-ly1)), Point(lx1, ly1 + 1.0/16.0*(ly2-ly1))]
    # domain = Polygon(domain_vertices) - Polygon(scaffold_vertices)
    # domain = Polygon(domain_vertices)
    domain = Rectangle(Point(lx1, ly1), Point(lx2, ly2))
    rect_left = Rectangle(Point(lx1, ly1), Point(lx1 + 0.5 * (lx2 - lx1), ly2))
    rect_right = Rectangle(Point(lx1 + 0.5 * (lx2 - lx1), ly1), Point(lx2, ly2))
    domain = rect_left + rect_right
    # domain.set_subdomain(1, rect_left)
    # domain.set_subdomain(2, rect_right)
    mesh = generate_mesh(domain, 100)
    # mesh = mesh_rect
    plot(mesh, linewidth=0.25)
    pp.savefig(rd + '/mesh.png', bbox_inches='tight', dpi=300)
    pp.close()
    mesh.init(2, 1)
    mesh.init(1, 2)
    mesh.init(1, 0)
    f2e = mesh.topology()(2, 1)
    e2f = mesh.topology()(1, 2)
    e2v = mesh.topology()(1, 0)
    f = open(rd + '/edge_list.txt', "w")
    flog = open(rd + '/log.txt', "w")
    V = FunctionSpace(mesh, 'DG', 0)
    Vvec = VectorFunctionSpace(mesh, 'DG', 0)
    Vmat = TensorFunctionSpace(mesh, 'DG', 0)
    VL = FunctionSpace(mesh, 'CG', 1)
    VLvec = VectorFunctionSpace(mesh, 'CG', 1)
    vol_face_fn = Function(V)
    bdy_length_fn = Function(V)
    src_sink_cap = Function(V)
    face_weight = Function(V)
    input_data = Function(V)
    # P1.---END
    # P2.---BOUNDARY PARAMETERS
    face_coeff = 30;
    bdy_coeff_bone = -0.95;
    bdy_coeff_scaffold = -0.9;
    bdy_coeff_side = 1;
    flog.write("Face coeff was %1.8f\n" % (face_coeff))
    # P2.---END
    # P3.---ELASTICITY PARAMETERS
    mu = 8.0  # Lame coefficient
    lmbda = 16.0  # Lame coefficient
    eps1 = 0.0001  # small parameter avoiding division by zero in the normal computation
    weak = 0.01  # multiplicative coefficient for the "weak" material mimicking void
    e0 = 0.9  # spontaneous strain along bottom boundary
    elas_coeff = 0.00  # coefficient in front of shape derivative for descent
    # P3.---END
    # P4.---SPONTANEOUS DISPLACEMENT
    '''u0x_mat = np.multiply( np.power( np.abs(XX/0.5), 1/3.0 ), -1.0*np.sign(XX) )
	u0y_mat = np.multiply(0.0*XX+0.0,np.arctan(4*pi*(YY+0.6)))	
	u0r = [Function(Vr) for _ in range(Vvec.num_sub_spaces())]
	u0r[0]=_mat2func(pxr, pyr, u0r[0], u0x_mat, dofsVr_max)
	u0r[1]=_mat2func(pxr, pyr, u0r[1], u0y_mat, dofsVr_max)
	u0r[0].set_allow_extrapolation(True)
	u0r[1].set_allow_extrapolation(True)
	u0s = [Function(VL) for _ in range(Vvec.num_sub_spaces())]
	u0s[0] = interpolate(u0r[0], VL)
	u0s[1] = interpolate(u0r[1], VL)	
	u0 = Function(VLvec)
	assign(u0, u0s)
	ax=plot(u0)
	pp.colorbar(ax, shrink=0.55, format ='%04.1f')
	pp.savefig(rd+'/spont_disp.png',bbox_inches='tight',dpi=300)
	pp.close()'''
    # P4.---END
    # P5.---EXPORT GRAPH FROM FENICS TO NETWORKX
    # geometry loop
    for face in range(mesh.num_faces()):
        face_obj = Face(mesh, face)
        edges = f2e(face)
        bdy_length = 0
        # print("At face ", face, "with area ", face_obj.area())
        for edge in edges:
            edge_obj = Edge(mesh, edge)
            mid = edge_obj.midpoint()
            length = edge_obj.length()
            # edge_verts = e2v(edge)
            edge_faces = e2f(edge)
            # print("Found faces ", edge_faces)
            center = Cell(mesh, face).midpoint()
            if len(edge_faces) > 1:
                if face == edge_faces[0]:
                    other_face = edge_faces[1]
                else:
                    other_face = edge_faces[0]
                other_center = Cell(mesh, other_face).midpoint()
                dist_cen = np.sqrt(np.power(center[0] - other_center[0], 2) + np.power(center[0] - other_center[0], 2))
                # print("Triangle", face, "is contiguous to triangle", other_face, "through an edge of length", length)
                # first capacity, then distance between centers
                f.write("%5d %5d %1.8f %1.8f\n" % (face, other_face, length, dist_cen))
            else:
                # print("--Triangle", face, "touches the boundary through an edge of length", length)
                if (mid[1] == 0.5):
                    bdy_length = bdy_length + bdy_coeff_side * length
                elif (mid[0] == -0.5) | (mid[0] == 0.5):
                    bdy_length = bdy_length + bdy_coeff_bone * length
                else:
                    bdy_length = bdy_length + bdy_coeff_scaffold * length
        vol_face_fn.vector()[face] = face_obj.area()
        bdy_length_fn.vector()[face] = bdy_length
    f.close()
    flog.close()
    # input loop
    for face in range(mesh.num_faces()):
        face_obj = Face(mesh, face)
        center = Cell(mesh, face).midpoint()
        prob = 0.0
        # input_val = ( ( center[0] + 0.25*center[1] < 0 ) & ( rnd.random() > prob ) ) | ( rnd.random() < prob )
        # input_val = ( ( np.maximum( np.abs( center[0] ), np.abs ( center[1] ) ) < 0.4 ) & ( rnd.random() > prob ) ) | ( rnd.random() < prob )
        # input_val = ( ( center[1] > 0.5*center[0] + 0.475 ) & ( center[1] < -0.5*center[0] + 0.525 ) ) |\
        #	        ( ( center[1] < 0.5*center[0] + 0.525 ) & ( center[1] > -0.5*center[0] + 0.475 ) ) | ( rnd.random() < prob )
        input_val = ((center[0] - lx1) ** 2 + (center[1] - (ly1 + 0.5 * (ly2 - ly1))) ** 2 < np.minimum(
            0.5 * (lx2 - lx1), 0.5 * (ly2 - ly1)) ** 2) | \
                    ((center[0] - lx2) ** 2 + (center[1] - (ly1 + 0.5 * (ly2 - ly1))) ** 2 < np.minimum(
                        0.5 * (lx2 - lx1), 0.5 * (ly2 - ly1)) ** 2)
        sl = 6.0
        input_val = (sl * (np.abs(center[0] - lx1)) + (center[1] - ly2) < 0.0) | \
                    ((center[0] - lx2) ** 2 + (center[1] - (ly1 + 0.5 * (ly2 - ly1))) ** 2 < np.minimum(
                        0.5 * (lx2 - lx1), 0.5 * (ly2 - ly1)) ** 2)
        # input_val = 0.0
        input_data.vector()[face] = input_val
    # plot other things
    # plot(mesh, linewidth=0.25)
    ax = plot(input_data, vmin=0.0, vmax=1.0)
    # pp.colorbar(ax, shrink=0.55, format ='%04.1f')
    pp.savefig(rd + '/input.png', bbox_inches='tight', dpi=300)
    pp.savefig(rd + '/cut' + str(0).zfill(3) + '.png', bbox_inches='tight', dpi=300)
    pp.close()
    # plot(mesh, linewidth=0.25)
    ax = plot(bdy_length_fn)
    pp.colorbar(ax, shrink=0.55, format='%01.5f')
    pp.savefig(rd + '/bdy_length.png', bbox_inches='tight', dpi=300)
    pp.close()
    G = nx.read_edgelist(rd + '/edge_list.txt', nodetype=int, data=(('capacity', float), ('separation', float),))
    print("Loaded graph has", nx.number_of_nodes(G), "nodes, and", nx.number_of_edges(G), "edges")
    rnd.seed(1234)
    """
	path = nx.shortest_path(G, source=0, target=rnd.randint(0, nx.number_of_nodes(G)-1), weight='capacity')
	print("A shortest path is", path)
	#print("DOFs are", len(dofsV))
	path_ind = Function(V)	
	for face in path:
		path_ind.vector()[face] = 1.0
	ax=plot(path_ind)			
	pp.colorbar(ax, shrink=0.55, format ='%04.1f')
	pp.savefig(rd+'/path.png',bbox_inches='tight', dpi=300)
	pp.close()	
	"""
    G.add_node('source')
    G.add_node('sink')
    for face in range(mesh.num_faces()):
        G.add_edge('source', face)
        G.add_edge(face, 'sink')
    print("After adding source and sink the graph has", nx.number_of_nodes(G), "nodes, and", nx.number_of_edges(G),
          "edges")
    # P5.---END
    # P6.---DECLARE SOME FUNCTIONS TO USE IN LOOP
    U = Function(VLvec)
    Ug = Function(VL)
    gradU = Function(Vmat)
    mismatch = Expression(("-1.0 * E0 * ( x[0] - 0.5*(Lx2+Lx1) )", " 0.0"), E0=e0, Lx1=lx1, Lx2=lx2, degree=1)
    normal_der = Function(Vvec)
    Ug_normal_der = Function(V)
    Ug_normal_der_fnr = Function(Vr)
    HUg_fnr = Function(Vr)
    dist_fnr = Function(Vr)
    dist_fn = Function(V)
    dist_mat_signed = np.zeros((Ny + 1, Nx + 1))
    normal_fnr = [Function(Vr) for _ in range(Vrvec.num_sub_spaces())]
    normal_fn = [Function(V) for _ in range(Vrvec.num_sub_spaces())]
    normal_fn_v = Function(Vvec)
    curv_fnr = Function(Vr)
    curv_fn = Function(V)
    sd_fn = Function(Vvec)
    sd_fnr = Function(Vrvec)
    sdn_fnr = Function(Vr)
    sdn_fn = Function(V)
    sdn_mat = np.zeros((Ny + 1, Nx + 1))
    cut_result = Function(V)
    cut_result.assign(input_data)

    # P6.---END
    # P7.---ELASTIC SUBDOMAIN AND ITS BOUNDARY
    class Omega(SubDomain):
        def inside(self, x, on_boundary):
            return lx1 <= x[0] <= lx2 and ly1 <= x[1] <= ly2 and cut_result(x) > 0.5

    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
    # P7.---END
    # P8.---SOLVER TO SMOOTH THE SHAPE DERIVATIVE
    normal = FacetNormal(mesh)
    theta, xi = [TrialFunction(VLvec), TestFunction(VLvec)]
    av = assemble((0.1 * inner(grad(theta), grad(xi)) + inner(theta, xi)) * dX \
                  + 1.0e4 * (inner(dot(theta, normal), dot(xi, normal)) * ds))
    solverav = LUSolver(av)
    # P8.---END
    # ---MAIN EVOLUTION LOOP
    max_it, it, stop = [50, 1, False]
    while it <= max_it and stop == False:
        face_coeff = face_coeff * 0.9
        print("--Doing iteration", it)
        # L1.---ELASTICITY, SHAPE DERIVATIVE
        omega = Omega()
        domains.set_all(0)
        omega.mark(domains, 1)
        dx = Measure('dx')(subdomain_data=domains)
        edens, elas_energy = _solve_elasticity_dirichlet(VLvec, U, V, cut_result, dx, lx1, lx2, ly1, ly2, mu, lmbda,
                                                         weak, e0)
        Ug = project(dot(U - mismatch, mismatch), VL)
        grad_Ug = project(grad(Ug), Vvec)
        # edens.set_allow_extrapolation(True)
        sdbdy_fnr = interpolate(edens, Vr)
        sdbdy_mat = _func2mat(pxr, pyr, sdbdy_fnr, sdbdy_mat, dofsVr_max)
        sdbdy_mat = gaussian_filter(sdbdy_mat, sigma=2)
        sdbdy_fnr = _mat2func(pxr, pyr, sdbdy_fnr, sdbdy_mat, dofsVr_max)
        edens = interpolate(sdbdy_fnr, VL)
        print('elastic energy is ', elas_energy)
        # need to add -\partial_n(gu)-Hgu where g = e_0 x extends the boundary condition
        # sd = _shape_der_dirichlet(VLvec, U, weak, mu, lmbda, dx, solverav)
        # sd.set_allow_extrapolation(True)
        # sd_fn = interpolate(sd, Vvec)
        # sd_fnr = interpolate(sd, Vrvec)
        # L1.---END
        # L2.---COMPUTE DISTANCE FUNCTION AND NORMAL TO CURRENT SET
        # --Distances by fmm
        # cut_result.set_allow_extrapolation(True)
        dist_fnr = interpolate(cut_result, Vr)
        dist_mat = _func2mat(pxr, pyr, dist_fnr, dist_mat, dofsVr_max)
        dist_mat = dist_mat - 0.5 * np.ones((Ny + 1, Nx + 1))
        dist_mat_signed = skfmm.distance(dist_mat, dx=(lx2 - lx1) / Nx)
        dist_mat = -1.0 * dist_mat_signed
        dist_fnr = _mat2func(pxr, pyr, dist_fnr, dist_mat, dofsVr_max)
        dist_fnr.set_allow_extrapolation(True)
        dist_fn = interpolate(dist_fnr, V)
        # --Global normal by finite differences on fmm result
        normal_mat = _normal(dist_mat_signed, lx2 - lx1, ly2 - ly1, Nx, Ny, eps1)
        normal_mat[0] *= -1.0
        normal_mat[1] *= -1.0
        curv_mat = _div(normal_mat[0], normal_mat[1], lx2 - lx1, ly2 - ly1, Nx, Ny)
        curv_mat = gaussian_filter(curv_mat, sigma=3)
        curv_fnr = _mat2func(pxr, pyr, curv_fnr, curv_mat, dofsVr_max)
        curv_fn = interpolate(curv_fnr, V)
        normal_fnr[0] = _mat2func(pxr, pyr, normal_fnr[0], normal_mat[0], dofsVr_max)
        normal_fnr[1] = _mat2func(pxr, pyr, normal_fnr[1], normal_mat[1], dofsVr_max)
        normal_fnr[0].set_allow_extrapolation(True)
        normal_fnr[1].set_allow_extrapolation(True)
        normal_fn[0] = interpolate(normal_fnr[0], V)
        normal_fn[1] = interpolate(normal_fnr[1], V)
        assign(normal_fn_v.sub(0), normal_fn[0])
        assign(normal_fn_v.sub(1), normal_fn[1])
        Ug_normalder = dot(grad_Ug, normal_fn_v)
        HUg = dot(Ug, curv_fn)
        Ug_normalder_fnr = interpolate(project(Ug_normalder, V), Vr)
        HUg_fnr = interpolate(project(HUg, V), Vr)
        # sd_fnr_0, sd_fnr_1  = sd_fnr.split(deepcopy=True)
        # sdn_fnr_vec = np.multiply( sd_fnr_0.vector(), normal_fnr[0].vector() ) + \
        #		       np.multiply( sd_fnr_1.vector(), normal_fnr[1].vector() )
        # sdn_fnr.vector()[:] = sdn_fnr_vec
        sdbdy_fnr.vector()[:] *= 0.0
        sdbdy_fnr.vector()[:] += HUg_fnr.vector()
        sdbdy_fnr.vector()[:] += Ug_normalder_fnr.vector()
        sdn_mat = _func2mat(pxr, pyr, sdbdy_fnr, sdn_mat, dofsVr_max)
        # sdn_mat = gaussian_filter(sdn_mat, sigma=5)
        sdn_mat = \
        skfmm.extension_velocities(dist_mat_signed - 0.0 * np.ones((Ny + 1, Nx + 1)), sdn_mat, dx=(lx2 - lx1) / Nx)[1]
        sdn_fnr = _mat2func(pxr, pyr, sdn_fnr, sdn_mat, dofsVr_max)
        sdn_fnr.set_allow_extrapolation(True)
        sdn_fn = interpolate(sdn_fnr, V)
        active_list = []
        # --Normal derivatives around interface and graph distance (unused atm)
        """
		gradU_array = gradU.vector().get_local().reshape((-1,2,2))
		gradU_array[:,0,1] = gradU_array[:,0,1] + gradU_array[:,1,0]
		gradU_array[:,1,0] = gradU_array[:,0,1]
		n0_array = normal_fn[0].vector().get_local()
		n1_array = normal_fn[1].vector().get_local()
		for face in range(mesh.num_faces()):
			nd0 = gradU_array[face,0,0] * n0_array[face] + gradU_array[face,0,1] * n1_array[face]
			nd1 = gradU_array[face,1,0] * n0_array[face] + gradU_array[face,1,1] * n1_array[face]
			normal_der.vector()[2*face]   = nd0
			normal_der.vector()[2*face+1] = nd1
		for face in range(mesh.num_faces()):
			edges = f2e(face)
			for edge in edges:
				edge_obj = Edge(mesh, edge)
				edge_faces = e2f(edge)
				#print("Found faces ", edge_faces)
				if len(edge_faces) > 1:
					if face == edge_faces[0]:
						other_face = edge_faces[1]
					else:
						other_face = edge_faces[0]
					if cut_result.vector()[face] != cut_result.vector()[other_face]:
						active_list.append(face)
						if cut_result.vector()[face] > 0.5:
							normal_der.vector().get_local().reshape((-1, 2))[2*face] = gradU_array[face,0,0] * n0_array[face] + gradU_array[face,0,1] * n1_array[face]
							normal_der.vector().get_local().reshape((-1, 2))[2*face+1] = gradU_array[face,1,0] * n0_array[face] + gradU_array[face,1,1] * n1_array[face]
		"""
        # dist_vector = nx.multi_source_dijkstra_path_length(G, active_list, cutoff=None, weight='separation')
        # for face in range(mesh.num_faces()):
        #	dist_fn.vector()[face] = dist_vector[face]
        # L2.---END
        # L3.---UPDATE CAPACITIES AND DO CUT
        for face in range(mesh.num_faces()):
            # src_sink_cap.vector()[face] = face_coeff*(1.0-2.0*input_data.vector()[face])*vol_face_fn.vector()[face] + bdy_length_fn.vector()[face]
            src_sink_cap.vector()[face] = face_coeff * (
                        dist_fn.vector()[face] - elas_coeff * sdn_fn.vector()[face] + 0.25) * (
                                                      1.0 - 2.0 * cut_result.vector()[face]) * vol_face_fn.vector()[
                                              face] + \
                                          bdy_length_fn.vector()[face]
            G['source'][face]['capacity'] = np.maximum(0.0, src_sink_cap.vector()[face])
            G[face]['sink']['capacity'] = np.maximum(0.0, -1.0 * src_sink_cap.vector()[face])
        cut_value, partition = nx.minimum_cut(G, 'source', 'sink')
        reachable, non_reachable = partition
        print("Cut value is", cut_value, "with face coeff", face_coeff)
        if len(reachable) == 1:
            print("--Not a great cut: Only source is reachable")
            stop = True
        elif len(non_reachable) == 1:
            print("--Not a great cut: Everything except sink is reachable")
            stop = True
        cut_result.assign(Constant(0))
        for face in non_reachable:
            if (face != 'source') & (face != 'sink'):
                cut_result.vector()[face] = 1.0
        # L3.---END
        # L4.---PLOT PROGRESS
        ax = plot(cut_result, vmin=0.0, vmax=1.0)
        # pp.colorbar(ax, shrink=0.55, format ='%04.1f')
        pp.savefig(rd + '/cut' + str(it).zfill(3) + '.png', bbox_inches='tight', dpi=300)
        pp.close()
        ax = plot(dist_fn)
        pp.colorbar(ax, shrink=0.55, format='%01.5f')
        pp.savefig(rd + '/dist' + str(it).zfill(3) + '.png', bbox_inches='tight', dpi=300)
        pp.close()
        ax = plot(curv_fn)
        pp.colorbar(ax, shrink=0.55, format='%01.5f')
        pp.savefig(rd + '/curv' + str(it).zfill(3) + '.png', bbox_inches='tight', dpi=300)
        pp.close()
        ax = plot(sdn_fn)
        pp.colorbar(ax, shrink=0.55, format='%01.5f')
        pp.savefig(rd + '/shapeder_normal' + str(it).zfill(3) + '.png', bbox_inches='tight', dpi=300)
        pp.close()
        # ax=plot(sd_fn)
        # pp.colorbar(ax, shrink=0.55, format ='%04.1f')
        # pp.savefig(rd+'/shapeder'+str(it).zfill(3)+'.png',bbox_inches='tight',dpi=300)
        # pp.close()
        ax = plot(U)
        pp.colorbar(ax, shrink=0.55, format='%04.1f')
        pp.savefig(rd + '/disp' + str(it).zfill(3) + '.png', bbox_inches='tight', dpi=300)
        pp.close()
        ax = plot(interpolate(project(Ug_normalder, V), VL))
        pp.colorbar(ax, shrink=0.55, format='%04.1f')
        pp.savefig(rd + '/normalder' + str(it).zfill(3) + '.png', bbox_inches='tight', dpi=300)
        pp.close()
        # edens_cut = Function(V)
        # cut_result.set_allow_extrapolation(True)
        # cut_result_cg = Function(VL)
        # cut_result_cg = interpolate(cut_result, VL)
        # edens_cut_vec = np.multiply( edens.vector(), cut_result.vector() )
        # edens_cut.vector()[:] = edens_cut_vec
        ax = plot(edens)
        pp.colorbar(ax, shrink=0.55, format='%01.5f')
        pp.savefig(rd + '/energy_density' + str(it).zfill(3) + '.png', bbox_inches='tight', dpi=300)
        pp.close()
        # L4.---END
        it = it + 1
    return


# -----------------------------------------------------------------------
def _solve_elasticity_spontaneous(Vvec, U, V, cut, dx, lx1, lx2, ly1, ly2, mu, lmbda, weak, u0):
    def left(x, on_boundary): return x[0] < lx1 + DOLFIN_EPS

    def right(x, on_boundary): return x[0] > lx2 - DOLFIN_EPS

    def up(x, on_boundary): return x[1] > ly2 - DOLFIN_EPS

    def down(x, on_boundary): return x[1] < ly1 + DOLFIN_EPS

    noslip = Constant((0.0, 0.0))
    bc0 = DirichletBC(Vvec, noslip, left)
    bc1 = DirichletBC(Vvec, noslip, right)
    bc2 = DirichletBC(Vvec, noslip, up)
    bc3 = DirichletBC(Vvec, noslip, down)
    bcs = [bc0, bc1, bc2, bc3]
    u, v = [TrialFunction(Vvec), TestFunction(Vvec)]
    S1 = 2.0 * mu * inner(sym(grad(u)), sym(grad(v))) + lmbda * div(u) * div(v)
    a = S1 * weak * dx(0) + S1 * dx(1)
    L0 = 2.0 * mu * inner(sym(grad(u0)), sym(grad(v))) + lmbda * div(u0) * div(v)
    L = L0 * weak * dx(0) + L0 * dx(1)
    A, b = assemble_system(a, L, bcs)
    solver = LUSolver(A)
    solver.solve(U.vector(), b)
    SE = 2.0 * mu * inner(sym(grad(U)), sym(grad(U))) + lmbda * div(U) * div(U) - \
         4.0 * mu * inner(sym(grad(u0)), sym(grad(U))) - 2.0 * lmbda * div(u0) * div(U) + \
         2.0 * mu * inner(sym(grad(u0)), sym(grad(u0))) + lmbda * div(u0) * div(u0)
    energy = assemble(SE * weak * dx(0) + SE * dx(1))
    edens_aux = Function(V)
    edens_aux = project(SE, V)
    edens = Function(V)
    edens_vec = np.multiply(edens_aux.vector(), cut.vector()) + \
                weak * np.multiply(edens_aux.vector(), 1.0 - cut.vector())
    edens.vector()[:] = edens_vec
    return edens_aux, energy


# -----------------------------------------------------------------------
def _solve_elasticity_dirichlet(Vvec, U, V, cut, dx, lx1, lx2, ly1, ly2, mu, lmbda, weak, e0):
    def left(x, on_boundary): return x[0] < lx1 + DOLFIN_EPS

    def right(x, on_boundary): return x[0] > lx2 - DOLFIN_EPS

    def up(x, on_boundary): return x[1] > ly2 - DOLFIN_EPS

    def down(x, on_boundary): return x[1] < ly1 + DOLFIN_EPS

    noslip = Constant((0.0, 0.0))
    mismatch = Expression(("-1.0 * E0 * ( x[0] - 0.5*(Lx2+Lx1) )", " 0.0"), E0=e0, Lx1=lx1, Lx2=lx2, degree=1)
    bc3 = DirichletBC(Vvec, mismatch, down)
    bcs = [bc3]
    u, v = [TrialFunction(Vvec), TestFunction(Vvec)]
    S1 = 2.0 * mu * inner(sym(grad(u)), sym(grad(v))) + lmbda * div(u) * div(v)
    a = S1 * weak * dx(0) + S1 * dx(1)
    L = inner(Constant((0.0, 0.0)), v) * dx
    A, b = assemble_system(a, L, bcs)
    solver = LUSolver(A)
    solver.solve(U.vector(), b)
    SE = 2.0 * mu * inner(sym(grad(U)), sym(grad(U))) + lmbda * div(U) * div(U)
    energy = assemble(SE * weak * dx(0) + SE * dx(1))
    edens_aux = Function(V)
    edens_aux = project(SE, V)
    edens = Function(V)
    edens_vec = np.multiply(edens_aux.vector(), cut.vector()) + \
                weak * np.multiply(edens_aux.vector(), 1.0 - cut.vector())
    edens.vector()[:] = edens_vec
    return edens, energy


# -----------------------------------------------------------------------
"""
def _solve_adjoint_dirichlet(Vvec, U, Q, V, cut, dx, lx1, lx2, ly1, ly2, mu, lmbda, weak, e0):
	def left (x, on_boundary): return x[0] < lx1 + DOLFIN_EPS
	def right(x, on_boundary): return x[0] > lx2 - DOLFIN_EPS
	def up   (x, on_boundary): return x[1] > ly2 - DOLFIN_EPS
	def down (x, on_boundary): return x[1] < ly1 + DOLFIN_EPS
	noslip = Constant((0.0, 0.0))
	mismatch = Expression( ("-1.0 * E0 * ( x[0] - 0.5*(Lx2+Lx1) )", " 0.0"), E0=e0, Lx1=lx1, Lx2=lx2, degree=1)
	bc3 = DirichletBC(Vvec, noslip, down)
	bcs = [bc3]
	u,v = [TrialFunction(Vvec), TestFunction(Vvec)]
	S1 = 2.0*mu*inner(sym(grad(u)),sym(grad(v))) + lmbda*div(u)*div(v)
	a = S1*weak*dx(0) + S1*dx(1)
	L = inner(Constant( (0.0, 0.0) ), v)*dx#need RHS from derivative of energy density using U and e0
	A, b = assemble_system(a, L, bcs)
	solver = LUSolver(A)
	solver.solve(Q.vector(), b)
	SE = 2.0*mu*inner(sym(grad(U)),sym(grad(U))) + lmbda*div(U)*div(U)
	energy = assemble(SE*weak*dx(0) + SE*dx(1))
	edens_aux = Function(V)
	edens_aux = project(SE, V)
	edens = Function(V)
	edens_vec = np.multiply( edens_aux.vector(), cut.vector() ) +\
		        weak*np.multiply( edens_aux.vector(), 1.0-cut.vector() )
	edens.vector()[:] = edens_vec
	return edens, energy
"""


# -----------------------------------------------------------------------
def _solve_elasticity_neumann(Vvec, U, V, cut, dx, lx1, lx2, ly1, ly2, mu, lmbda, weak, e0):
    def left(x, on_boundary): return x[0] < lx1 + DOLFIN_EPS

    def right(x, on_boundary): return x[0] > lx2 - DOLFIN_EPS

    def up(x, on_boundary): return x[1] > ly2 - DOLFIN_EPS

    def down(x, on_boundary): return x[1] < ly1 + DOLFIN_EPS

    noslip = Constant((0.0, 0.0))
    mismatch = Expression(("-1.0 * E0 * ( x[0] - 0.5*(Lx2+Lx1) ) * ( x[1] < 0.01 )", " 0.0"), E0=e0, Lx1=lx1, Lx2=lx2,
                          degree=1)
    bc0 = DirichletBC(Vvec, noslip, left)
    bc1 = DirichletBC(Vvec, noslip, right)
    bc2 = DirichletBC(Vvec, noslip, up)
    bc3 = DirichletBC(Vvec, noslip, down)
    bcs = [bc0, bc1]
    u, v = [TrialFunction(Vvec), TestFunction(Vvec)]
    S1 = 2.0 * mu * inner(sym(grad(u)), sym(grad(v))) + lmbda * div(u) * div(v)
    a = S1 * weak * dx(0) + S1 * dx(1)
    L = inner(Constant((0.0, 0.0)), v) * dx + inner(mismatch, v) * ds
    A, b = assemble_system(a, L, bcs)
    solver = LUSolver(A)
    solver.solve(U.vector(), b)
    SE = 2.0 * mu * inner(sym(grad(U)), sym(grad(U))) + lmbda * div(U) * div(U)
    energy = assemble(SE * weak * dx(0) + SE * dx(1))
    edens_aux = Function(V)
    edens_aux = project(SE, V)
    edens = Function(V)
    edens_vec = np.multiply(edens_aux.vector(), cut.vector()) + \
                weak * np.multiply(edens_aux.vector(), 1.0 - cut.vector())
    edens.vector()[:] = edens_vec
    return edens, energy


# -----------------------------------------------------------------------
def _solve_elasticity_alt(Vvec, U, dx, lx1, lx2, ly1, ly2, mu, lmbda, weak, u0):
    def left(x, on_boundary): return x[0] < lx1 + DOLFIN_EPS

    def right(x, on_boundary): return x[0] > lx2 - DOLFIN_EPS

    def up(x, on_boundary): return x[1] > ly2 - DOLFIN_EPS

    def down(x, on_boundary): return x[1] < ly1 + DOLFIN_EPS

    noslip = Constant((0.0, 0.0))
    bc0 = DirichletBC(Vvec, noslip, left)
    bc1 = DirichletBC(Vvec, noslip, right)
    bc2 = DirichletBC(Vvec, noslip, up)
    bc3 = DirichletBC(Vvec, noslip, down)
    bcs = [bc0, bc1, bc2, bc3]
    u = TrialFunction(Vvec)
    d = u.geometric_dimension()  # space dimension

    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    # return sym(nabla_grad(u))
    def sigma(u):
        return lmbda * div(u) * Identity(d) + 2 * mu * epsilon(u)

    # Define variational problem
    v = TestFunction(Vvec)
    aa = inner(sigma(u), epsilon(v))
    a = aa * weak * dx(0) + aa * dx(1)
    Ll = inner(sigma(u), epsilon(u0))
    L = Ll * weak * dx(0) + Ll * dx(1)
    # Compute solution
    solve(a == L, U, bcs)
    SE = inner(sigma(u), epsilon(u)) - inner(sigma(u), epsilon(u0))
    energy = 0.0  # assemble(weak*SE*dx(0) + SE*dx(1))
    return energy


# -----------------------------------------------------------------------
def _shape_der_spontaneous(Vvec, u, u0, weak, mu, lmbda, dx, solver):
    xi = TestFunction(Vvec)
    rv = 0.0
    eu, Du, Dxi = [sym(grad(u)), grad(u), grad(xi)]
    eu0, Du0 = [sym(grad(u0)), grad(u0)]
    S1 = 2 * mu * (2 * inner((Du.T) * eu, Dxi) - inner(eu, eu) * div(xi)) \
         + lmbda * (2 * inner(Du.T, Dxi) * div(u) - div(u) * div(u) * div(xi)) \
         + 2 * mu * (2 * inner((Du.T) * eu0, Dxi) + 2 * inner((Du0.T) * eu, Dxi) + 2 * inner(eu, eu0) * div(xi)) \
         + lmbda * (2 * inner(Du.T, Dxi) * div(u0) + 2 * inner(Du0.T, Dxi) * div(u) + 2 * div(u) * div(u0) * div(xi))
    rv += -assemble(weak * S1 * dx(0) + S1 * dx(1))
    th = Function(Vvec)
    solver.solve(th.vector(), rv)
    return th


# -----------------------------------------------------------------------
def _shape_der_dirichlet(Vvec, u, weak, mu, lmbda, dx, solver):
    xi = TestFunction(Vvec)
    rv = 0.0
    eu, Du, Dxi = [sym(grad(u)), grad(u), grad(xi)]
    S1 = 2 * mu * (2 * inner((Du.T) * eu, Dxi) - inner(eu, eu) * div(xi)) \
         + lmbda * (2 * inner(Du.T, Dxi) * div(u) - div(u) * div(u) * div(xi))
    rv += -assemble(weak * S1 * dx(0) + S1 * dx(1))
    th = Function(Vvec)
    solver.solve(th.vector(), rv)
    return th


# -----------------------------------------------------------------------
def _mat2func(px, py, fn, fn_mat, dofsV_max):
    for dof in range(0, dofsV_max):
        if np.rint(px[dof]) % 2 == .0:
            cx, cy = np.int_(np.rint([px[dof] / 2, py[dof] / 2]))
            fn.vector()[dof] = fn_mat[cy, cx]
        else:
            cx, cy = np.int_(np.floor([px[dof] / 2, py[dof] / 2]))
            fn.vector()[dof] = 0.25 * (fn_mat[cy, cx] + fn_mat[cy + 1, cx] \
                                       + fn_mat[cy, cx + 1] + fn_mat[cy + 1, cx + 1])
    return fn


# ----------------------------------------------------------------------
def _func2mat(px, py, fn, fn_mat, dofsV_max):
    fn_array = fn.vector().get_local()
    for dof in range(0, dofsV_max):
        cx, cy = np.int_(np.rint([px[dof] / 2, py[dof] / 2]))
        fn_mat[cy, cx] = fn_array[dof]
    return fn_mat


# -----------------------------------------------------------------------
def _normal(phi, lx, ly, Nx, Ny, eps):  # takes a level set function
    Dx = np.gradient(phi, lx / Nx, axis=1)
    Dy = np.gradient(phi, ly / Ny, axis=0)
    gradNorm = np.sqrt(Dx ** 2 + Dy ** 2 + eps)
    normalx = Dx / gradNorm
    normaly = Dy / gradNorm
    normal = [normalx, normaly]
    return normal


# -----------------------------------------------------------------------
def _div(vx, vy, lx, ly, Nx, Ny):
    Dx = np.gradient(vx, lx / Nx, axis=1)
    Dy = np.gradient(vy, ly / Ny, axis=0)
    return Dx + Dy


# -----------------------------------------------------------------------
def _curvature(phi, lx, ly, Nx, Ny):  # takes a level set function
    # careful grid scaling!
    stencil = (1.0 / (12.0 * lx / Nx * lx / Nx)) * np.array(
        [[0, 0, -1, 0, 0],
         [0, 0, 16, 0, 0],
         [-1, 16, -60, 16, -1],
         [0, 0, 16, 0, 0],
         [0, 0, -1, 0, 0]])
    curv = convolve(phi, stencil, mode='constant')
    return curv


# -----------------------------------------------------------------------
if __name__ == '__main__':
    _main()
