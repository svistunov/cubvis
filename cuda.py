# -*- coding: utf-8 -*-
load_cuda = True
import Blender
import numpy
import sys
import time
import math
from pprint import pprint
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
if (load_cuda):
    import pycuda.driver as cuda_driver
    import pycuda.gl as cuda_gl
    from pycuda.compiler import SourceModule
import os
from Cheetah.Template import Template


"""
TODO:
    refactoring:
        split to classes
    blender:
        create configuration window
        create separated export to very simple format
    use 2d texture memory for cuda
    write comments
"""


class CudaRender:
    """Main class. Do all stuff"""
    def __init__(self, w = 800, h = 800, name = "CudaRender"):
        self.w = w
        self.h = h
        self.name = name
        self.buffers = {}
        self.pointers = {}
        self.scn = Blender.Scene.GetCurrent()
        self.mouse_states = {}
        self.cuda_buffers = {}
        self.mouse_coords = {GLUT_LEFT_BUTTON : (0,0)}
        self.frame = 0
        self.timebase = 0
        self.calculated_vis = False
        self.from_files = {'model' : False, 'cudaRes': False}
        self.kernel_step = 8
        self.secs = 0
        self.cuda_stop = False
        self.live = True
        
    def create_events(self):
        self.start = cuda_driver.Event()
        self.end = cuda_driver.Event()
        
    def load_models_from_blender(self):
        self.objs = Blender.Object.GetSelected()
        if self.objs:
            self.verts = numpy.array([]).astype(numpy.float32)
            self.normals = numpy.array([]).astype(numpy.float32)
            self.indexes = numpy.array([]).astype(numpy.ushort)
            index_offset = 0
            for obj in self.objs:
                mesh = obj.getData(mesh=True)
                mesh.quadToTriangle()
                mesh.transform(obj.getMatrix())
                counter = 0
                for face in mesh.faces:
                    self.indexes = numpy.append(self.indexes, range(index_offset + counter,index_offset + counter + 3)).astype(numpy.ushort)
                    counter += 3
                    for v in face.v:
                        self.verts = numpy.append(self.verts, v.co).astype(numpy.float32)
                        self.normals = numpy.append(self.normals, face.no).astype(numpy.float32)
                index_offset = self.indexes.max() + 1
        
    def save_models_data_to_files(self):
        numpy.savetxt("data/verts.dat", self.verts)
        numpy.savetxt("data/normals.dat", self.normals)
        numpy.savetxt("data/indexes.dat", self.indexes)
    
    def save_cuda_res_to_files(self):
        numpy.savetxt("data/vis.dat", self.vis)
        
    def load_cuda_res_from_files(self):
        self.vis = numpy.loadtxt("data/vis.dat").astype(numpy.float32)
        
    def load_models_from_files(self):
        self.verts = numpy.loadtxt("data/verts.dat").astype(numpy.float32)
        self.normals = numpy.loadtxt("data/normals.dat").astype(numpy.float32)
        self.indexes = numpy.loadtxt("data/indexes.dat").astype(numpy.ushort)
    
    def load(self):
        self.load_cam()
        self.design = numpy.loadtxt('data/des3d_240_21.txt').astype(numpy.float32)
        if (self.from_files['model']):
            self.load_models_from_files()
        else:
            self.load_models_from_blender()
            self.save_models_data_to_files()
        if (self.from_files['cudaRes']):
            self.load_cuda_res_from_files()
        else:
            self.vis = numpy.zeros((1, self.verts.size/3), numpy.float32)
    
    def create_glut_window(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.w,self.h)
        glutCreateWindow(self.name)
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION)
        glEnable(GL_DEPTH_TEST)
        #TODO: get bgColor form blender
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse_button)
        glutMotionFunc(self.mouse_move)
        if (load_cuda):
            import pycuda.gl.autoinit
            import pycuda.driver as cuda_driver
            self.create_events()
    
    def mouse_button(self, button, state, x, y):
        self.mouse_states[button] = state
        self.mouse_coords[button] = (x,y)
        
    def mouse_move(self, x, y):
        cam_move_button = GLUT_LEFT_BUTTON
        f = 0.01
        if cam_move_button in self.mouse_states and self.mouse_states[cam_move_button] == GLUT_DOWN:
            mod =  glutGetModifiers()
            dx = self.mouse_coords[cam_move_button][0] - x
            dy = self.mouse_coords[cam_move_button][1] - y
            self.mouse_coords[cam_move_button] = (x,y)
            if mod <> GLUT_ACTIVE_CTRL:
                self.cam['phi'] += f*dx
                self.cam['theta'] += f*dy
            if mod == GLUT_ACTIVE_CTRL:
                self.cam['r'] += f*dy
                if self.cam['r'] < 4: self.cam['r'] = 4
                if self.cam['r'] > 20 : self.cam['r'] = 20
            self.set_cam()
            glutPostRedisplay()
            
    def keyboard(self, key, x, y):
        code = ord(key)
        if (code == 27):#Esc
            glutLeaveMainLoop()
        if (key == ' '):#Space
            self.cuda_stop = not self.cuda_stop

    def reshape(self, width, height):
        self.w = width
        self.h = height
        glutPostRedisplay()
        
    def cuda_from_display(self):
        #TODO: cuda streams ?
        if (not self.calculated_vis and load_cuda and not self.cuda_stop):
            try:
                step = self.kernel_iterator.next()
                self.cuda_map()
                self.cuda_call_kernal_step(step)
                glutSetWindowTitle("%s kernel steps: %d" % (self.name, step))
                self.cuda_unmap()
            except StopIteration:
                self.cuda_print_secs()
                self.cuda_free_memory()
                self.calculated_vis = True
    
    def display(self):
        self.cuda_from_display()
        glViewport(0, 0, self.w, self.h)
        #TODO: do smoothing
        glEnable(GL_POLYGON_SMOOTH)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.put_buffer('aPos', 3)
        #self.put_buffer('aNorm', 3)
        self.put_buffer('aVis')
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers['ind'])
        glDrawElements(GL_TRIANGLES, self.indexes.size, GL_UNSIGNED_SHORT, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        if (self.calculated_vis or self.cuda_stop): self.fps()
        glutSwapBuffers()
        if (not self.calculated_vis): glutPostRedisplay()
    
    def fps(self):
        self.frame += 1;
        time = glutGet(GLUT_ELAPSED_TIME)
        if (time - self.timebase > 1000):
            s = "FPS:%4.2f" % (self.frame*1000/(time - self.timebase))
            self.timebase = time
            self.frame = 0
            s = self.name + " " + s
            glutSetWindowTitle(s)
        
    def put_buffer(self, name, size = 1, type=GL_FLOAT):
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[name])
        glVertexAttribPointer(self.pointers[name], size, type, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
    def make_buffer(self, name, data, target=GL_ARRAY_BUFFER, usage=GL_STATIC_DRAW):
        self.buffers[name] = glGenBuffers(1)
        glBindBuffer(target, self.buffers[name])
        glBufferData(target, data.size*data.itemsize, data, usage)
        glBindBuffer(target, 0)
    
    def make_cuda_buffer(self, name):
        self.cuda_buffers[name] = cuda_gl.BufferObject(long(self.buffers[name]))

    def init_buffers(self):
        self.make_buffer('aVis', self.vis, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW)
        self.make_buffer('aPos', self.verts)
        #self.make_buffer('aNorm', self.normals)
        self.make_buffer('ind', self.indexes, GL_ELEMENT_ARRAY_BUFFER)
        if (not self.calculated_vis):
            self.make_cuda_buffer('aVis')
            #self.make_cuda_buffer('aPos')
            #self.make_cuda_buffer('ind')
        
    def create_shaders(self):
        #TODO: get errors
        self.shaders = {}
        self.shaders['vertex'] = glCreateShader(GL_VERTEX_SHADER)
        self.shaders['fragment'] = glCreateShader(GL_FRAGMENT_SHADER)
        for (name,shader) in self.shaders.iteritems():
            source = open(name + '.glsl').read()
            glShaderSource(shader, source)
            glCompileShader(shader)
        self.program = glCreateProgram()
        glAttachShader(self.program, self.shaders['vertex'])
        glAttachShader(self.program, self.shaders['fragment'])
        glLinkProgram(self.program)
        glUseProgram(self.program)
        #self.program.mvMatrixUniform = glGetUniformLocation(self.program, "uMVMatrix");
        
    def make_pointer(self, name):
        self.pointers[name] = glGetAttribLocation(self.program, name)
        glEnableVertexAttribArray(self.pointers[name])
        
    def init_pointers(self):
        self.make_pointer('aVis')
        self.make_pointer('aPos')
        #self.make_pointer('aNorm')
        
    def load_cam(self):
        self.cam_obj = self.scn.objects.camera
        cam = self.cam_obj.getData()
        self.cam = {}
        matrix = self.cam_obj.getMatrix()
        self.cam['pos'] = pos = matrix[3]
        self.cam['forwards'] = -matrix[2]
        self.cam['target'] = matrix[3] - matrix[2]
        self.cam['up'] = matrix[1]
        self.cam['fov'] = cam.angle
        self.cam['start'] = cam.clipStart
        self.cam['end'] = cam.clipEnd
        #TODO: FIX
        self.cam['r'] = math.sqrt(numpy.dot(pos,pos))
        self.cam['theta'] = math.acos(pos[2]/self.cam['r'])
        self.cam['phi'] = math.atan(pos[1]/pos[0])
    
    def set_cam(self):
        r = self.cam['r']; theta = self.cam['theta']; phi = self.cam['phi']
        self.cam['pos'][0] = r*math.sin(theta)*math.cos(phi)
        self.cam['pos'][1] =  r*math.sin(theta)*math.sin(phi)
        self.cam['pos'][2] = r*math.cos(theta)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.cam['fov'], self.w / self.h, self.cam['start'], self.cam['end']) 
        gluLookAt(self.cam['pos'][0], self.cam['pos'][1], self.cam['pos'][2],\
            0, 0, 0,\
            0, 0, 1)
    
    def set_matrix(self):
        self.set_cam()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
    def cuda_print_secs(self):
        print "cuda time: %fs" % self.secs
        
    #TODO: optim block size and grid size
    def init_cuda(self):
        template_params = {'dN' : self.design.size, 'visN': self.vis.size, 'vN' : self.verts.size, 'kernelStep' : self.kernel_step}
        kernel_code = Template(
            file = 'vis.cu', 
            searchList = [template_params],
          )
        self.cuda_module = SourceModule(kernel_code)
        self.cuda_call = self.cuda_module.get_function("vis")
        self.cuda_call.prepare("P", (256, 1, 1))
        
    def cuda_get_memory(self):
        self.grid_dimensions =  (min(32, (self.vis.size+256-1) // 256 ), 1)
        self.cuda_mem = {}
        #self.cuda_mem['normals_gpu'] = cuda_driver.mem_alloc(self.normals.nbytes)
        #cuda_driver.memcpy_htod(self.cuda_mem['normals_gpu'], self.normals)
        self.cuda_mem['design_gpu'] = self.cuda_module.get_global('design')[0]
        cuda_driver.memcpy_htod(self.cuda_mem['design_gpu'], self.design)
        self.cuda_mem['kernel_n'] = self.cuda_module.get_global('kernelN')[0]
        self.put_data_to_cudatex(self.normals, 'n_tex')
        self.put_data_to_cudatex(self.verts, 'v_tex')
        
    def put_data_to_cudatex(self, data, name):
        if (not name in self.cuda_mem):
            self.cuda_mem[name] = self.cuda_module.get_texref(name)
        self.cuda_mem[name + '_gpu'] = cuda_driver.to_device(data)
        self.cuda_mem[name].set_address(self.cuda_mem[name + '_gpu'], data.nbytes)
        self.cuda_mem[name].set_format(cuda_driver.array_format.FLOAT, 1)
        
    def cuda_map(self):
        self.cuda_mem['map_vis'] = self.cuda_buffers['aVis'].map()
        #self.cuda_mem['map_pos'] = self.cuda_buffers['aPos'].map()
        
    def cuda_unmap(self):
        cuda_driver.Context.synchronize()
        self.cuda_mem['map_vis'].unmap()
        #self.cuda_mem['map_pos'].unmap()
    
    def cuda_free_memory(self):
        pass
        
    def cuda_call_kernal_step(self, step):
        self.start.record()
        cuda_driver.memcpy_htod(self.cuda_mem['kernel_n'],  numpy.array([step]).astype(numpy.int32))
        self.cuda_call.prepared_call(self.grid_dimensions, self.cuda_mem['map_vis'].device_ptr()
            #self.cuda_mem['map_pos'].device_ptr(), self.cuda_mem['normals_gpu']
            )
        self.end.record()
        self.end.synchronize()
        #cuda_driver.Context.synchronize() # ????/???/
        self.secs += self.start.time_till(self.end)*1e-3
        
    def cuda_kernel_iterator(self):
        self.kernel_iterator = iter(range(0, self.design.size, self.kernel_step))
        
    def call_cuda(self):
        self.secs = 0
        for i in range(0, self.design.size, self.kernel_step):
            self.cuda_call_kernal_step(i)
        self.cuda_print_secs()
        
    def run(self):
        self.load()
        self.create_glut_window()
        self.set_matrix()
        self.init_buffers()
        self.create_shaders()
        self.init_pointers()
        if (not self.calculated_vis and load_cuda):
            self.init_cuda()
            self.cuda_kernel_iterator()
            self.cuda_get_memory()
            if (not self.live):
                self.cuda_map()
                self.call_cuda()
                self.cuda_free_memory()
                self.cuda_unmap()
                self.calculated_vis = True
        glutMainLoop()

    
render = CudaRender()
render.run()
