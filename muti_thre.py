import threading
import time
from tkinter import *
import pyglet
from pyglet.window import key
from pyglet.window import mouse
from pyglet.gl import *
import numpy as np
import numpy.linalg as linalg
import noise
import math



class Element:
    # 对三角形单元进行节点编码
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        # print(self.p1 , self.p2 , self.p3)  #debug

    # 定义elem_array函数，返回的是一个单元节点编码组成的行向量
    def elem_array(self):
        return [self.p1, self.p2, self.p3]

class Constraint:
    def __init__(self, vert_a, fixed_point=False):
        self.vert_a = vert_a
        self.fixed_point = fixed_point

class Cloth:
    def __init__(self, verts, elements, uvs = [], constraints=[]):
        self.n = len(verts)
        self.verts = verts
        self.q_n = np.copy(verts)
        self.elements = elements
        self.uvs = uvs
        self.stepsize = 0.3    #!!! stepsize,论文中的h
        self.solver_iteration_times = 3  # 迭代次数
        self.velocities = np.zeros((self.n, 3))    # n*3速度阵
        self.mass_matrix = np.identity(self.n) / (len(self.elements))  # n*n集中质量阵
        self.constraints = constraints    # constraints里包含了所有固定端的点的Constraint函数
        self.fixed_points = []
        for con in self.constraints:
            self.fixed_points.append(con)
        self.damping_ratio = 0.3 # 阻尼比
        self.wind_magnitude = 4  # 控制风荷载的大小
        self.concentrated_force_is_on = 0  # 用1和0控制集中力是否施加
        self.concentrated_force_magnitude = 100  # 控制集中荷载的大小
        self.gravity_magnitude = 9  # 控制重力的大小
        self.fixed_points_displacement_factor = 1  # 用系数和坐标相乘得固定端的变化后的坐标，相当于固定端发生位移
        self.svd_clip_factor = 0.03  # 裁剪范围,调整变形
        self.solve_count = 0  # debug
        self.iter_count = 0  # debug

    # 用x_g和x_f来计算三角形单元变形能，公式16
    def potential_for_triangle(self, element, q_n_plus, point):
        r_m = set([element.p1, element.p2, element.p3])
        r_m.remove(point)
        other_points = list(r_m)

        q1 = q_n_plus[point]
        q2 = q_n_plus[other_points[0]]
        q3 = q_n_plus[other_points[1]]
        x_f = np.matrix((q3 - q1, q2 - q1, (0, 0, 0))).T  # .T表示转置
        # print (x_f)  # debug

        q1 = self.verts[point]
        q2 = self.verts[other_points[0]]
        q3 = self.verts[other_points[1]]
        x_g = np.matrix((q3 - q1, q2 - q1, (0, 0, 0))).T

        # 奇异值分解为U,s,V_t三个矩阵，返回点积
        U, s, V_t = np.linalg.svd(x_f.dot(np.linalg.pinv(x_g)))
        # print(s)  # debug
        s = np.clip(s, 1-self.svd_clip_factor, 1+3*self.svd_clip_factor)  # 裁剪
        # print(s)  # debug
        s = np.diag(s)
        # return np.around(U.dot(s).dot(V_t), 10)  # 四舍五入
        return U.dot(s).dot(V_t)  # 四舍五入

     # 力在求解前叠加,位移不可叠加
    def forces(self, time):

        # 风荷载，使用perlins noise来模拟风荷载
        time /= 100
        elem_wind_force = np.zeros((self.n, 3))
        wind_force = np.zeros((self.n, 3))
        angle = noise.pnoise1(time) * math.pi * 0.5
        elem_wind_force[:, 0] = 0.6*math.cos(angle)
        elem_wind_force[:, 1] = -math.sin(angle)
        elem_wind_force[:, 2] = 0.3*math.sin(angle)
        wind_force[:, :] = elem_wind_force * self.wind_magnitude * (noise.pnoise1(time) + 0.2) * (noise.pnoise1(noise.pnoise1(time)) * 0.5 + 0.5)

        # 集中力，暂时先作用在布料的中心上，以后再改
        concentrated_force = np.zeros((self.n, 3))
        if np.sqrt(self.n)%2 == 0:
            concentrated_force[int((self.n)/2)-1+int(np.sqrt(self.n)/2), 2] = -self.concentrated_force_magnitude
            concentrated_force *= self.concentrated_force_is_on
        else:
            concentrated_force[int((self.n)/2), 2] = -self.concentrated_force_magnitude
            concentrated_force *= self.concentrated_force_is_on

        # 重力荷载
        gravity_force = np.zeros((self.n, 3))
        gravity_force[:, 1] += -self.gravity_magnitude/3

        return wind_force + concentrated_force + gravity_force

    # local+global
    def simulate(self):
        self.solve_count += 1
        # print("Solve count:", self.solve_count)  # debug
        forces = self.forces(self.solve_count)
####### Algorithm 1:
        s_n = self.q_n + self.velocities * self.stepsize + (self.stepsize * self.stepsize) * linalg.inv(self.mass_matrix).dot(forces) # 公式4:sn
        q_n_plus = np.copy(s_n)
        b = np.zeros((self.n + len(self.fixed_points), 3))

        # 开始迭代
        for iter in range(self.solver_iteration_times):
            self.iter_count += 1
            b[:self.n] = self.mass_matrix.dot(s_n) / (self.stepsize * self.stepsize)  # b的前n行=[M]/h^2*s(n)
            for element in self.elements:
                elem_array = element.elem_array()
                for i in range(3):
                    p1 = elem_array[i]
                    p2 = elem_array[(i + 1) % 3] # 0和1,1和2,2和3%3，防止超出！！
                    p_f_t = self.potential_for_triangle(element, q_n_plus, p2)
                    g = p_f_t.dot(self.verts[p2] - self.verts[p1])
                    b[p1] = b[p1] - g
                    b[p2] = b[p2] + g

            # 集装固定点
            for con_i in range(len(self.fixed_points)):
                con = self.fixed_points[con_i]
                b[-(con_i + 1)] = self.verts[con.vert_a]

            q_n_plus = np.linalg.solve(self.global_matrix(), b)  # global_matrix * q_n_plus = b
            q_n_plus = q_n_plus[:-len(self.fixed_points), :]
        # 迭代结束
        # print(q_n_plus) # debug
        self.velocities = ((q_n_plus - self.q_n)*( 1-self.damping_ratio))/ self.stepsize
        self.q_n = q_n_plus

    # global step
    def global_matrix(self):
        fixed_point_num = len(self.fixed_points)
        M = np.zeros((self.n + fixed_point_num, self.n + fixed_point_num))
        M[:self.n, :self.n] = self.mass_matrix # M：质量矩阵

        weights = np.zeros((self.n + fixed_point_num, self.n + fixed_point_num))
        for element in self.elements:
            verts = element.elem_array()
            for k in range(3):
                v_1 = verts[k]
                v_2 = verts[(k + 1) % 3]
                weights[v_1, v_2] -= 1
                weights[v_2, v_1] -= 1
                weights[v_1, v_1] += 1
                weights[v_2, v_2] += 1

        M /= (self.stepsize * self.stepsize)
        for i in range(fixed_point_num):
            con = self.fixed_points[i]
            weights[con.vert_a, -(i + 1)] = 1/self.fixed_points_displacement_factor
            weights[-(i + 1), con.vert_a] = 1/self.fixed_points_displacement_factor
            M[-(i + 1), -(i + 1)] = 0

        return M + weights

    # 生成
    def generate(width, height, MAX_WIDTH_SIZE=300, MAX_HEIGHT_SIZE=300):
        n = width * height
        elem_width = MAX_WIDTH_SIZE / width  # 单元宽度
        elem_height = -MAX_HEIGHT_SIZE / height  # 单元高度
        # verts(n行3列)表示n个节点的“模型坐标”!!!重点
        verts = np.zeros((n, 3))
        elements = []
        constraints = []
        # uvs(n行2列)表示n个节点的“相对于模型的坐标”，范围为[0,1]
        uvs = np.zeros((n, 2))
        for x in range(width):
            for y in range(height):
                verts[ x + (y * width) ] = np.array((x * elem_width, y * elem_height, 0 ))
                # 第x + (y * width)行 = 向量((x % width) / width, 1 - (y % height) / height)
                uvs[ x + (y * width) ]  = np.array(( (x % width) / width, 1 - (y % height) / height ))

        # 生成三角形单元
        for p_id in range(n):
            # 最右侧那一列节点跳过
            if p_id % width == width - 1:
                continue
            # 第一行至倒数第二行的上三角形
            if p_id < n - width:
                v_1 = p_id
                v_2 = p_id + width
                v_3 = p_id + 1
                elements.append(Element(v_1, v_2, v_3))
            # 第二行至最后一行的下三角形
            if p_id >= width:
                v_1 = p_id
                v_2 = p_id + 1
                v_3 = p_id - (width - 1)
                elements.append(Element(v_1, v_2, v_3))

        # 固定形式：固定全部的上边点
        for i in range(width):
            constraints.append(Constraint(vert_a = i, fixed_point=verts[i]))

        return Cloth(verts, elements, uvs, constraints=constraints)

class Draw:

    DRAW_LINES = True  # 是否绘制frame线
    dt = 0.033  # 更新时间

    def __init__(self):
        self.size = 13
        self.r_x = -30
        self.r_y = 0
        self.z = -1000
        self.y = 180
        self.x = -100
        self.cloth_set = []

        self.cloth_set.append(Cloth.generate(self.size, self.size))
        self.count = 0  # debug

    def main(self):
        self.window = pyglet.window.Window(800, 600)

        @self.window.event
        def on_draw():
            glClearColor(92 / 256, 131 / 256, 178 / 256, 1)  # 背景色
            glClear(GL_COLOR_BUFFER_BIT)

            self.setup_camera()

            glTranslatef(self.x, self.y, self.z)
            glRotatef(self.r_x, 0, 1, 0)
            glRotatef(self.r_y, 1, 0, 0)

            glPushMatrix()
            glTranslatef(0, -75 * (len(self.cloth_set) - 1), 0)
            self.draw_object(self.cloth_set[0])
            glTranslatef(0, 300, 0)
            glPopMatrix()

            self.draw_ground()

        # 鼠标右键拖动控制平移，中键拖动控制旋转
        self.window.on_mouse_drag = self.on_mouse_drag
        # 滚轮上下滚动控制视图缩放
        self.window.on_mouse_scroll = self.on_mouse_scroll
        # 鼠标左键点击与放开来施加集中荷载
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_mouse_release = self.on_mouse_release
        pyglet.clock.schedule_interval(self.update, self.dt)
        self.setup()
        glPushMatrix()
        textures = []
        self.load_texture("fg_image.png", textures)
        pyglet.app.run()
        return

    def update(self, dt):
        self.cloth_set[0].simulate()

    # 鼠标右键拖动控制平移，中键拖动控制旋转
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if (buttons & mouse.RIGHT):
            self.x += dx
            self.y += dy
        if (buttons & mouse.MIDDLE):
            self.r_x += dx
            # self.r_y -= dy

    # 滚轮上下滚动控制视图缩放
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.z += 200 * scroll_y

    # 用鼠标左键点击产生集中荷载
    def on_mouse_press(self, x, y, button, modifiers):
        if (button & mouse.LEFT):
            self.cloth_set[0].concentrated_force_is_on = 1
            print("{0} pressed at: {1},{2}.".format("LEFT_BUTTON", x, y))
            print("Concentrated force is on")

    # 松开鼠标左键集中荷载消失
    def on_mouse_release(self, x, y, button, modifiers):
        if (button & mouse.LEFT):
            self.cloth_set[0].concentrated_force_is_on = 0
            print("Concentrated force is off")

    def setup_camera(self):
        glMatrixMode(gl.GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(30, self.window.width / self.window.height, 0.1, 10000)
        glMatrixMode(gl.GL_MODELVIEW)
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def draw_object(self, cloth):
        # 绘制单元的分割线
        if self.DRAW_LINES:
            glLineWidth(1)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glBegin(GL_TRIANGLES)
            for element in cloth.elements:
                for p_id in element.elem_array():
                    vert = cloth.q_n[p_id]
                    glColor3f(1, 0, 0)
                    glVertex3f(vert[0], vert[1], vert[2])
            glEnd()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBegin(GL_TRIANGLES)
        for element in cloth.elements:
            for p_id in element.elem_array():
                vert = cloth.q_n[p_id]
                uv = cloth.uvs[p_id]
                glColor3f(1, 1, 1)
                glTexCoord2f(uv[0], uv[1])
                glVertex3f(vert[0], vert[1], vert[2])
        glEnd()

    def draw_line(self, p1, p2):
        glBegin(GL_LINE_STRIP)
        glColor3f(0.8, 0.8, 0.8)
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glEnd()

    def draw_ground(self):
        p1 = [-10000, -450, -10000]
        p2 = [-10000, -450, 10000]
        for i in range(200):
            self.draw_line(p1, p2)
            p1[0] = p1[0] + 100
            p2[0] = p2[0] + 100

        p1 = [-10000, -450, -10000]
        p2 = [10000, -450, -10000]
        for i in range(200):
            self.draw_line(p1, p2)
            p1[2] = p1[2] + 100
            p2[2] = p2[2] + 100

    def setup(self):
        glMatrixMode(gl.GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.window.width / self.window.height, -100, 100)
        glMatrixMode(gl.GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(-0.7, -0.3, -1.5)
        glEnable(GL_DEPTH_TEST)

    def load_texture(self, filename, textures):
        image = pyglet.image.load(filename)
        textures.append(image.get_texture())
        glEnable(textures[-1].target)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE,
                     image.get_image_data().get_data('RGBA',
                                                     image.width * 4))


############### 子程序：control
def target():
    print ('the curent threading "control" (%s) is running' % threading.current_thread().name)

    root = Tk()
    lock = threading.Lock()

    def get_frame_line():
        lock.acquire()
        draw_instance.DRAW_LINES = not draw_instance.DRAW_LINES
        lock.release()

    def get_size(i):
        lock.acquire()
        draw_instance.size = int(i)  # 不能像之前那样，那样拿到的是Draw的实例化！
        lock.release()

    def get_fixed_points_displacement_factor(i):
        lock.acquire()
        draw_instance.cloth_set[0].fixed_points_displacement_factor = float(i)
        lock.release()

    def get_svd_clip_factor(i):
        lock.acquire()
        draw_instance.cloth_set[0].svd_clip_factor = float(i)
        lock.release()

    def get_concentrated_force_magnitude(i):
        lock.acquire()
        draw_instance.cloth_set[0].concentrated_force_magnitude = int(i)
        lock.release()

    def get_wind_magnitude(i):
        lock.acquire()
        draw_instance.cloth_set[0].wind_magnitude = int(i)
        lock.release()
        # print (draw_instance.cloth_set[0].wind_magnitude, i)  # debug

    def get_gravity_magnitude(i):
        lock.acquire()
        draw_instance.cloth_set[0].gravity_magnitude = int(i)
        lock.release()

    def get_damping_ratio(i):
        lock.acquire()
        draw_instance.cloth_set[0].damping_ratio = float(i)
        lock.release()

    def get_stepsize(i):
        lock.acquire()
        draw_instance.cloth_set[0].stepsize = float(i)
        lock.release()

    def get_dt(i):
        lock.acquire()
        draw_instance.dt = float(i)
        lock.release()

    def get_solver_iteration_times(i):
        lock.acquire()
        draw_instance.cloth_set[0].solver_iteration_times = int(i)
        lock.release()

    frame_line_checkb = Checkbutton(text = "Frame line",command = get_frame_line, width = 20, height=3,)
    frame_line_checkb.select()
    frame_line_checkb.pack()

    size_scale = Scale(root,
                                 from_=1,  # 最小值
                                 to=12,  # 最大值
                                 resolution=1,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Size',  # 标签
                                 command=get_size  # 回调函数
                                 )
    size_scale.set(9)
    size_scale.pack()

    fixed_points_displacement_factor_scale = Scale(root,
                                 from_=0.1,  # 最小值
                                 to=10,  # 最大值
                                 resolution=0.001,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Fixed points displacement factor',  # 标签
                                 command=get_fixed_points_displacement_factor  # 回调函数
                                 )
    fixed_points_displacement_factor_scale.set(1)
    fixed_points_displacement_factor_scale.pack()

    svd_clip_factor_scale = Scale(root,
                                 from_=0.01,  # 最小值
                                 to=1,  # 最大值
                                 resolution=0.01,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Svd clip factor',  # 标签
                                 command=get_svd_clip_factor  # 回调函数
                                 )
    svd_clip_factor_scale.set(0.03)
    svd_clip_factor_scale.pack()

    concentrated_force_magnitude_scale = Scale(root,
                                 from_=0,  # 最小值
                                 to=200,  # 最大值
                                 resolution=1,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Concentrated force magnitude',  # 标签
                                 command=get_concentrated_force_magnitude  # 回调函数
                                 )
    concentrated_force_magnitude_scale.set(100)
    concentrated_force_magnitude_scale.pack()

    wind_magnitude_scale = Scale(root,
                                 from_=-64,  # 最小值
                                 to=64,  # 最大值
                                 resolution=1,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Wind magnitude',  # 标签
                                 command=get_wind_magnitude  # 回调函数
                                 )
    wind_magnitude_scale.set(4)
    wind_magnitude_scale.pack()

    gravity_magnitude_scale = Scale(root,
                                 from_=0,  # 最小值
                                 to=64,  # 最大值
                                 resolution=1,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Gravity magnitude',  # 标签
                                 command=get_gravity_magnitude  # 回调函数
                                 )
    gravity_magnitude_scale.set(9)
    gravity_magnitude_scale.pack()

    damping_ratio_scale = Scale(root,
                                 from_=0,  # 最小值
                                 to=0.8,  # 最大值
                                 resolution=0.1,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Damping ratio',  # 标签
                                 command=get_damping_ratio  # 回调函数
                                 )
    damping_ratio_scale.set(0.3)
    damping_ratio_scale.pack()

    stepsize_scale = Scale(root,
                             from_=0.05,  # 最小值
                             to=1,  # 最大值
                             resolution=0.05,  # 步距值
                             orient=HORIZONTAL,  # 水平的
                             length=200,
                             width=15,
                             sliderlength=15,
                             label='Stepsize',  # 标签
                             command=get_stepsize  # 回调函数
                             )
    stepsize_scale.set(0.3)
    stepsize_scale.pack()

    dt_scale = Scale(root,
                             from_=0.001,  # 最小值
                             to=1,  # 最大值
                             resolution=0.001,  # 步距值
                             orient=HORIZONTAL,  # 水平的
                             length=200,
                             width=15,
                             sliderlength=15,
                             label='Update time',  # 标签
                             command=get_dt  # 回调函数
                             )
    dt_scale.set(0.033)
    dt_scale.pack()

    solver_iteration_times_scale = Scale(root,
                                 from_=1,  # 最小值
                                 to=15,  # 最大值
                                 resolution=1,  # 步距值
                                 orient=HORIZONTAL,  # 水平的
                                 length=200,
                                 width=15,
                                 sliderlength=15,
                                 label='Solver iteration times',  # 标签
                                 command=get_solver_iteration_times  # 回调函数
                                 )
    solver_iteration_times_scale.set(3)
    solver_iteration_times_scale.pack()

#####text

    # text.insert(INSERT, ["Information:",str(draw_instance.cloth_set[0].solve_count),"\n"])

    mainloop()

    time.sleep(1)
    print ('the curent threading "control" (%s) is ended' % threading.current_thread().name)



############### 主程序：Draw
print('the curent threading  "cloth" (%s) is running' % threading.current_thread().name)
t = threading.Thread(target=target)
t.start()

draw_instance = Draw()
draw_instance.main()

t.join()
print ('the curent threading  "cloth" (%s) is ended' % threading.current_thread().name)