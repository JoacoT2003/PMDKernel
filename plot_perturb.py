"""
plot_perturb.py
Visualiza en OpenGL 3D los imanes del OSI² OpenMRI:
  - Punto en la posición de cada imán.
  - Flecha BLANCA  → momento magnético original.
  - Flecha NARANJA → momento magnético perturbado.

Controles:
  Arrastrar (botón izquierdo) → rotar
  Rueda del ratón            → zoom
  Q / Escape                 → salir

"""

import os, sys, argparse
import math
import ctypes
import numpy as np

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# ─── Parámetros ────────────────────────────────────────────────────────────────
ARROW_LEN   = 14.0   # longitud de las flechas en mm
ARROW_HEAD  = 0.35   # fracción de la flecha que es la cabeza del cono
CONE_R      = 0.4    # radio base del cono (mm)
POINT_SIZE  = 3.0    # tamaño de los puntos de posición

# Colores RGBA
COL_POS      = (0.55, 0.85, 1.00, 1.0)   # azul claro — puntos
COL_ORIG     = (1.00, 1.00, 1.00, 1.0)   # blanco     — momento original
COL_PERTURB  = (1.00, 0.55, 0.05, 1.0)   # naranja    — momento perturbado
COL_BG       = (0.08, 0.08, 0.12, 1.0)   # fondo oscuro


# ─── Carga de datos ─────────────────────────────────────────────────────────────
def load_data(base_dir):
    b0      = np.load(os.path.join(base_dir, "data", "B0.npz"))
    b0p     = np.load(os.path.join(base_dir, "data", "B0_perturbed.npz"))

    pos  = np.hstack([b0["array1"],  b0["array3"]]).T.astype(np.float32)   # [N,3] mm
    mom  = np.hstack([b0["array2"],  b0["array4"]]).T.astype(np.float32)   # [N,3] unit
    momp = np.hstack([b0p["array2"], b0p["array4"]]).T.astype(np.float32)  # [N,3] unit

    return pos, mom, momp


# ─── Geometría: cilindro + cono para representar una flecha ───────────────────
def _rotation_matrix_to_vec(direction):
    """Devuelve la matriz 4x4 (column-major, OpenGL) que rota Z→direction."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, d))
    if sin_a < 1e-8:
        # paralelo o antiparalelo
        if cos_a > 0:
            return np.eye(4, dtype=np.float64)
        else:
            return np.diag([1.0, -1.0, -1.0, 1.0])
    axis = axis / sin_a
    angle_deg = math.degrees(math.atan2(sin_a, cos_a))
    # Rodrigues
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    t = 1 - c
    x, y, z2 = axis
    R = np.array([
        [t*x*x+c,   t*x*y-s*z2, t*x*z2+s*y, 0],
        [t*x*y+s*z2, t*y*y+c,   t*y*z2-s*x, 0],
        [t*x*z2-s*y, t*y*z2+s*x, t*z2*z2+c, 0],
        [0,          0,          0,           1],
    ], dtype=np.float64)
    return R


def draw_arrow(origin, direction, length, color, slices=6):
    """Dibuja una flecha (cilindro + cono) de 'length' mm en la dirección 'direction'."""
    shaft_len = length * (1.0 - ARROW_HEAD)
    head_len  = length * ARROW_HEAD
    shaft_r   = CONE_R * 0.35
    head_r    = CONE_R

    glColor4f(*color)

    glPushMatrix()
    glTranslatef(*origin)
    R = _rotation_matrix_to_vec(direction)
    glMultMatrixd(R.T.flatten())   # OpenGL es column-major

    quad = gluNewQuadric()
    # Cilindro (eje +Z)
    gluCylinder(quad, shaft_r, shaft_r, shaft_len, slices, 1)
    # Tapa base
    gluDisk(quad, 0, shaft_r, slices, 1)
    # Cono
    glTranslatef(0, 0, shaft_len)
    gluCylinder(quad, head_r, 0, head_len, slices, 1)
    gluDisk(quad, 0, head_r, slices, 1)
    gluDeleteQuadric(quad)

    glPopMatrix()


def draw_text_3d(x, y, z, text, color=(1.0, 1.0, 1.0)):
    """Dibuja 'text' en la posición 3D (x,y,z) usando fuente bitmap de GLUT."""
    glColor3f(*color)
    glRasterPos3f(x, y, z)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))


def draw_point(pos, color):
    glColor4f(*color)
    glBegin(GL_POINTS)
    glVertex3f(*pos)
    glEnd()


# ─── Construcción de la escena en display lists ────────────────────────────────
def build_scene(pos, mom, momp):
    """Compila toda la geometría en una display list de OpenGL."""
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)

    glPointSize(POINT_SIZE)
    for i in range(len(pos)):
        draw_point(pos[i], COL_POS)
        draw_arrow(pos[i], mom[i],  ARROW_LEN, COL_ORIG)
        draw_arrow(pos[i], momp[i], ARROW_LEN, COL_PERTURB)

    glEndList()
    return dl


# ─── Estado de la cámara / interacción ────────────────────────────────────────
class Camera:
    def __init__(self):
        self.yaw    = 30.0
        self.pitch  = 20.0
        self.dist   = 600.0
        self.pan_x  = 0.0
        self.pan_y  = 0.0
        self._last_x  = None
        self._last_y  = None
        self._drag    = False   # botón izquierdo — rotar
        self._panning = False   # botón derecho   — pan

    def mouse_button(self, btn, action):
        if btn == glfw.MOUSE_BUTTON_LEFT:
            self._drag = (action == glfw.PRESS)
            if not self._drag:
                self._last_x = self._last_y = None
        elif btn == glfw.MOUSE_BUTTON_RIGHT:
            self._panning = (action == glfw.PRESS)
            if not self._panning:
                self._last_x = self._last_y = None

    def mouse_move(self, x, y):
        if not self._drag and not self._panning:
            return
        if self._last_x is not None:
            dx = x - self._last_x
            dy = y - self._last_y
            if self._drag:
                self.yaw   += dx * 0.4
                self.pitch += dy * 0.4
                self.pitch  = max(-89.0, min(89.0, self.pitch))
            elif self._panning:
                # escalar el pan en función de la distancia para que sea proporcional
                scale = self.dist * 0.001
                self.pan_x += dx * scale
                self.pan_y -= dy * scale   # Y invertido respecto a pantalla
        self._last_x, self._last_y = x, y

    def scroll(self, dy):
        self.dist *= 0.9 ** dy
        self.dist  = max(50.0, min(3000.0, self.dist))

    def apply(self):
        glTranslatef(0, 0, -self.dist)
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.yaw,   0, 1, 0)
        glTranslatef(self.pan_x, self.pan_y, 0)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualizador 3D de perturbaciones de imanes")
    parser.add_argument("--base", default=None,
                        help="Ruta raíz del proyecto (default: detectada automáticamente)")
    args = parser.parse_args()

    # Detectar directorio base
    if args.base:
        base_dir = args.base
    else:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = this_dir

    base_dir = os.path.normpath(base_dir)
    print(f"Cargando datos desde: {base_dir}")

    pos, mom, momp = load_data(base_dir)
    center = pos.mean(axis=0)
    pos_centered = pos - center
    print(f"  {len(pos)} imanes cargados. Centro de masa: {center}")

    # ── GLFW / OpenGL ─────────────────────────────────────────────────────────
    glutInit()

    if not glfw.init():
        sys.exit("No se pudo inicializar GLFW")

    glfw.window_hint(glfw.SAMPLES, 4)
    win = glfw.create_window(1280, 800, "OSI² — Perturbaciones de momentos magnéticos", None, None)
    if not win:
        glfw.terminate()
        sys.exit("No se pudo crear la ventana")

    glfw.make_context_current(win)

    cam = Camera()

    def on_mouse_button(w, btn, action, mods):
        cam.mouse_button(btn, action)

    def on_cursor_pos(w, x, y):
        cam.mouse_move(x, y)

    def on_scroll(w, dx, dy):
        cam.scroll(dy)

    def on_key(w, key, sc, action, mods):
        if action == glfw.PRESS:
            if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                glfw.set_window_should_close(w, True)
            elif key == glfw.KEY_R:
                cam.yaw, cam.pitch, cam.dist = 30.0, 20.0, 600.0
                cam.pan_x, cam.pan_y = 0.0, 0.0

    glfw.set_mouse_button_callback(win, on_mouse_button)
    glfw.set_cursor_pos_callback(win,   on_cursor_pos)
    glfw.set_scroll_callback(win,       on_scroll)
    glfw.set_key_callback(win,          on_key)

    # ── OpenGL setup ──────────────────────────────────────────────────────────
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glShadeModel(GL_SMOOTH)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 2.0, 3.0, 0.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.3, 0.3, 0.3, 1.0])
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    print("Compilando display list…", end=" ", flush=True)
    scene_dl = build_scene(pos_centered, mom, momp)
    print("listo.")

    print("Ventana abierta. Arrastra para rotar, rueda para zoom, Q/Esc para salir.")
    print("  BLANCO  = momento original")
    print("  NARANJA = momento perturbado")

    # ── Bucle de renderizado ──────────────────────────────────────────────────
    while not glfw.window_should_close(win):
        w, h = glfw.get_framebuffer_size(win)
        glViewport(0, 0, w, h)

        glClearColor(*COL_BG)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Proyección
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 1.0, 5000.0)

        # Vista
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        cam.apply()

        # Ejes de referencia (desactivar iluminación para líneas planas)
        glDisable(GL_LIGHTING)
        axis_len = 200.0
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0,0,0); glVertex3f(axis_len,0,0)
        glColor3f(0, 1, 0); glVertex3f(0,0,0); glVertex3f(0,axis_len,0)
        glColor3f(0, 0, 1); glVertex3f(0,0,0); glVertex3f(0,0,axis_len)
        glEnd()

        # Etiquetas de ejes
        offset = axis_len * 0.08   # pequeño desplazamiento para no tapar la punta
        draw_text_3d(axis_len + offset, 0, 0, "X", color=(1.0, 0.3, 0.3))
        draw_text_3d(0, axis_len + offset, 0, "Y", color=(0.3, 1.0, 0.3))
        draw_text_3d(0, 0, axis_len + offset, "Z", color=(0.4, 0.6, 1.0))

        glEnable(GL_LIGHTING)

        # Escena
        glCallList(scene_dl)

        glfw.swap_buffers(win)
        glfw.poll_events()

    glDeleteLists(scene_dl, 1)
    glfw.destroy_window(win)
    glfw.terminate()


if __name__ == "__main__":
    main()
