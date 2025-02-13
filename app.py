import numpy as np
from scipy.integrate import solve_ivp
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Slider, PointDrawTool, TextInput, Div, Arrow, VeeHead
from bokeh.io import curdoc

# Función que define el sistema de ecuaciones diferenciales
def system(t, x, A):
    return A @ x

# Función para resolver el sistema y obtener las soluciones
def solve_system(a, b, c, d, x1_0, x2_0):
    A = np.array([[a, b],
                  [c, d]])
    x0 = np.array([x1_0, x2_0])
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 300)
    sol = solve_ivp(system, t_span, x0, t_eval=t_eval, args=(A,))
    return sol

# Función para verificar valores propios
def check_eigenvalues(a, b, c, d):
    A = np.array([[a, b],
                  [c, d]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Considerar como 0 si están en el rango [-1e-8, 1e-8]
    eigenvalues = [0 if -1e-8 < eig < 1e-8 else eig for eig in eigenvalues]
    return eigenvalues, eigenvectors

# Crear datos iniciales
initial_a = 1
initial_b = 1
initial_c = 1
initial_d = 1
initial_x1 = 1
initial_x2 = 1
sol = solve_system(initial_a, initial_b, initial_c, initial_d, initial_x1, initial_x2)

# Fuente de datos para los gráficos
source = ColumnDataSource(data={
    't': sol.t,
    'x1': sol.y[0],
    'x2': sol.y[1],
    'a': [initial_a] * len(sol.t),
    'b': [initial_b] * len(sol.t),
    'c': [initial_c] * len(sol.t),
    'd': [initial_d] * len(sol.t),
    'tr': [initial_a + initial_d] * len(sol.t),  # Traza
    'det': [initial_a * initial_d - initial_b * initial_c] * len(sol.t)  # Determinante
})

# Gráfico de las soluciones temporales
time_plot = figure(width=400, height=300, title="Soluciones x1(t) y x2(t)", 
                   x_axis_label='Tiempo', y_axis_label='x(t)')
time_plot.line('t', 'x1', source=source, legend_label='x1(t)', line_width=2, color="blue")
time_plot.line('t', 'x2', source=source, legend_label='x2(t)', line_width=2, color="orange")

# Gráfico del retrato de fase
phase_plot = figure(width=400, height=300, title="Retrato de Fase", 
                    x_axis_label='x1', y_axis_label='x2')
phase_plot.line('x1', 'x2', source=source, line_width=2, color="green")

# Gráfico de la relación entre traza y determinante
tr_det_plot = figure(width=400, height=300, title="Relación entre Traza y Determinante", 
                     x_axis_label='Traza (tr)', y_axis_label='Determinante (det)')
tr_det_plot.line(x=np.linspace(-10, 10, 400), y=(np.linspace(-10, 10, 400)**2) / 4, 
                 line_width=2, color="red", legend_label="det = (tr^2)/4")

# Punto en el gráfico de la traza/determinante
tr_det_point = tr_det_plot.scatter('tr', 'det', source=source, size=10, color="blue", 
                                   legend_label="Punto (tr, det)")

# Herramienta de arrastre del punto
draw_tool = PointDrawTool(renderers=[tr_det_point])
tr_det_plot.add_tools(draw_tool)

# Función para actualizar los gráficos cuando se mueve el punto
def update_from_point(attr, old_data, new_data):
    tr_new = new_data['geometry']['x']
    det_new = new_data['geometry']['y']
    # Calcular nuevos valores de a y d basados en tr y det
    # tr = a + d, det = a * d - b * c
    # Resolver el sistema de ecuaciones para a y d
    b = source.data['b'][0]
    c = source.data['c'][0]
    # Ecuación cuadrática: a^2 - tr * a + (det + b * c) = 0
    discriminant = tr_new**2 - 4 * (det_new + b * c)
    if discriminant < 0:
        return  # No hay solución real, no actualizar
    a_new = (tr_new + np.sqrt(discriminant)) / 2
    d_new = tr_new - a_new
    # Actualizar los valores de a y d
    source.data['a'] = [a_new] * len(source.data['t'])
    source.data['d'] = [d_new] * len(source.data['t'])
    # Resolver el sistema con los nuevos valores
    sol = solve_system(a_new, b, c, d_new, float(x1_input.value), float(x2_input.value))
    source.data = {
        't': sol.t,
        'x1': sol.y[0],
        'x2': sol.y[1],
        'a': [a_new] * len(sol.t),
        'b': [b] * len(sol.t),
        'c': [c] * len(sol.t),
        'd': [d_new] * len(sol.t),
        'tr': [tr_new] * len(sol.t),
        'det': [det_new] * len(sol.t)
    }
    a_slider.value = a_new
    d_slider.value = d_new
    a_input.value = str(a_new)
    d_input.value = str(d_new)
    update_matrix_equation()
    

# Conectar el evento de mover el punto al callback
tr_det_point.data_source.on_change('selected', update_from_point)

# Sliders para la matriz A
a_slider = Slider(start=-10, end=10, value=initial_a, step=0.1, title="a", width=150)
b_slider = Slider(start=-10, end=10, value=initial_b, step=0.1, title="b", width=150)
c_slider = Slider(start=-10, end=10, value=initial_c, step=0.1, title="c", width=150)
d_slider = Slider(start=-10, end=10, value=initial_d, step=0.1, title="d", width=150)

# Entradas de texto para los valores de a, b, c, d
a_input = TextInput(value=str(initial_a), title="a:", width=75)
b_input = TextInput(value=str(initial_b), title="b:", width=75)
c_input = TextInput(value=str(initial_c), title="c:", width=75)
d_input = TextInput(value=str(initial_d), title="d:", width=75)

# Entradas de texto para las condiciones iniciales
x1_input = TextInput(value=str(initial_x1), title="Condición inicial x1:", width=150)
x2_input = TextInput(value=str(initial_x2), title="Condición inicial x2:", width=150)

# Función para actualizar la ecuación matricial en LaTeX
def update_matrix_equation():
    a = float(a_input.value)
    b = float(b_input.value)
    c = float(c_input.value)
    d = float(d_input.value)
    tr = a + d
    det = a * d - b * c
    eigenvalues, eigenvectors = check_eigenvalues(a, b, c, d)
    # Limitar a 4 decimales
    a_rounded = round(a, 4)
    b_rounded = round(b, 4)
    c_rounded = round(c, 4)
    d_rounded = round(d, 4)
    tr_rounded = round(tr, 4)
    det_rounded = round(det, 4)
    eigenvalues_rounded = [round(ev, 4) for ev in eigenvalues]
    eigenvectors_rounded = [[round(ev, 4) for ev in vec] for vec in eigenvectors.T]
    matrix_equation.text = fr"""
    <h3 style='text-align: center;'>Ecuación:</h3>
    <div style='text-align: center;'>
        \(
        \begin{{bmatrix}} \dot{{x}} \\ \dot{{y}} \end{{bmatrix}} =
        \begin{{bmatrix}} {a_rounded} & {b_rounded} \\ {c_rounded} & {d_rounded} \end{{bmatrix}}
        \begin{{bmatrix}} x \\ y \end{{bmatrix}}
        \)
    </div>
    <h3 style='text-align: center;'>Valores Propios:</h3>
    <div style='text-align: center;'>
        \(
        \lambda_1 = {eigenvalues_rounded[0]}, \quad \lambda_2 = {eigenvalues_rounded[1]}
        \)
    </div>
    <h3 style='text-align: center;'>Vectores Propios:</h3>
    <div style='text-align: center;'>
        \(
        v_1 = \begin{{bmatrix}} {eigenvectors_rounded[0][0]} \\ {eigenvectors_rounded[1][0]} \end{{bmatrix}}, \quad
        v_2 = \begin{{bmatrix}} {eigenvectors_rounded[0][1]} \\ {eigenvectors_rounded[1][1]} \end{{bmatrix}}
        \)
    </div>
    <h3 style='text-align: center;'>Traza y Determinante:</h3>
    <div style='text-align: center;'>
        \(
        \text{{Traza}} = {tr_rounded}, \quad \text{{Determinante}} = {det_rounded}
        \)
    </div>
    """

# Función para actualizar los datos cuando se cambian los valores en los inputs de texto
def update_from_input(attrname, old, new):
    a = float(a_input.value)
    b = float(b_input.value)
    c = float(c_input.value)
    d = float(d_input.value)
    x1 = float(x1_input.value)
    x2 = float(x2_input.value)
    sol = solve_system(a, b, c, d, x1, x2)
    tr = a + d
    det = a * d - b * c
    source.data = {
        't': sol.t,
        'x1': sol.y[0],
        'x2': sol.y[1],
        'a': [a] * len(sol.t),
        'b': [b] * len(sol.t),
        'c': [c] * len(sol.t),
        'd': [d] * len(sol.t),
        'tr': [tr] * len(sol.t),
        'det': [det] * len(sol.t)
    }
    a_slider.value = a
    b_slider.value = b
    c_slider.value = c
    d_slider.value = d
    update_matrix_equation()

# Vincular las entradas de texto a la función de actualización
a_input.on_change('value', update_from_input)
b_input.on_change('value', update_from_input)
c_input.on_change('value', update_from_input)
d_input.on_change('value', update_from_input)
x1_input.on_change('value', update_from_input)
x2_input.on_change('value', update_from_input)

# Función para actualizar los datos cuando se mueven los sliders
def update_from_slider(attrname, old, new):
    a = a_slider.value
    b = b_slider.value
    c = c_slider.value
    d = d_slider.value
    x1 = float(x1_input.value)
    x2 = float(x2_input.value)
    sol = solve_system(a, b, c, d, x1, x2)
    tr = a + d
    det = a * d - b * c
    source.data = {
        't': sol.t,
        'x1': sol.y[0],
        'x2': sol.y[1],
        'a': [a] * len(sol.t),
        'b': [b] * len(sol.t),
        'c': [c] * len(sol.t),
        'd': [d] * len(sol.t),
        'tr': [tr] * len(sol.t),
        'det': [det] * len(sol.t)
    }
    a_input.value = str(a)
    b_input.value = str(b)
    c_input.value = str(c)
    d_input.value = str(d)
    update_matrix_equation()

# Vincular los sliders a la función de actualización
a_slider.on_change('value', update_from_slider)
b_slider.on_change('value', update_from_slider)
c_slider.on_change('value', update_from_slider)
d_slider.on_change('value', update_from_slider)

# Controles de la matriz A
a_group = column(a_input, a_slider, width=150, sizing_mode="fixed")
b_group = column(b_input, b_slider, width=150, sizing_mode="fixed")
c_group = column(c_input, c_slider, width=150, sizing_mode="fixed")
d_group = column(d_input, d_slider, width=150, sizing_mode="fixed")

# Matriz 2x2 de controles
controls_matrix = gridplot([[a_group, b_group], [c_group, d_group]], toolbar_location=None)

# Condiciones iniciales y ecuación matricial
initial_conditions = column(x1_input, x2_input, width=200, sizing_mode="fixed")
matrix_equation = Div(text="", width=200, height=300, sizing_mode="fixed")
update_matrix_equation()

# Crear el gráfico de campo vectorial
def create_vector_field_plot(a, b, c, d):
    x_vals = np.linspace(-2, 2, 20)
    y_vals = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x_vals, y_vals)
    A = np.array([[a, b],
                  [c, d]])
    DX, DY = A @ np.array([X.flatten(), Y.flatten()])
    DX = DX.reshape(X.shape)
    DY = DY.reshape(Y.shape)
    magnitude = np.sqrt(DX**2 + DY**2)
    DX /= magnitude
    DY /= magnitude
    x_start = X.flatten()
    y_start = Y.flatten()
    x_end = x_start + 0.1 * DX.flatten()
    y_end = y_start + 0.1 * DY.flatten()
    vector_field_plot = figure(width=400, height=300, title="Campo Vectorial", 
                               x_range=(-2, 2), y_range=(-2, 2))
    for xs, ys, xe, ye in zip(x_start, y_start, x_end, y_end):
        vector_field_plot.add_layout(Arrow(x_start=xs, y_start=ys, x_end=xe, y_end=ye,
                                           end=VeeHead(size=7), line_width=1.5))
    return vector_field_plot

# Crear el gráfico de campo vectorial inicial
vector_field_plot = create_vector_field_plot(initial_a, initial_b, initial_c, initial_d)

# Encabezado HTML
header = Div(text="<h1 style='text-align: center; color: white; background-color: blue; padding: 10px;'>Análisis y Visualización de Sistemas de Ecuaciones Diferenciales 2x2</h1>", width=800)

# Layout principal
plots = row(time_plot, phase_plot, tr_det_plot, vector_field_plot)
layout = column(
    header,
    row(controls_matrix, initial_conditions, matrix_equation),
    plots
)

curdoc().add_root(layout)