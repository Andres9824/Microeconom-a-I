## Análisis y Visualización de Sistemas de Ecuaciones Diferenciales 2x2

**Autores**

- Andrés Lema
- Tito Madrid

FLACSO - Ecuador

**Descripción**

Este proyecto permite analizar y visualizar sistemas de ecuaciones diferenciales lineales.

$$
\begin{pmatrix}
\dot{x} \\
\dot{y}
\end{pmatrix} = 
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\begin{pmatrix}
x \\
y
\end{pmatrix}
$$

donde A, B, C y D son coeficientes modificables. Se utilizan herramientas de Python para resolver el sistema, visualizar su evolución temporal, su diagrama de fase y analizar la relación entre la traza y el determinante de la matriz del sistema.

**Funcionalidades**

1. **Resolución del sistema dinámico**

   - Se utiliza `solve_ivp` de `scipy.integrate` para resolver el sistema de ecuaciones diferenciales.
   - Se representan gráficamente las funciones `x(t)` e `y(t)` en el tiempo.

2. **Diagrama de fase**

   - Se muestra el campo vectorial del sistema.
   - Se incluyen las nullclines, que indican los puntos donde `dx/dt = 0` y `dy/dt = 0`.

3. **Relación entre traza y determinante**

   - Se grafica la relación `traza(M)` vs `determinante(M)`, donde `M` es la matriz del sistema.

4. **Cálculo de valores propios y vectores propios**

   - Se calcula la traza y el determinante de la matriz del sistema.
   - Se determinan los valores propios y sus vectores asociados.

5. **Interfaz interactiva**

   - Se emplean `ipywidgets` para modificar los parámetros del sistema (A, B, C, D) y las condiciones iniciales `x(0)` e `y(0)`.
   - Los cambios en los sliders o campos de texto actualizan automáticamente los gráficos y cálculos.

**Uso**

1. Ejecutar el script en un entorno Jupyter Notebook.
2. Ajustar los valores de A, B, C, D y las condiciones iniciales usando los sliders o campos de texto.
3. Observar los cambios en los gráficos y cálculos mostrados.

**Requisitos**

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- IPython
- ipywidgets

