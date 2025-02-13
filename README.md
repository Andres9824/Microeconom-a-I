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

## Capacidad del Dashboard para Ecuaciones Lineales con Valores Propios Reales y Distintos

Este dashboard ha sido diseñado específicamente para analizar y visualizar sistemas de ecuaciones diferenciales lineales de 2x2 que tienen valores propios reales y distintos. Esto significa que la matriz del sistema, al ser diagonalizada, produce dos valores propios diferentes que son números reales. 

**¿Por qué es importante esta condición?**

La naturaleza de los valores propios de un sistema de ecuaciones diferenciales lineales determina el comportamiento cualitativo de sus soluciones. Cuando los valores propios son reales y distintos, las soluciones del sistema exhiben un comportamiento bien definido que puede ser clasificado y visualizado de manera clara.

**¿Cómo se visualizan las soluciones?**

El dashboard ofrece diversas herramientas de visualización para comprender el comportamiento de estos sistemas:

1.  **Retratos de fase:** Muestran el campo vectorial del sistema y las trayectorias de las soluciones en el plano de fase (x, y). Esto permite visualizar la dirección y la estabilidad de las soluciones.
2.  **Gráficas de series de tiempo:** Representan la evolución de las variables x(t) e y(t) en función del tiempo. Esto permite analizar el comportamiento temporal de las soluciones, como oscilaciones o convergencia a un punto de equilibrio.
3.  **Análisis de valores propios y vectores propios:** El dashboard calcula y muestra los valores propios y vectores propios del sistema. Estos elementos son cruciales para entender la estabilidad y la dirección de las soluciones.

**En resumen,** este dashboard es una herramienta poderosa para estudiar sistemas de ecuaciones diferenciales lineales con valores propios reales y distintos. Permite visualizar de manera intuitiva el comportamiento de las soluciones y comprender la dinámica de una amplia variedad de sistemas en diversas áreas de estudio.

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

## Bibliografía Fundamental

Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering* (2nd ed.). Boca Raton, FL: CRC Press.

Devaney, R. L. (2003). *An Introduction to Chaotic Dynamical Systems* (2nd ed.).  Westview Press.

Lorenz, E. N. (1993). *The Essence of Chaos*. University of Washington Press.

