<div align="center">

# 🧮 Solucionador de Sistemas Lineales

### Álgebra Lineal — Universidad Metropolitana (UNIMET)

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-cálculo%20matricial-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![Universidad](https://img.shields.io/badge/UNIMET-Ing.%20Sistemas-004A8F?style=flat-square)](https://www.unimet.edu.ve)

**CLI en Python que implementa tres métodos numéricos para resolver sistemas de ecuaciones lineales `Ax = b`, cada uno con manejo de casos especiales, pivoteo y suite de pruebas.**

</div>

---

## ¿Qué es?

Este proyecto implementa desde cero tres métodos numéricos clásicos para resolver sistemas `Ax = b`. No usa `numpy.linalg.solve` — cada algoritmo está construido manualmente sobre operaciones matriciales de NumPy, incluyendo los casos bordes que hacen no trivial cada método: matrices singulares, dominancia diagonal, pivoteo parcial y completo, y convergencia iterativa.

---

## Métodos implementados

### 1. Factorización PALU — `lu_factorization.py`

Descompone `A = P·L·U` donde `P` es la matriz de permutaciones del **pivoteo parcial**. Luego resuelve en dos pasos: sustitución hacia adelante (`Ly = Pb`) y hacia atrás (`Ux = y`).

El pivoteo parcial es la parte clave — en cada columna busca el elemento de mayor valor absoluto y permuta filas en `U`, `L` y `P` para evitar división por valores cercanos a cero:

```python
for k in range(n-1):
    # Pivoteo parcial: busca el máximo en la columna k desde la fila k
    max_row = np.argmax(np.abs(U[k:, k])) + k
    if max_row != k:
        U[[k, max_row], :] = U[[max_row, k], :]
        P[[k, max_row], :] = P[[max_row, k], :]
        if k > 0:
            L[[k, max_row], :k] = L[[max_row, k], :k]

    if np.isclose(U[k, k], 0):
        return None  # Matriz singular

    for j in range(k+1, n):
        L[j, k] = U[j, k] / U[k, k]
        U[j, k:] -= L[j, k] * U[k, k:]
```

Esto permite resolver sistemas como `A = [[0,2,3],[4,5,6],[7,8,9]]` donde sin pivoteo habría división por cero en la primera columna.

---

### 2. Método de Jacobi — `jacobi.py`

Método iterativo que descompone `A = D + R` y en cada paso calcula `x_new = (b - R·x) / D`. Converge garantizadamente si la matriz es **estrictamente diagonalmente dominante** (`|a_ii| > Σ|a_ij|` para `j ≠ i`).

Cuando la matriz no cumple la condición, intenta **reordenar las filas heurísticamente** (maximizando la diagonal) antes de iterar:

```python
def is_diagonally_dominant(A):
    for i in range(n):
        diagonal = np.abs(A[i, i])
        row_sum = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= row_sum:
            return False
    return True

def reorder_matrix(A, b):
    for i in range(n):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max_row]], b[[i, max_row]] = A[[max_row, i]].copy(), b[[max_row, i]].copy()
    return A, b
```

Si tras reordenar sigue sin ser dominante, igual intenta las iteraciones con aviso — en muchos sistemas en la práctica converge de todas formas. El criterio de parada es `||x_new - x|| < 1e-4` con un máximo de 5000 iteraciones.

---

### 3. Eliminación de Gauss-Jordan — `gauss_jordan.py`

Trabaja sobre la **matriz aumentada** `[A|b]` y aplica **pivoteo completo** — busca el máximo en toda la submatriz restante (no solo en la columna), intercambiando tanto filas como columnas. Esto es más robusto que el pivoteo parcial pero requiere rastrear las permutaciones de columnas para reordenar la solución al final:

```python
col_order = list(range(n))  # Registro de permuaciones de columnas

for r in range(n):
    # Pivoteo completo: máximo en toda la submatriz [r:, r:]
    max_index = np.argmax(np.abs(augmented[r:, r:]))
    max_row, max_col = np.unravel_index(max_index, (n - r, n - r))
    max_row += r; max_col += r

    augmented[[r, max_row]] = augmented[[max_row, r]]          # permuta filas
    augmented[:, [r, max_col]] = augmented[:, [max_col, r]]    # permuta columnas
    col_order[r], col_order[max_col] = col_order[max_col], col_order[r]

# Al final, reordena la solución según col_order
solution = np.zeros(n)
for idx, original_col in enumerate(col_order):
    solution[original_col] = augmented[idx, -1]
```

Tras obtener la solución, verifica la consistencia calculando el residual `||b - Ax||`. Si supera `1e-6`, retorna `None`.

---

## Entrada del sistema

El programa acepta dos modos de entrada:

**Manual por consola:**
```
Ingrese el tamaño del sistema (n): 3

INGRESE LA MATRIZ A (3x3):
  → Fila 1: 4 -1 0
  → Fila 2: -1 4 -1
  → Fila 3: 0 -1 3

INGRESE EL VECTOR B (3 elementos):
  → Vector b: 5 5 5
```

**Desde archivo `.txt`:**
```
3
4 -1 0
-1 4 -1
0 -1 3
5 5 5
```
Primera línea: `n`. Siguientes `n` líneas: filas de `A`. Última línea: vector `b`.

---

## Suite de pruebas — `test_cases.py`

Incluye casos de prueba para cada método y sus casos bordes:

| Test | Método | Caso |
|------|--------|------|
| Jacobi bien condicionado | Jacobi | Matriz diagonalmente dominante, convergencia garantizada |
| Jacobi con reordenamiento | Jacobi | Matriz que requiere permutación de filas |
| LU sin permutaciones | PALU | Sistema directo sin pivoteo necesario |
| LU con pivoteo (PALU) | PALU | Primera columna con cero — requiere permutación |
| Gauss-Jordan solución única | Gauss-Jordan | Sistema con solución única, verifica residual |
| Sistema inconsistente | Gauss-Jordan | `A = [[1,2],[1,2]]` — detectado correctamente |
| Dimensiones incompatibles | Todos | `A` 2x2, `b` de 1 elemento |
| Matriz singular | Todos | Detectada y retorna `None` |

```bash
python test_cases.py
```

---

## Instalación y uso

```bash
git clone https://github.com/GianfrancoMongiell0/Proyecto-Algebra
cd Proyecto-Algebra
pip install numpy
python main.py
```

```
=== SOLUCIONADOR DE SISTEMAS LINEALES ===
1. Factorización LU/PALU
2. Método de Jacobi
3. Gauss-Jordan
4. Salir

Seleccione un método (1-4):
```

---

## Autor

**Gianfranco Mongiello** — [@GianfrancoMongiell0](https://github.com/GianfrancoMongiell0)  
Materia: Álgebra Lineal — Universidad Metropolitana (UNIMET), Caracas 🇻🇪
