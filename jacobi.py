# jacobi.py
import numpy as np

def is_diagonally_dominant(A):
    """Verifica si la matriz es estrictamente diagonalmente dominante."""
    n = A.shape[0]
    for i in range(n):
        diagonal = np.abs(A[i, i])
        row_sum = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= row_sum:  # Usar <= para detectar no estricto
            return False
    return True

def reorder_matrix(A, b):
    """Reordena filas para maximizar la diagonal (heurística mejorada)."""
    n = A.shape[0]
    for i in range(n):
        # Buscar máximo en la columna actual desde la fila i hacia abajo
        max_row = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max_row]], b[[i, max_row]] = A[[max_row, i]].copy(), b[[max_row, i]].copy()
    return A, b

def solve_jacobi(A, b, initial_guess=None, max_iter=5000, tol=1e-4):
    """
    Versión mejorada del método de Jacobi con:
    - Mayor número de iteraciones
    - Tolerancia relajada
    - Manejo de advertencias en lugar de errores
    """
    try:
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64).flatten()
        n = A.shape[0]
        
        # Verificar y reordenar sin detener la ejecución
        if not is_diagonally_dominant(A):
            print("\n[ADVERTENCIA] Matriz no diagonalmente dominante. Reordenando...")
            A, b = reorder_matrix(A.copy(), b.copy())
            if not is_diagonally_dominant(A):
                print("[ADVERTENCIA] Convergencia no garantizada. Intentando igualmente...")

        # Inicialización
        x = initial_guess if initial_guess is not None else np.zeros(n)
        x_new = np.zeros(n)
        D = np.diag(A)
        R = A - np.diagflat(D)
        
        # Iteraciones
        for k in range(max_iter):
            x_new = (b - np.dot(R, x)) / D
            if np.linalg.norm(x_new - x) < tol:
                print(f"Convergió en {k+1} iteraciones")
                return x_new
            x = x_new.copy()
        
        print(f"No convergió en {max_iter} iteraciones (Última aproximación: {np.round(x_new, 4)})")
        return x_new

    except Exception as e:
        print(f"Error crítico: {str(e)}")
        return None

# Ejemplo de prueba con el nuevo caso del test_cases.py
if __name__ == "__main__":
    A = np.array([[10, 1, 2], [2, 15, 1], [1, 2, 20]])
    b = np.array([13, 18, 23])
    sol = solve_jacobi(A, b)
    print("Solución Jacobi:", np.round(sol, 4))