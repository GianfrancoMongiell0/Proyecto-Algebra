# jacobi.py
import numpy as np

def is_diagonally_dominant(A):
    """Verifica si la matriz es diagonalmente dominante."""
    n = A.shape[0]
    for i in range(n):
        diagonal = abs(A[i, i])
        row_sum = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal < row_sum:
            return False
    return True

def reorder_matrix(A, b):
    """Intenta reordenar filas para hacerla diagonalmente dominante (heurística simple)."""
    n = A.shape[0]
    for i in range(n):
        max_idx = np.argmax(np.abs(A[i:, i])) + i
        if i != max_idx:
            A[[i, max_idx]], b[[i, max_idx]] = A[[max_idx, i]].copy(), b[[max_idx, i]].copy()
    return A, b

def solve_jacobi(A, b, initial_guess=None, max_iter=1000, tol=1e-6):
    """
    Resuelve Ax = b usando el método de Jacobi.
    
    Args:
        A: Matriz de coeficientes
        b: Vector de términos independientes
        initial_guess: Vector inicial (opcional)
        max_iter: Máximo de iteraciones
        tol: Tolerancia para convergencia
    
    Returns:
        Solución como array NumPy o None si no converge
    """
    # Verificar y ajustar condicionamiento
    if not is_diagonally_dominant(A):
        print("Advertencia: Matriz no diagonalmente dominante. Intentando reordenar...")
        A_modified, b_modified = reorder_matrix(A.copy(), b.copy())
        if not is_diagonally_dominant(A_modified):
            raise ValueError("No se pudo convertir a diagonalmente dominante")
        A, b = A_modified, b_modified
    
    n = len(b)
    x = initial_guess if initial_guess is not None else np.zeros(n)
    x_new = np.zeros(n)
    
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for _ in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new.copy()
    
    print("No convergió en el máximo de iteraciones")
    return x_new

# Ejemplo de uso (para test_cases.py)
if __name__ == "__main__":
    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
    b = np.array([5, 5, 5])
    sol = solve_jacobi(A, b)
    print("Solución Jacobi:", np.round(sol, 4))