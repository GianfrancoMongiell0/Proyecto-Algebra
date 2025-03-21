# lu_factorization.py
import numpy as np

def solve_lu(A, b):
    """
    Resuelve Ax = b usando factorización LU con pivoteo parcial (PA = LU).
    
    Args:
        A: Matriz de coeficientes (n x n)
        b: Vector de términos independientes (n)
        
    Returns:
        Vector solución x (n) o None si la matriz es singular
    """
    try:
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64).flatten()
        n = A.shape[0]
        
        # Validar dimensiones
        if A.shape != (n, n) or b.size != n:
            raise ValueError("Dimensiones incorrectas")
            
        # Inicializar matrices L, U y P
        L = np.eye(n)
        U = A.copy()
        P = np.eye(n)
        
        for k in range(n-1):
            # Pivoteo parcial
            max_row = np.argmax(np.abs(U[k:, k])) + k
            if max_row != k:
                U[[k, max_row]] = U[[max_row, k]]
                P[[k, max_row]] = P[[max_row, k]]
                if k > 0:
                    L[[k, max_row], :k] = L[[max_row, k], :k]
            
            # Verificar si la matriz es singular
            if np.isclose(U[k, k], 0):
                return None
            
            # Calcular multiplicadores y actualizar matrices
            for j in range(k+1, n):
                L[j, k] = U[j, k] / U[k, k]
                U[j, k:] -= L[j, k] * U[k, k:]
                
        # Resolver Ly = Pb y Ux = y
        Pb = np.dot(P, b)
        
        # Sustitución hacia adelante (L)
        y = np.zeros(n)
        for i in range(n):
            y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
        
        # Sustitución hacia atrás (U)
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
            
        return x
        
    except Exception as e:
        print(f"Error en LU: {str(e)}")
        return None

# Ejemplo de uso (para test_cases.py)
if __name__ == "__main__":
    # Sistema que requiere pivoteo
    A = [[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    b = [3, 6, 9]
    sol = solve_lu(A, b)
    print("Solución LU/PALU:", np.round(sol, 4) if sol is not None else "Sistema singular")