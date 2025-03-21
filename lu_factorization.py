# lu_factorization.py
import numpy as np

def solve_lu(A, b):
    """
    Resuelve Ax = b usando factorización PALU con pivoteo parcial.
    
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
            raise ValueError("Dimensiones incorrectas de A o b")
            
        # Inicializar matrices
        U = A.copy()
        L = np.eye(n)
        P = np.eye(n)
        
        for k in range(n-1):
            # Pivoteo parcial: encontrar fila con máximo valor en columna k
            max_row = np.argmax(np.abs(U[k:, k])) + k
            if max_row != k:
                # Intercambiar filas en U, L y P
                U[[k, max_row], :] = U[[max_row, k], :]
                P[[k, max_row], :] = P[[max_row, k], :]
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
        Pb = P @ b
        
        # Sustitución hacia adelante (Ly = Pb)
        y = np.zeros(n)
        for i in range(n):
            y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
        
        # Sustitución hacia atrás (Ux = y)
        x = np.zeros(n)
        for i in reversed(range(n)):
            if np.isclose(U[i, i], 0):
                return None
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
            
        return x
        
    except Exception as e:
        print(f"Error en LU: {str(e)}")
        return None

# Ejemplo de uso con pivoteo
if __name__ == "__main__":
    A = [[0, 2, 3], 
         [4, 5, 6], 
         [7, 8, 9]]
    b = [3, 6, 9]
    
    sol = solve_lu(A, b)
    print("Solución LU/PALU:", np.round(sol, 4) if sol is not None else "Matriz singular")