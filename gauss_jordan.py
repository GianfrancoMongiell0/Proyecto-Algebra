# gauss_jordan.py
import numpy as np

def solve_gauss_jordan(A, b, tol=1e-8):
    """
    Resuelve Ax = b usando eliminación de Gauss-Jordan con pivoteo completo.
    Corregido: Paréntesis balanceado en np.unravel_index.
    """
    try:
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64).flatten()
        n = A.shape[0]
        
        # Validar dimensiones
        if A.shape != (n, n) or b.size != n:
            raise ValueError("Dimensiones incorrectas")
            
        # Matriz aumentada
        augmented = np.c_[A, b].astype(np.float64)
        col_order = list(range(n))  # Seguimiento de permutaciones de columnas
        
        for r in range(n):
            # Pivoteo completo: buscar máximo en submatriz
            max_index = np.argmax(np.abs(augmented[r:, r:]))
            max_row, max_col = np.unravel_index(max_index, (n - r, n - r))
            max_row += r
            max_col += r
            
            # Intercambiar filas y columnas
            augmented[[r, max_row]] = augmented[[max_row, r]]  # Filas
            augmented[:, [r, max_col]] = augmented[:, [max_col, r]]  # Columnas
            col_order[r], col_order[max_col] = col_order[max_col], col_order[r]
            
            # Verificar singularidad
            if np.abs(augmented[r, r]) < tol:
                return None
                
            # Normalizar fila pivote
            pivot = augmented[r, r]
            augmented[r] /= pivot
            
            # Eliminación en todas las filas excepto pivote
            for i in range(n):
                if i != r:
                    factor = augmented[i, r]
                    augmented[i] -= factor * augmented[r]
        
        # Reordenar columnas y extraer solución
        solution = np.zeros(n)
        for idx, original_col in enumerate(col_order):
            solution[original_col] = augmented[idx, -1]
        
        # Verificar consistencia
        residual = np.abs(augmented[:, -1] - augmented[:, :-1] @ solution)
        if np.any(residual > 1e-6):
            return None
            
        return solution
        
    except Exception as e:
        print(f"Error en Gauss-Jordan: {str(e)}")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    A = [[3, -2, 4], [1, 1, 2], [2, 3, -1]]
    b = [11, 4, 3]  # Solución exacta: [3, -1, 2]
    sol = solve_gauss_jordan(A, b)
    print("Solución exacta:" if sol is not None else "Sistema incompatible:", 
          np.round(sol, 10) if sol is not None else "")