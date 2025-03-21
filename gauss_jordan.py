# gauss_jordan.py
import numpy as np

def solve_gauss_jordan(A, b):
    """
    Resuelve el sistema Ax = b usando eliminación de Gauss-Jordan.
    
    Args:
        A: Matriz de coeficientes (n x n)
        b: Vector de términos independientes (n)
        
    Returns:
        Vector solución x (n) o None si el sistema no tiene solución única.
    """
    try:
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64).flatten()
        n = A.shape[0]
        
        # Validación de dimensiones
        if A.shape != (n, n) or b.shape != (n,):
            raise ValueError("Dimensiones incorrectas de A o b")
            
        # Matriz aumentada
        augmented = np.hstack((A, b.reshape(-1, 1)))
        
        # Eliminación de Gauss-Jordan
        for col in range(n):
            # Pivoteo parcial para evitar división por cero
            max_row = np.argmax(np.abs(augmented[col:, col])) + col
            if np.isclose(augmented[max_row, col], 0):
                return None  # Matriz singular
            
            # Intercambiar filas
            augmented[[col, max_row]] = augmented[[max_row, col]]
            
            # Normalizar fila pivote
            pivot = augmented[col, col]
            augmented[col] /= pivot
            
            # Eliminar en otras filas
            for r in range(n):
                if r != col:
                    factor = augmented[r, col]
                    augmented[r] -= factor * augmented[col]
        
        # Verificar consistencia (filas 0x = b?)
        for row in augmented:
            if np.allclose(row[:-1], 0) and not np.isclose(row[-1], 0):
                return None  # Sistema inconsistente
        
        # Extraer solución
        solution = augmented[:, -1]
        return solution
        
    except Exception as e:
        print(f"Error en Gauss-Jordan: {str(e)}")
        return None

# Ejemplo de uso (para test_cases.py)
if __name__ == "__main__":
    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
    b = np.array([5, 5, 5], dtype=float)
    sol = solve_gauss_jordan(A, b)
    print("Solución Gauss-Jordan:", np.round(sol, 4))