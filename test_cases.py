# test_cases.py
import numpy as np
from jacobi import solve_jacobi
from lu_factorization import solve_lu
from gauss_jordan import solve_gauss_jordan

def print_test_case(title, A, b):
    print(f"\n{'='*50}")
    print(f" TEST CASE: {title} ")
    print(f"{'='*50}")
    print("Sistema:")
    print(f"A =\n{np.array(A)}\n")
    print(f"b = {np.array(b)}\n")

def test_jacobi():
    # Caso 1: Sistema diagonalmente dominante (sin reordenar)
    A1 = [[4, -1, 0], [-1, 4, -1], [0, -1, 3]]
    b1 = [5, 5, 5]
    print_test_case("Jacobi - Bien condicionado", A1, b1)
    sol = solve_jacobi(np.array(A1), np.array(b1))
    print(f"Solución: {np.round(sol, 4) if sol is not None else 'No convergió'}")

    # Caso 2: Sistema que requiere reordenamiento y converge
    A2 = [[10, 1, 2], [2, 15, 1], [1, 2, 20]]  # Estrictamente diagonal dominante
    b2 = [13, 18, 23]                          # Solución esperada: [1, 1, 1]
    print_test_case("Jacobi - Reordenado exitoso", A2, b2)
    sol = solve_jacobi(np.array(A2), np.array(b2))
    print(f"Solución: {np.round(sol, 4) if sol is not None else 'Error'}")

def test_lu():
    # Caso 1: Matriz sin pivoteo requerido (solución exacta)
    A1 = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b1 = [8, -11, -3]  # Solución: [2, 3, -1]
    print_test_case("LU - Sin permutaciones", A1, b1)
    sol = solve_lu(np.array(A1), np.array(b1))
    print(f"Solución: {np.round(sol, 4) if sol is not None else 'Matriz singular'}")

    # Caso 2: Matriz que requiere pivoteo (solución única)
    A2 = [[0, 2, 3], [4, 5, 6], [7, 8, 9]]
    b2 = [3, 6, 9]  # Solución: [0, 0, 1]
    print_test_case("LU - Con permutaciones (PALU)", A2, b2)
    sol = solve_lu(np.array(A2), np.array(b2))
    print(f"Solución: {np.round(sol, 4) if sol is not None else 'Matriz singular'}")

def test_gauss_jordan():
    # Caso 1: Sistema con solución única (precisión numérica)
    A1 = [[3, -2, 4], [1, 1, 2], [2, 3, -1]]
    b1 = [11, 4, 3]  # Solución aproximada: [2.2593, -0.1852, 0.963]
    print_test_case("Gauss-Jordan - Solución única", A1, b1)
    sol = solve_gauss_jordan(np.array(A1), np.array(b1))
    print(f"Solución: {np.round(sol, 4) if sol is not None else 'No solución única'}")

    # Caso 2: Sistema inconsistente
    A2 = [[1, 2], [1, 2]]
    b2 = [5, 7]
    print_test_case("Gauss-Jordan - Sistema inconsistente", A2, b2)
    sol = solve_gauss_jordan(np.array(A2), np.array(b2))
    print(f"Solución: {'Sistema incompatible' if sol is None else 'Error'}")

def test_errores():
    # Caso 1: Dimensiones incompatibles
    A = [[1, 2], [3, 4]]
    b = [5]
    print_test_case("Error - Dimensiones incompatibles", A, b)
    print("Intentando resolver con LU:")
    sol = solve_lu(np.array(A), np.array(b))
    print(f"Resultado: {'Error manejado' if sol is None else 'Fallo en validación'}")

    # Caso 2: Matriz singular
    A = [[1, 1], [1, 1]]
    b = [2, 2]
    print_test_case("Error - Matriz singular", A, b)
    print("Intentando resolver con Gauss-Jordan:")
    sol = solve_gauss_jordan(np.array(A), np.array(b))
    print(f"Resultado: {'Sistema incompatible' if sol is None else 'Fallo en detección'}")

if __name__ == "__main__":
    print("\n" + "="*20 + " INICIO DE PRUEBAS " + "="*20)
    test_jacobi()
    test_lu()
    test_gauss_jordan()
    test_errores()
    print("\n" + "="*20 + " PRUEBAS COMPLETADAS " + "="*20)