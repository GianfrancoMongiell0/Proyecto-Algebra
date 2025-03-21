# main.py
import numpy as np
from jacobi import solve_jacobi
from lu_factorization import solve_lu
from gauss_jordan import solve_gauss_jordan
from utils import load_system, print_solution

def main():
    print("\n=== SOLUCIONADOR DE SISTEMAS LINEALES ===")
    print("Métodos disponibles:")
    print("1. Factorización LU/PALU")
    print("2. Método de Jacobi")
    print("3. Gauss-Jordan")
    print("4. Salir")
    
    while True:
        try:
            choice = int(input("\nSeleccione un método (1-4): "))
            if choice == 4:
                print("¡Hasta luego!")
                break
            if choice not in [1, 2, 3]:
                raise ValueError("Selección inválida. Debe ser un número entre 1 y 4.")
            
            # Cargar sistema
            A, b = load_system()
            
            # Ejecutar método seleccionado
            if choice == 1:
                solution = solve_lu(A, b)
            elif choice == 2:
                initial_guess = input("Ingrese vector inicial (ej. 0,0,0,0) o enter para [0,...,0]: ")
                initial_guess = np.array([float(x) for x in initial_guess.split(',')]) if initial_guess else None
                solution = solve_jacobi(A, b, initial_guess)
            elif choice == 3:
                solution = solve_gauss_jordan(A, b)
            
            # Mostrar resultados
            print("\n=== SOLUCIÓN ===")
            print_solution(solution)
            
        except ValueError as ve:
            print(f"Error: {ve}")
        except Exception as e:
            print(f"Error inesperado: {str(e)}")

if __name__ == "__main__":
    main()