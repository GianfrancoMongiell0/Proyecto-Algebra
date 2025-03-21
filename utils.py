# utils.py (funciones necesarias para main)
import numpy as np

def load_system():
    source = input("¿Cargar desde archivo? (s/n): ").lower()
    if source == 's':
        filename = input("Nombre del archivo (ej. sistema.txt): ")
        return load_from_file(filename)
    else:
        return manual_input()

def load_from_file(filename):
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        n = int(lines[0])
        A = np.array([list(map(float, line.split())) for line in lines[1:n+1]])
        b = np.array(list(map(float, lines[n+1].split())))
        return A, b
    except:
        raise Exception("Formato de archivo incorrecto")

def manual_input():
    print("\n=== ENTRADA MANUAL ===")
    print("Ingrese el tamaño del sistema (n). Ejemplo: para 3 ecuaciones, escriba 3")
    n = int(input("  → Tamaño del sistema (n): "))
    
    print(f"\nINGRESE LA MATRIZ A ({n}x{n}):")
    print("- Separe los valores con espacios")
    print("- Ejemplo para una fila de 3 elementos: 4 -1 0\n")
    
    A = []
    for i in range(n):
        while True:
            row_input = input(f"  → Fila {i+1}: ")
            try:
                row = list(map(float, row_input.split()))
                if len(row) != n:
                    raise ValueError(f"Debe ingresar exactamente {n} valores")
                A.append(row)
                break
            except ValueError as e:
                print(f"Error: {e}. Intente nuevamente.")
    
    print(f"\nINGRESE EL VECTOR B ({n} elementos):")
    print("- Ejemplo para 3 elementos: 5 5 5")
    while True:
        b_input = input("  → Vector b: ")
        try:
            b = list(map(float, b_input.split()))
            if len(b) != n:
                raise ValueError(f"Debe ingresar exactamente {n} valores")
            break
        except ValueError as e:
            print(f"Error: {e}. Intente nuevamente.")
    
    # Confirmación de datos
    print("\n=== RESUMEN DE ENTRADA ===")
    print("Matriz A:")
    for row in A:
        print("  [", "  ".join(f"{x:>5}" for x in row), "]")
    print("\nVector b:")
    print("  [", "  ".join(f"{x:>5}" for x in b), "]")
    print("="*30)
    
    return np.array(A), np.array(b)

def print_solution(solution):
    if solution is None:
        print("El sistema no tiene solución única")
    else:
        for i, x in enumerate(solution):
            print(f"x{i+1} = {x:.6f}")