import numpy as np
import sympy as sp
from sympy import Matrix, Rational, sqrt, S, nsimplify, simplify
import tkinter as tk
from tkinter import ttk, messagebox, StringVar
import re
import sys

def find_basis_from_system(equations):
    """Find the basis for the null space of the system using SymPy"""
    # Convert the equations to a SymPy Matrix with rational coefficients
    A = Matrix([[Rational(str(coeff)) for coeff in eq] for eq in equations])
    
    # Get the null space (kernel) basis vectors
    null_space = A.nullspace()
    
    # Convert to list of vectors
    basis = [v for v in null_space]
    return basis


def orthonormalize_gram_schmidt(vectors):
    """
    Apply the Gram-Schmidt process using SymPy for exact calculations
    """
    if not vectors:
        return []
        
    ortho_vectors = []
    for v in vectors:
        # Make a copy of the original vector
        w = v.copy()
        
        # Subtract projections onto previous orthonormal vectors
        for u in ortho_vectors:
            # Calculate dot product symbolically
            dot_prod = sum(w[i] * u[i] for i in range(len(w)))
            w = w - dot_prod * u
        
        # Calculate the norm symbolically
        norm = sp.sqrt(sum(w[i]**2 for i in range(len(w))))
        
        # Only add if norm is not too close to zero
        if norm != 0:
            # Normalize the vector
            ortho_vectors.append(w / norm)
    
    return ortho_vectors


def calculate_projection_matrix(vectors):
    """Calculate the projection matrix using SymPy for exact calculations"""
    if not vectors:
        return None
    
    n = len(vectors[0])
    P = Matrix.zeros(n, n)
    
    for v in vectors:
        # Compute outer product v⊗v
        P += v * v.transpose()
    
    # Simplify each entry
    for i in range(P.rows):
        for j in range(P.cols):
            P[i, j] = simplify(P[i, j])
    
    return P


class SubspaceProjectionCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Subspace Projection Calculator")
        self.root.geometry("950x800")
        
        # Variables
        self.num_equations = tk.IntVar(value=2)
        self.num_variables = tk.IntVar(value=3)
        self.display_format = StringVar(value="exact")  # "exact" or "decimal"
        
        self.equation_entries = []
        self.basis_entries = []
        self.variable_labels = []
        
        # Create the main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.input_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.help_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.input_tab, text="Input")
        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.help_tab, text="Help")
        
        # Setup the tabs
        self.setup_input_tab()
        self.setup_results_tab()
        self.setup_help_tab()
        
        # Add menu
        self.create_menu()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Calculation", command=self.reset_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def show_about(self):
        messagebox.showinfo("About", "Subspace Projection Calculator\n\nA tool for computing bases and projections for linear subspaces with exact symbolic calculations.")
        
    def reset_all(self):
        """Reset all inputs to default values"""
        self.num_equations.set(2)
        self.num_variables.set(3)
        self.create_equation_inputs()
        self.results_text.delete(1.0, tk.END)
        
    def setup_input_tab(self):
        # Frame for system dimensions
        dimensions_frame = ttk.LabelFrame(self.input_tab, text="Vector Space Dimension", padding="10")
        dimensions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Description
        ttk.Label(dimensions_frame, text="Define the vector space E = ℝⁿ where n is the dimension:").grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # Number of variables (dimension)
        ttk.Label(dimensions_frame, text="Dimension n:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(dimensions_frame, from_=1, to=10, textvariable=self.num_variables, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(dimensions_frame, text="(This is the number of variables in your vector space)").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Number of equations
        ttk.Label(dimensions_frame, text="Number of equations:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(dimensions_frame, from_=1, to=10, textvariable=self.num_equations, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(dimensions_frame, text="(These equations define your subspace F)").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Apply button
        ttk.Button(dimensions_frame, text="Apply Dimensions", command=self.create_equation_inputs).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Frame for equations
        self.equations_frame = ttk.LabelFrame(self.input_tab, text="System of Linear Equations (defining the subspace)", padding="10")
        self.equations_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame for basis vectors
        self.basis_frame = ttk.LabelFrame(self.input_tab, text="Custom Basis Vectors (C) for ℝⁿ", padding="10")
        self.basis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Display format options
        format_frame = ttk.Frame(self.input_tab)
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(format_frame, text="Result display format:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Exact (symbolic)", variable=self.display_format, value="exact").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Decimal", variable=self.display_format, value="decimal").pack(side=tk.LEFT, padx=5)
        
        # Calculate button
        calculate_btn = ttk.Button(self.input_tab, text="Calculate", command=self.calculate)
        calculate_btn.pack(pady=10)

    def setup_results_tab(self):
        # Create a frame for results display with scrollbar
        results_frame = ttk.Frame(self.results_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for results - use a monospaced font for better alignment
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=80, height=30,
                                  yscrollcommand=scrollbar.set, font=('Courier New', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Buttons to recalculate with different formats
        format_frame = ttk.Frame(self.results_tab)
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(format_frame, text="Toggle Display Format", 
                  command=self.toggle_display_format).pack(side=tk.LEFT, padx=5)
        
    def setup_help_tab(self):
        # Create a frame for help text with scrollbar
        help_frame = ttk.Frame(self.help_tab)
        help_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(help_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for help content
        help_text = tk.Text(help_frame, wrap=tk.WORD, width=80, height=30,
                          yscrollcommand=scrollbar.set)
        help_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=help_text.yview)
        
        # Add help content
        help_content = """
# Subspace Projection Calculator Help

This application helps you work with linear subspaces and projections in linear algebra, using symbolic calculations for exact results.

## Basic Concepts:

- **Vector Space**: We work in ℝⁿ, an n-dimensional vector space over the real numbers.
- **Subspace F**: A subspace defined by a system of homogeneous linear equations (equations of the form a₁x₁ + a₂x₂ + ... + aₙxₙ = 0).
- **Basis**: A set of linearly independent vectors that span a space.
- **Orthogonal Projection**: A linear transformation projecting vectors onto the subspace.

## How to Use:

1. **Set Dimensions**:
   - Set the dimension n of your vector space (ℝⁿ)
   - Set the number of equations that define your subspace

2. **Enter Equations**:
   - For each equation, enter the coefficients for each variable
   - Example: For the equation x + 2y - 3z = 0, enter [1, 2, -3]
   - You can enter fractions like 1/2, 2/3, etc.

3. **Enter Custom Basis (Optional)**:
   - By default, the standard basis is used (e₁, e₂, ..., eₙ)
   - You can modify this to any valid basis of ℝⁿ
   - A valid basis must contain n linearly independent vectors

4. **Calculate**:
   - The program will:
     a) Find a basis for the null space of your system of equations, which represents the subspace F
     b) Create an orthonormal basis for F using the Gram-Schmidt process
     c) Calculate the orthogonal projection matrix onto the subspace F
     d) Express this projection in terms of your custom basis, if provided

5. **Display Format**:
   - Switch between exact (symbolic) and decimal formats
   - Use the "Toggle Display Format" button to switch between the two formats

## Mathematical Background:

- The null space of the coefficient matrix corresponds to the solution space of your homogeneous system of equations, which defines your subspace F.
- If F is a subspace of ℝⁿ, the orthogonal projection p_F: ℝⁿ → F maps each vector to its closest point in F.
- The matrix of this projection operator with respect to the standard basis is denoted Mat(p_F).
- If C is a custom basis for ℝⁿ, then Mat_C(p_F) = C⁻¹ · Mat(p_F) · C represents the projection matrix in terms of the basis C.
"""
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)  # Make read-only
        
    def create_equation_inputs(self):
        # Clear existing entries
        for widget in self.equations_frame.winfo_children():
            widget.destroy()
        
        self.equation_entries = []
        self.variable_labels = []
        
        # Generate variable names based on dimension
        var_names = self.get_variable_names(self.num_variables.get())
        
        # Create variable labels at the top
        explanation = ttk.Label(self.equations_frame, 
                             text=f"Enter coefficients for each equation (where equations are in the form a₁{var_names[0]} + a₂{var_names[1]} + ... = 0):")
        explanation.grid(row=0, column=0, columnspan=self.num_variables.get()+2, sticky=tk.W, pady=5)
        
        # Create variable headers
        for j in range(self.num_variables.get()):
            label = ttk.Label(self.equations_frame, text=f"{var_names[j]}")
            label.grid(row=1, column=j+1, padx=5, pady=2)
            self.variable_labels.append(label)
            
        # Create entries for each equation
        for i in range(self.num_equations.get()):
            ttk.Label(self.equations_frame, text=f"Equation {i+1}:").grid(row=i+2, column=0, sticky=tk.W, padx=5, pady=2)
            row_entries = []
            
            for j in range(self.num_variables.get()):
                entry = ttk.Entry(self.equations_frame, width=8)
                entry.grid(row=i+2, column=j+1, padx=2, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            
            # Add equation preview label
            eq_preview = ttk.Label(self.equations_frame, text="")
            eq_preview.grid(row=i+2, column=self.num_variables.get()+1, padx=5, pady=2, sticky=tk.W)
            
            # Bind updates to each entry
            for j, entry in enumerate(row_entries):
                entry.bind("<KeyRelease>", lambda e, row=i, col=j: self.update_equation_preview(row, col))
                
            self.equation_entries.append(row_entries)
            
        # Add explanation for the system
        system_note = ttk.Label(self.equations_frame, 
                             text=f"Note: This homogeneous system defines your subspace F ⊂ ℝ^{self.num_variables.get()}")
        system_note.grid(row=self.num_equations.get()+2, column=0, columnspan=self.num_variables.get()+2, sticky=tk.W, pady=5)
        
        # Create basis vector input
        self.create_basis_inputs()
    
    def update_equation_preview(self, row, col):
        """Update the preview of the equation as coefficients are entered"""
        if not self.equation_entries or row >= len(self.equation_entries):
            return
            
        # Get all coefficients for this row
        coeffs = []
        for entry in self.equation_entries[row]:
            try:
                # Support for entering fractions and decimals
                val_str = entry.get().strip()
                if not val_str:
                    coeffs.append(0)
                elif '/' in val_str:
                    # Handle fractions like 1/2, 3/4
                    num, denom = val_str.split('/')
                    coeffs.append(float(num) / float(denom))
                else:
                    coeffs.append(float(val_str))
            except ValueError:
                coeffs.append(0)
        
        # Generate variable names
        var_names = self.get_variable_names(self.num_variables.get())
        
        # Build equation string
        eq_str = ""
        for i, coeff in enumerate(coeffs):
            if coeff == 0:
                continue
                
            if eq_str and coeff > 0:
                eq_str += " + "
            elif eq_str and coeff < 0:
                eq_str += " - "
            elif coeff < 0:
                eq_str += "-"
                
            abs_coeff = abs(coeff)
            
            # Format coefficient
            if abs_coeff == 1:
                coeff_str = ""
            else:
                # Check if it's a simple fraction
                if abs_coeff.is_integer():
                    coeff_str = str(int(abs_coeff))
                else:
                    try:
                        # Try to express as a fraction
                        from fractions import Fraction
                        frac = Fraction(abs_coeff).limit_denominator(100)
                        if abs(float(frac) - abs_coeff) < 1e-10:
                            coeff_str = str(frac)
                        else:
                            coeff_str = f"{abs_coeff:.3f}"
                    except:
                        coeff_str = f"{abs_coeff:.3f}"
            
            if abs_coeff == 1:
                eq_str += f"{var_names[i]}"
            else:
                eq_str += f"{coeff_str}{var_names[i]}"
        
        if not eq_str:
            eq_str = "0"
            
        eq_str += " = 0"
        
        # Update the label
        eq_preview = self.equations_frame.grid_slaves(row=row+2, column=self.num_variables.get()+1)[0]
        eq_preview.config(text=eq_str)
        
    def get_variable_names(self, count):
        """Generate appropriate variable names based on dimension"""
        if count <= 4:
            return ["x", "y", "z", "w"][:count]
        elif count <= 8:
            return ["x₁", "x₂", "x₃", "x₄", "x₅", "x₆", "x₇", "x₈"][:count]
        else:
            return [f"x_{i+1}" for i in range(count)]
            
    def create_basis_inputs(self):
        # Clear existing entries
        for widget in self.basis_frame.winfo_children():
            widget.destroy()
        
        # Explanation
        explanation = ttk.Label(self.basis_frame, 
                             text=f"Custom basis for ℝ^{self.num_variables.get()} (default is the standard basis):")
        explanation.grid(row=0, column=0, columnspan=self.num_variables.get()+2, sticky=tk.W, pady=5)
        
        # Generate variable names based on dimension
        var_names = self.get_variable_names(self.num_variables.get())
        
        # Create variable headers
        for j in range(self.num_variables.get()):
            ttk.Label(self.basis_frame, text=f"{var_names[j]}").grid(row=1, column=j+1, padx=5, pady=2)
        
        # Set identity matrix as default basis
        self.basis_entries = []
        for i in range(self.num_variables.get()):
            ttk.Label(self.basis_frame, text=f"Vector {i+1}:").grid(row=i+2, column=0, sticky=tk.W, padx=5, pady=2)
            row_entries = []
            
            for j in range(self.num_variables.get()):
                entry = ttk.Entry(self.basis_frame, width=8)
                entry.grid(row=i+2, column=j+1, padx=2, pady=2)
                entry.insert(0, "1" if i == j else "0")  # Standard basis
                row_entries.append(entry)
                
            self.basis_entries.append(row_entries)
            
        # Add buttons and explanation
        button_frame = ttk.Frame(self.basis_frame)
        button_frame.grid(row=self.num_variables.get()+2, column=0, columnspan=self.num_variables.get()+2, pady=5)
        
        ttk.Button(button_frame, text="Reset to Standard Basis", 
                  command=self.reset_to_standard_basis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Check Linear Independence", 
                  command=self.check_basis_independence).pack(side=tk.LEFT, padx=5)
                  
        basis_note = ttk.Label(self.basis_frame, 
                            text="Note: A valid basis must contain n linearly independent vectors")
        basis_note.grid(row=self.num_variables.get()+3, column=0, columnspan=self.num_variables.get()+2, sticky=tk.W, pady=5)
    
    def reset_to_standard_basis(self):
        """Reset the basis to the standard basis (identity matrix)"""
        for i, row_entries in enumerate(self.basis_entries):
            for j, entry in enumerate(row_entries):
                entry.delete(0, tk.END)
                entry.insert(0, "1" if i == j else "0")
                
    def check_basis_independence(self):
        """Check if the current basis vectors are linearly independent"""
        basis = self.get_basis_vectors()
        if basis is None:
            return
            
        try:
            # Create a SymPy matrix from the basis vectors
            C_matrix = Matrix([[v[i] for i in range(len(v))] for v in basis])
            
            # Calculate determinant - non-zero means linearly independent
            det = C_matrix.det()
            
            if det == 0:
                messagebox.showwarning("Linearly Dependent", 
                                    "These vectors are linearly dependent and do not form a valid basis.")
            else:
                messagebox.showinfo("Linearly Independent", 
                                 "These vectors are linearly independent and form a valid basis.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error checking linear independence: {str(e)}")
    
    def toggle_display_format(self):
        """Toggle between exact and decimal formats"""
        current = self.display_format.get()
        self.display_format.set("decimal" if current == "exact" else "exact")
        # Recalculate to update display
        self.calculate()
    
    def get_equations(self):
        equations = []
        for row in self.equation_entries:
            equation = []
            for entry in row:
                try:
                    # Support for entering fractions
                    val_str = entry.get().strip()
                    if not val_str:
                        equation.append(0)
                    elif '/' in val_str:
                        num, denom = val_str.split('/')
                        equation.append(Rational(int(num), int(denom)))
                    else:
                        # Try to convert to a rational first
                        val = float(val_str)
                        equation.append(nsimplify(val, rational=True))
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid input: '{val_str}'. Please enter numbers only.")
                    return None
                except Exception as e:
                    messagebox.showerror("Input Error", f"Error processing input: {str(e)}")
                    return None
            equations.append(equation)
        return equations
    
    def get_basis_vectors(self):
        basis = []
        for row in self.basis_entries:
            vector = []
            for entry in row:
                try:
                    # Support for entering fractions
                    val_str = entry.get().strip()
                    if not val_str:
                        vector.append(Rational(0))
                    elif '/' in val_str:
                        num, denom = val_str.split('/')
                        vector.append(Rational(int(num), int(denom)))
                    else:
                        # Try to convert to a rational first
                        val = float(val_str)
                        vector.append(nsimplify(val, rational=True))
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid input: '{val_str}'. Please enter numbers only.")
                    return None
                except Exception as e:
                    messagebox.showerror("Input Error", f"Error processing input: {str(e)}")
                    return None
            basis.append(Matrix(vector))
        return basis
    
    def format_vector(self, vector, exact=True):
        """Format a SymPy vector for display"""
        if exact:
            # Use SymPy's LaTeX-like representation for exact display
            components = []
            for x in vector:
                # Simplify the expression first
                x_simple = simplify(x)
                
                # Special handling for common expressions
                if x_simple == 0:
                    components.append("0           ")
                elif x_simple == 1:
                    components.append("1           ")
                elif x_simple == -1:
                    components.append("-1          ")
                else:
                    # Handle square root expressions
                    x_str = str(x_simple)
                    
                    # Replace sqrt expressions with simpler notation
                    x_str = x_str.replace("sqrt(", "√(")
                    
                    # Handle special cases like 1/sqrt(2)
                    if "1/√(2)" in x_str:
                        x_str = x_str.replace("1/√(2)", "1/√2")
                    if "-1/√(2)" in x_str:
                        x_str = x_str.replace("-1/√(2)", "-1/√2")
                    
                    # Replace sqrt(n)/m expressions
                    import re
                    x_str = re.sub(r'√\((\d+)\)/(\d+)', r'√\1/\2', x_str)
                    
                    # Ensure fixed width for alignment
                    components.append(f"{x_str:<12}")
            
            return "[" + "  ".join(components) + "]"
        else:
            # Format as decimal
            components = []
            for x in vector:
                # Convert to float and format with fixed width
                components.append(f"{float(x):12.6f}")
            
            return "[" + "  ".join(components) + "]"
    
    def format_matrix(self, matrix, exact=True):
        """Format a SymPy matrix for display"""
        result = ""
        for i in range(matrix.rows):
            row = [matrix[i, j] for j in range(matrix.cols)]
            if exact:
                result += self.format_vector(row, exact=True) + "\n"
            else:
                components = []
                for x in row:
                    components.append(f"{float(x):12.6f}")
                result += "[" + "  ".join(components) + "]\n"
        return result
        
    def calculate(self):
        equations = self.get_equations()
        if equations is None:
            return
            
        # Clear results
        self.results_text.delete(1.0, tk.END)
        
        try:
            # Get the current display format
            is_exact = self.display_format.get() == "exact"
            
            # Step 1: Find null space basis vectors
            basis_vectors = find_basis_from_system(equations)
            
            if not basis_vectors:
                self.results_text.insert(tk.END, "The null space is trivial (only contains zero vector).\n")
                self.results_text.insert(tk.END, "This means the subspace F is just the zero vector {0}.\n")
                return
                
            self.results_text.insert(tk.END, "[1] Null space (ker(A)) basis vectors:\n")
            for i, v in enumerate(basis_vectors, 1):
                self.results_text.insert(tk.END, f"  v{i} = {self.format_vector(v, exact=is_exact)}\n")
            
            # Step 2: Orthonormalize using Gram-Schmidt
            orthonormal_vectors = orthonormalize_gram_schmidt(basis_vectors)
            self.results_text.insert(tk.END, "\n[2] Orthonormal basis for F:\n")
            for i, v in enumerate(orthonormal_vectors, 1):
                self.results_text.insert(tk.END, f"  u{i} = {self.format_vector(v, exact=is_exact)}\n")
            
            # Step 3: Calculate projection matrix
            projection_matrix_F = calculate_projection_matrix(orthonormal_vectors)
            self.results_text.insert(tk.END, "\n[3] Projection Matrix Mat(p_F):\n")
            self.results_text.insert(tk.END, self.format_matrix(projection_matrix_F, exact=is_exact))
            
            # Step 4: Get custom basis C
            basis_C = self.get_basis_vectors()
            if basis_C is None:
                return
                
            if len(basis_C) != self.num_variables.get():
                self.results_text.insert(tk.END, "\nError: Number of basis vectors must equal the number of variables for a complete basis.\n")
                return
                
            # Step 5: Calculate MatC(pF)
            C_matrix = Matrix.hstack(*basis_C)
            
            try:
                C_inv = C_matrix.inv()
                MatC_pF = C_inv * projection_matrix_F * C_matrix
                
                # Check if custom basis is different from standard basis
                is_different = True  # In symbolic math, we'll always show the result
                
                if is_different:
                    self.results_text.insert(tk.END, "\n[4] Mat_C(p_F) matrix (projection in custom basis):\n")
                    self.results_text.insert(tk.END, self.format_matrix(MatC_pF, exact=is_exact))
                else:
                    self.results_text.insert(tk.END, "\n[4] Note: Custom basis is equivalent to standard basis (within calculation tolerance).\n")
                    
            except Exception as e:
                self.results_text.insert(tk.END, f"\nError: Matrix C is not invertible. Please enter linearly independent basis vectors.\n")
                
            except Exception as e:
                self.results_text.insert(tk.END, f"\nError calculating with custom basis: {str(e)}\n")
                
            # Switch to results tab
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Calculation Error", f"An error occurred during calculation: {str(e)}")


def main():
    root = tk.Tk()
    app = SubspaceProjectionCalculator(root)
    
    # Handle the window close event properly
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    
    root.mainloop()

if __name__ == "__main__":
    main()