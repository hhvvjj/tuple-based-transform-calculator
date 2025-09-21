# Tuple-Based Transform Calculator

[![Research](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15546925-orange.svg)](https://doi.org/10.5281/zenodo.15546925)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

A Python implementation of tuple-based transform for Collatz sequences, providing a novel mathematical perspective on the famous $3n+1$ problem through isomorphic mappings between consecutive pairs and their tuple representations.

## Mathematical Foundation

The Tuple-Based Transform representation is a novel mathematical approach to analyzing Collatz sequences through isomorphic mappings. Rather than studying these hailstone sequences in their traditional form, this tool converts each consecutive Collatz pair $(c_i, c_{i+1})$ into a unique 4-element tuple $[p, f(p), m, q]$ that preserves all mathematical information while revealing hidden structural patterns.

This transformation is particularly compelling because it is completely reversible: every tuple can be perfectly reconstructed back to its original consecutive Collatz pair, ensuring no mathematical information is lost in the process.

The result is an alternative lens through which to examine one of mathematics' most famous unsolved problems: the Collatz Conjecture.

### Transformation Formulas

For any consecutive pair $(c_i, c_{i+1})$ in a Collatz sequence, the tuple $[p, f(p), m, q]$ satisfies:

$$
\begin{align}
	\nonumber c_i &= 2 \times q \times m + p &\text{(always)} \\
	\nonumber c_{i+1} &= q \times m + f(p) &\text{(if $p$ is even)} \\
	\nonumber c_{i+1} &= 6 \times q \times m + f(p) &\text{(if $p$ is odd)}
\end{align}
$$

Where:

- $p \in [1, 2q]$ with same parity as $c_i$
- $f(p)$ is the result of applying the Collatz function to $p$
- $m \leq 0$ is the multiplicity parameter
- $q$ is an odd positive transform parameter (most interesting results using $q = 1$)

## Computational Complexity

   - Time: $O(k \times q)$ where $k$ is sequence length and $q$ is transform parameter
   - Space: $O(k)$ for sequence storage

## Installation

### System Requirements

- Python 3.8 or higher
- No external dependencies required (uses only standard library)

### Package Installation

### Red Hat-based Systems (RHEL, CentOS, Fedora, Rocky Linux or AlmaLinux)

```bash
# RHEL/CentOS/Rocky/AlmaLinux 8+
sudo dnf install python3

# RHEL/CentOS 7
sudo yum install python3

# Fedora
sudo dnf install python3

# Verify installation
python3 --version
```
#### Debian-based Systems (Ubuntu, Debian or Linux Mint)

```bash
# Ubuntu/Debian/Mint
sudo apt update
sudo apt install python3 python3-pip

# Verify installation
python3 --version
```

### Setup

```bash
# Clone the repository
git clone https://github.com/hhvvjj/tuple-based-transform-calculator.git
cd tuple-based-transform-calculator
```

## Usage

```
python3 tuple_based_transform_calculator.py <n> [q]
```

### Parameters:

- **n**: Initial positive integer for Collatz sequence (mandatory)
- **q**: Odd positive parameter for transform (optional, default=1)

### Examples

```
# Basic Analysis
python3 tuple_based_transform_calculator.py 7

# Custom q Parameter
python3 tuple_based_transform_calculator.py 25 13
```

### Console Output

```
**************************************************************************
* Tuple-based transform calculator                                       *
**************************************************************************

[*] ALGORITHM SETUP:

	Initial values: n = 3 and q = 1 [default]

[*] COLLATZ SEQUENCE:

  [3, 10, 5, 16, 8, 4, 2, 1]

[*] TUPLE-BASED TRANSFORM AND RECONSTRUCTION:

	- Step 1:
		Original pair (ci, ci+1): ( 3 ,  10 )
		Transformation process:
			Testing p ∈ [1, 2] with same parity as ci=3 (odd):
				p=1 is odd, so parity OK, Then, the formula m = (ci - p) ÷ (2 · q) = (3 - 1) ÷ (2) = 1, f(1)=4, so OK
				p=2 is even, so parity is NOK
		Tuple-based transform: [p=1, f(p)=4, m=1, q=1]
		Reconstruction process:
			ci   = 2 · q · m + p    = 2 · 1 · 1 + 1 = 2 + 1 =  3 
			ci+1 = 6 · q · m + f(p) = 6 · 1 · 1 + 4 = 6 + 4 =  10 

	- Step 2:
		Original pair (ci, ci+1): ( 10 ,  5 )
		Transformation process:
			Testing p ∈ [1, 2] with same parity as ci=10 (even):
				p=1 is odd, so parity is NOK
				p=2 is even, so parity OK, Then, the formula m = (ci - p) ÷ (2 · q) = (10 - 2) ÷ (2) = 4, f(2)=1, so OK
		Tuple-based transform: [p=2, f(p)=1, m=4, q=1]
		Reconstruction process:
			ci   = 2 · q · m + p    = 2 · 1 · 4 + 2 = 8 + 2 =  10 
			ci+1 =     q · m + f(p) = 1 · 4 + 1     = 4 + 1 =  5 

	- Step 3:
		Original pair (ci, ci+1): ( 5 ,  16 )
		Transformation process:
			Testing p ∈ [1, 2] with same parity as ci=5 (odd):
				p=1 is odd, so parity OK, Then, the formula m = (ci - p) ÷ (2 · q) = (5 - 1) ÷ (2) = 2, f(1)=4, so OK
				p=2 is even, so parity is NOK
		Tuple-based transform: [p=1, f(p)=4, m=2, q=1]
		Reconstruction process:
			ci   = 2 · q · m + p    = 2 · 1 · 2 + 1 = 4 + 1  =  5 
			ci+1 = 6 · q · m + f(p) = 6 · 1 · 2 + 4 = 12 + 4 =  16 

	- Step 4:
		Original pair (ci, ci+1): ( 16 ,  8 )
		Transformation process:
			Testing p ∈ [1, 2] with same parity as ci=16 (even):
				p=1 is odd, so parity is NOK
				p=2 is even, so parity OK, Then, the formula m = (ci - p) ÷ (2 · q) = (16 - 2) ÷ (2) = 7, f(2)=1, so OK
		Tuple-based transform: [p=2, f(p)=1, m=7, q=1]
		Reconstruction process:
			ci   = 2 · q · m + p    = 2 · 1 · 7 + 2 = 14 + 2 =  16 
			ci+1 =     q · m + f(p) = 1 · 7 + 1     = 7 + 1  =  8 

	- Step 5:
		Original pair (ci, ci+1): ( 8 ,  4 )
		Transformation process:
			Testing p ∈ [1, 2] with same parity as ci=8 (even):
				p=1 is odd, so parity is NOK
				p=2 is even, so parity OK, Then, the formula m = (ci - p) ÷ (2 · q) = (8 - 2) ÷ (2) = 3, f(2)=1, so OK
		Tuple-based transform: [p=2, f(p)=1, m=3, q=1]
		Reconstruction process:
			ci   = 2 · q · m + p    = 2 · 1 · 3 + 2 = 6 + 2 =  8 
			ci+1 =     q · m + f(p) = 1 · 3 + 1     = 3 + 1 =  4 

	- Step 6:
		Original pair (ci, ci+1): ( 4 ,  2 )
		Transformation process:
			Testing p ∈ [1, 2] with same parity as ci=4 (even):
				p=1 is odd, so parity is NOK
				p=2 is even, so parity OK, Then, the formula m = (ci - p) ÷ (2 · q) = (4 - 2) ÷ (2) = 1, f(2)=1, so OK
		Tuple-based transform: [p=2, f(p)=1, m=1, q=1]
		Reconstruction process:
			ci   = 2 · q · m + p    = 2 · 1 · 1 + 2 = 2 + 2 =  4 
			ci+1 =     q · m + f(p) = 1 · 1 + 1     = 1 + 1 =  2 

	- Step 7:
		Original pair (ci, ci+1): ( 2 ,  1 )
		Transformation process:
			Testing p ∈ [1, 2] with same parity as ci=2 (even):
				p=1 is odd, so parity is NOK
				p=2 is even, so parity OK, Then, the formula m = (ci - p) ÷ (2 · q) = (2 - 2) ÷ (2) = 0, f(2)=1, so OK
		Tuple-based transform: [p=2, f(p)=1, m=0, q=1]
		Reconstruction process:
			ci   = 2 · q · m + p    = 2 · 1 · 0 + 2 = 0 + 2 =  2 
			ci+1 =     q · m + f(p) = 1 · 0 + 1     = 0 + 1 =  1 

[*] p-PARAMETERS SEQUENCE:

	[1, 2, 1, 2, 2, 2, 2]

[*] m-PARAMETERS SEQUENCE:

	[1, 4, 2, 7, 3, 1, 0]
```

## Contributing

Contributions are welcome! Please follow these guidelines:

**Code Contributions:**

- Maintain mathematical accuracy against the original article
- Preserve hash table integrity and parallel processing efficiency
- Follow existing documentation standards and code style

**Research Contributions:**

- Validate theoretical changes against sequence equivalence tests
- Provide mathematical proofs or references for algorithmic modifications
- Include performance benchmarks for optimization claims

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tuple_based_transform_calculator,
  title={Tuple-based Transform Calculator},
  author={Javier Hernandez},
  year={2025},
  url={https://github.com/hhvvjj/tuple-based-transform-calculator},
  note={Implementation based on research DOI: 10.5281/zenodo.15546925}
}
```

## Files

- `tuple_based_transform_calculator.py` - Main implementation
- `README.md` - This documentation
- `LICENSE` - License file

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

**You are free to:**

- Share — copy and redistribute the material
- Adapt — remix, transform and build upon the material

**Under the following terms:**

- Attribution — You must give appropriate credit
- NonCommercial — You may not use the material for commercial purposes
- ShareAlike — If you remix, transform or build upon the material, you must distribute your contributions under the same license

See [LICENSE](https://github.com/hhvvjj/tuple-based-transform-calculator/blob/main/LICENSE) for full details.

## Contact

For questions about the algorithm implementation, mathematical details or optimization strategies, please contact via email (271314@pm.me).
