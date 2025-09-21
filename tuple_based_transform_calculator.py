# ***********************************************************************************
# TUPLE-BASED TRANSFORM CALCULATOR - COLLATZ SEQUENCE ANALYSIS
# ***********************************************************************************
#
# Author: Javier Hernandez
#
# Email: 271314@pm.me
#
# Description:
# The tuple-based transform is a reversible procedure to represent Collatz sequences
# using the tuple [p, f(p), m, q]. This methodology converts each consecutive pair
# (ci, ci+1) from a Collatz sequence into a unique 4-element representation that
# allows perfect reconstruction of the original pair.
#
# Key Features:
# - Bijective transformation: Each consecutive pair maps to exactly one tuple
# - Complete reversibility: Original pairs can be perfectly reconstructed
# - Pattern analysis: Extracts p-parameter and m-parameter sequences
# - Mathematical insight: Provides alternative view of Collatz sequence structure
#
# The transformation uses reconstruction formulas:
#   ci   = 2 · q · m + p
#   ci+1 =     q · m + f(p)   [if p even]  
#   ci+1 = 6 · q · m + f(p)   [if p odd]
#
# This approach may reveal patterns and structural properties not visible in
# the original Collatz sequences, potentially contributing to research on the
# Collatz Conjecture.
#
# Usage:
# python3 tuple_based_transform_calculator <n> [q]
#
# Example:
# python3 tuple_based_transform_calculator 25
#
# Output:
# - Console display of the complete Collatz sequence
# - Detailed tuple-based transform for each consecutive pair
# - Step-by-step transformation and reconstruction process
# - Final p-parameters sequence
# - Final m-parameters sequence
#
# License:
# CC-BY-NC-SA 4.0 International
# For additional details, visit:
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# For full details, visit
# https://github.com/hhvvjj/tuple-based-transform-calculator/blob/main/LICENSE
#
# Research Reference:
# Based on the tuple-based transform methodology described in:
# https://doi.org/10.5281/zenodo.15546925
#
# ***********************************************************************************

# ***********************************************************************************
# * 1. STANDARD LIBRARY IMPORTS
# ***********************************************************************************

from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys

# ***********************************************************************************
# * 2. CONFIGURATION AND DATA STRUCTURES
# ***********************************************************************************

class Config:
    DEFAULT_Q = 1
    # Safety limit based on empirical observations of Collatz sequences:
    # Most sequences reach 1 within ~500 steps, 10000 provides generous buffer
    # while preventing potential infinite loops if conjecture fails
    MAX_SEQUENCE_LENGTH = 10000  
    COLORS = {
        'GREEN_BG': '\033[42m',
        'RED': '\033[31m', 
        'GREEN': '\033[32m',
        'RESET': '\033[0m'
    }

@dataclass
class TupleTransform:
    """
    Represents a tuple-based transform for a consecutive pair in Collatz sequence.
    
    A tuple transform converts a pair (ci, ci+1) from a Collatz sequence into
    a 4-element representation [p, f(p), m, q] that can be used to reconstruct
    the original pair.
    
    Attributes:
        original_pair: The original (ci, ci+1) pair from Collatz sequence
        p: Parameter value in range [1, 2q] with same parity as ci
        f_p: Result of applying Collatz function to p
        m: Non-negative quotient parameter calculated from ci, p, and q
        q: Transform parameter (usually 1)
    """
    original_pair: Tuple[int, int]
    p: int
    f_p: int
    m: int
    q: int
    
    @property
    def tuple_representation(self) -> List[int]:
        """Returns the 4-element tuple representation [p, f(p), m, q]"""
        return [self.p, self.f_p, self.m, self.q]

# ***********************************************************************************
# * 3. CORE MATHEMATICAL FUNCTIONS
# ***********************************************************************************

def collatz_function(n: int) -> int:
    """
    Applies the standard Collatz function to a number.
    
    The Collatz function is defined as:
    - If n is even: return n/2
    - If n is odd: return 3n+1
    
    Args:
        n: Input positive integer
        
    Returns:
        Result of applying Collatz function to n
        
    Example:
        >>> collatz_function(6)   # even
        3
        >>> collatz_function(7)   # odd  
        22
    
    Time Complexity: O(1) - constant time arithmetic operations  
    Space Complexity: O(1) - no additional space required
    """
    return n // 2 if n % 2 == 0 else 3 * n + 1


def generate_collatz_sequence(n: int) -> List[int]:
    """
    Generates the complete Collatz sequence starting from n until reaching 1.
    
    Starting with n, repeatedly applies the Collatz function:
    - If current is even: next = current / 2
    - If current is odd:  next = 3 * current + 1
    - Stop when reaching 1
    
    The Collatz Conjecture states that this sequence always terminates at 1
    for any positive integer, though this remains unproven. This implementation
    includes a safety limit to prevent infinite loops.
    
    Args:
        n: Starting positive integer
        
    Returns:
        Complete Collatz sequence as list, ending with 1
        
    Raises:
        RuntimeError: If sequence exceeds maximum length (potential infinite loop)
        
    Example:
        >>> generate_collatz_sequence(7)
        [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        
    Time Complexity: O(k) where k is the number of steps to reach 1
    Space Complexity: O(k) for storing the sequence
    """
    sequence = [n]
    current = n
    
    while current != 1 and len(sequence) < Config.MAX_SEQUENCE_LENGTH:
        current = collatz_function(current)
        sequence.append(current)
    
    if len(sequence) >= Config.MAX_SEQUENCE_LENGTH:
        raise RuntimeError(f"Sequence exceeded maximum length of {Config.MAX_SEQUENCE_LENGTH}")
    
    return sequence

# ***********************************************************************************
# * 4. TUPLE TRANSFORMATION UTILITIES
# ***********************************************************************************

def _has_matching_parity(first_num: int, second_num: int) -> bool:
    """
    Checks if two numbers have the same parity (both even or both odd).
    
    This is a general utility function to verify parity matching between
    any two integers. It is used in the tuple transformation process to
    ensure that parameter p has the same parity as ci, which is a mathematical
    requirement for valid tuple transforms.
    
    Args:
        first_num: First integer to check
        second_num: Second integer to check
        
    Returns:
        True if both numbers have same parity (both even or both odd),
        False otherwise
        
    Examples:
        >>> _has_matching_parity(4, 6)  # both even
        True
        >>> _has_matching_parity(3, 7)  # both odd
        True
        >>> _has_matching_parity(4, 7)  # different parity
        False
        >>> _has_matching_parity(0, 2)  # both even (zero is even)
        True
        
    Time Complexity: O(1) - constant time modulo operations
    Space Complexity: O(1) - no additional space needed
    
    Note:
        This function treats zero as even, following standard mathematical convention.
    """
    return first_num % 2 == second_num % 2


def _calculate_transform_parameters(ci: int, ci_plus_1: int, q: int) -> Optional[Tuple[int, int, int]]:
    """
    Calculates the transformation parameters (p, f_p, m) for a consecutive pair.
    
    This function implements the core algorithm for finding the unique tuple
    representation of a consecutive pair from a Collatz sequence. It performs
    an exhaustive linear search through all possible values of parameter p
    in the range [1, 2q] to find the one that satisfies all transformation
    constraints.
    
    The algorithm follows these validation steps for each candidate p:
    1. Verify p has the same parity as ci (both even or both odd)
    2. Check that m = (ci - p) / (2q) yields a non-negative integer
    3. Apply the appropriate reconstruction formula based on p's parity
    4. Verify the formula produces the correct ci+1 value
    
    Reconstruction formulas used:
    - For even p: ci+1 = q·m + f(p)
    - For odd p:  ci+1 = 6·q·m + f(p)
    
    Args:
        ci: First element of consecutive pair from Collatz sequence
        ci_plus_1: Second element of consecutive pair
        q: Odd positive parameter for transform (typically q=1)
        
    Returns:
        Tuple (p, f_p, m) if valid parameters found, None if no valid
        transformation exists (which should not occur for legitimate
        consecutive pairs from Collatz sequences)
        Where:
        - p: Parameter in range [1, 2q] with same parity as ci
        - f_p: Result of applying Collatz function to p
        - m: Non-negative quotient parameter m = (ci - p) / (2q)
        
    Example:
        >>> _calculate_transform_parameters(14, 7, 1)
        (2, 1, 6)  # p=2, f(2)=1, m=(14-2)/(2·1)=6
        
        Verification:
        - Parity: p=2 (even), ci=14 (even) ✓
        - m calculation: (14-2)/(2·1) = 6 (non-negative integer) ✓
        - Reconstruction: ci+1 = 1·6 + 1 = 7 ✓ (using even p formula)
        
    Note:
        Returns None only for invalid input pairs that don't represent
        legitimate consecutive pairs from actual Collatz sequences.
        
    Time Complexity: O(q) - linear search through p ∈ [1, 2q]
    Space Complexity: O(1) - only uses constant additional space
    
    Note:
        This function guarantees to find a unique solution for any valid
        consecutive pair from a Collatz sequence, as proven by the bijective
        nature of the tuple-based transformation.
    """
    for p in range(1, 2*q + 1):
        # Check if p has same parity as ci (required for valid transform)
        if not _has_matching_parity(p, ci):
            continue
            
        # Calculate m parameter - must be non-negative integer for valid transform
        if (ci - p) % (2 * q) != 0:
            continue
        m = (ci - p) // (2 * q)
        
        # m must be non-negative
        if m < 0:
            continue
        
        # Calculate f(p) and expected ci+1 using reconstruction formula
        f_p = collatz_function(p)
        expected_ci_plus_1 = q * m + f_p if p % 2 == 0 else 6 * q * m + f_p
        
        # Check if this p gives the correct ci+1
        if ci_plus_1 == expected_ci_plus_1:
            return (p, f_p, m)
    
    return None

# ***********************************************************************************
# * 5. MAIN TRANSFORMATION ALGORITHMS
# ***********************************************************************************

def get_tuple(ci: int, ci_plus_1: int, q: int) -> Optional[TupleTransform]:
    """
    Finds the unique tuple [p, f(p), m, q] representing consecutive pair (ci, ci+1).
    
    This function serves as the primary interface for obtaining the tuple-based
    transform of a consecutive pair from a Collatz sequence. It delegates the
    actual parameter calculation to _calculate_transform_parameters and wraps
    the result in a TupleTransform object for easier handling.
    
    The function is guaranteed to find a valid tuple for any legitimate consecutive
    pair from a Collatz sequence due to the bijective nature of the transformation.
    However, it may return None for invalid inputs or pairs that don't come from
    actual Collatz sequences.
    
    Args:
        ci: First element of consecutive pair from Collatz sequence
        ci_plus_1: Second element of consecutive pair  
        q: Odd positive parameter for transform (typically q=1)
        
    Returns:
        TupleTransform object containing:
        - original_pair: The input (ci, ci+1) pair
        - p: Found parameter value with same parity as ci
        - f_p: Collatz function applied to p
        - m: Calculated quotient parameter
        - q: Transform parameter (same as input)
        
        Returns None if no valid tuple exists. While this shouldn't happen for
        valid Collatz consecutive pairs due to the bijective nature of the
        transformation, it may occur for arbitrary input pairs.
        
    Raises:
        None: Returns None for invalid inputs rather than raising exceptions
        
    Example:
        >>> transform = get_tuple(14, 7, 1)
        >>> transform.tuple_representation
        [2, 1, 6, 1]  # [p, f(p), m, q]
        >>> transform.original_pair
        (14, 7)
        
    Time Complexity: O(q) - delegated to _calculate_transform_parameters
    Space Complexity: O(1) - creates single TupleTransform object
    
    Note:
        This function assumes inputs represent a valid consecutive pair from
        a Collatz sequence. For arbitrary integer pairs, the result may be None.
    """
    params = _calculate_transform_parameters(ci, ci_plus_1, q)
    if params:
        p, f_p, m = params
        return TupleTransform(
            original_pair=(ci, ci_plus_1),
            p=p,
            f_p=f_p,
            m=m,
            q=q
        )
    return None


def tuple_based_transform(n: int, q: int = Config.DEFAULT_Q) -> Tuple[List[int], List[TupleTransform], List[int], List[int]]:
    """
    Computes the complete tuple-based transform for a Collatz sequence.
    
    This function generates the full Collatz sequence starting from n, then
    converts each consecutive pair in the sequence to its tuple representation.
    The result provides four different views of the same mathematical object:
    1. The original Collatz sequence
    2. The tuple transforms for each consecutive pair
    3. The sequence of p-parameters extracted from the transforms
    4. The sequence of m-parameters extracted from the transforms
    
    The tuple-based representation can be useful for:
    - Analyzing patterns in Collatz sequences
    - Studying structural properties
    - Alternative computational approaches
    - Mathematical research into Collatz conjecture
    
    Args:
        n: Initial positive integer for Collatz sequence
        q: Odd positive parameter for transform (default=1)
        
    Returns:
        Tuple containing:
        - collatz_sequence: Original Collatz sequence as list of integers
        - tuple_transforms: List of TupleTransform objects for each consecutive pair
        - p_parameter_sequence: List of p values extracted from transforms
        - m_parameter_sequence: List of m values extracted from transforms
        
    Raises:
        RuntimeError: If Collatz sequence exceeds maximum length
        ValueError: If any consecutive pair fails to produce a valid tuple transform
        
    Example:
        >>> seq, transforms, p_seq, m_seq = tuple_based_transform(7, 1)
        >>> seq
        [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        >>> len(transforms)
        16  # One transform for each consecutive pair
        >>> transforms[0].tuple_representation
        [1, 4, 3, 1]  # Tuple for pair (7, 22)
        >>> p_seq[0]
        1  # p-parameter for first pair
        >>> m_seq[0] 
        3  # m-parameter for first pair

    Time Complexity: O(k×q) where k is sequence length, q is transform parameter  
    Space Complexity: O(k) for storing sequences and transform objects

    Note:
        This function validates that all consecutive pairs produce valid
        tuple transforms, ensuring mathematical consistency.
    """
    # Generate complete Collatz sequence
    collatz_seq = generate_collatz_sequence(n)
    
    # Generate tuple transforms for consecutive pairs
    tuple_transforms = []
    p_sequence = []
    m_sequence = []
    
    for i in range(len(collatz_seq) - 1):
        ci = collatz_seq[i]
        ci_plus_1 = collatz_seq[i + 1]
        
        # Get unique tuple representation for this consecutive pair
        transform = get_tuple(ci, ci_plus_1, q)
        if transform is None:
            raise ValueError(f"Failed to find valid tuple transform for pair ({ci}, {ci_plus_1}) at position {i}")
        
        tuple_transforms.append(transform)
        p_sequence.append(transform.p)
        m_sequence.append(transform.m)
    
    return collatz_seq, tuple_transforms, p_sequence, m_sequence

# ***********************************************************************************
# * 6. DISPLAY AND OUTPUT FUNCTIONS
# ***********************************************************************************

def display_banner() -> None:
    """
    Displays the program identification banner to stdout.
    
    Prints a formatted header with asterisk borders to clearly identify
    the program when run from command line. The banner includes the
    program title and serves as visual separation from other console output.
    
    Output format:
        **************************************************************************
        * Tuple-based transform calculator                                       *
        **************************************************************************
        
    Args:
        None
        
    Returns:
        None: Prints 74-character banner with asterisk borders to stdout
        
    Example:
        >>> display_banner()
        # Prints the formatted banner to console
    """
    print("")
    print("*" * 74)
    print("* Tuple-based transform calculator                                       *")
    print("*" * 74)
    print("")

def display_algorithm_setup(n: int, q: int) -> None:
    """
    Displays the algorithm setup parameters before computation begins.
    
    This function shows the input parameters that will be used for the
    tuple-based transform calculation, providing clear visibility of
    the configuration before processing starts. It indicates whether
    the q parameter was explicitly provided or uses the default value.
    
    Args:
        n: Initial positive integer for Collatz sequence
        q: Odd positive parameter for transform
        
    Returns:
        None: Prints algorithm setup information to stdout
        
    Output format:
        [*] ALGORITHM SETUP:
        
            Initial values: n = X and q = Y [provided]
    
    Example:
        >>> display_algorithm_setup(27, 1)
        [*] ALGORITHM SETUP:
        
            Initial values: n = 27 and q = 1 [default]

            
        >>> display_algorithm_setup(25, 3)
        [*] ALGORITHM SETUP:
        
            Initial values: n = 25 and q = 13 [provided]
    """
    print("[*] ALGORITHM SETUP:")
    print("")

    # Indicate if q is default or explicitly provided
    if q == Config.DEFAULT_Q:
        print(f"\tInitial values: n = {n} and q = {q} [default]")
    else:
        print(f"\tInitial values: n = {n} and q = {q} [provided]")

    print("")


def display_collatz_sequence(sequence: List[int]) -> None:
    """
    Displays the complete Collatz sequence in a readable list format.
    
    Prints the entire Collatz sequence as a Python list representation,
    with a clear section header. The sequence is displayed on a single
    line for easy reading and copying.
    
    Output format:
        [*] COLLATZ SEQUENCE:
        
          [n1, n2, n3, ..., 1]
    
    Args:
        sequence: List of integers representing the complete Collatz sequence
                 from starting value to 1
        
    Returns:
        None: Prints sequence header and Python list representation to stdout
        
    Example:
        >>> display_collatz_sequence([7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1])
        [*] COLLATZ SEQUENCE:
        
          [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
    """
    print("[*] COLLATZ SEQUENCE:")
    print("")
    print(f"  {sequence}")
    print("")


def display_tuple_transforms(transforms: List[TupleTransform]) -> None:
    """
    Displays all tuple transforms with detailed step-by-step analysis.
    
    For each consecutive pair in the Collatz sequence, this function shows:
    - The original pair values with color highlighting
    - The complete transformation process (testing each candidate p)
    - The final tuple representation [p, f(p), m, q]
    - The reconstruction verification using inverse formulas
    
    The output uses ANSI color codes for visual enhancement:
    - Green background: Successful values and results
    - Green text: Successful p candidates
    - Red text: Failed p candidates
    
    Output format for each transform:
        - Step N:
            Original pair (ci, ci+1): (value1, value2)
            Transformation process:
                Testing p ∈ [1, 2q] with same parity as ci=value1 (even/odd):
                    p=1 is odd/even, so parity OK/NOK, [further validation...]
                    ...
            Tuple-based transform: [p=X, f(p)=Y, m=Z, q=W]
            Reconstruction process:
                ci = 2 · q · m + p = calculation = result
                ci+1 = formula = calculation = result
    
    Args:
        transforms: List of TupleTransform objects containing transformation
                   data for each consecutive pair
        
    Returns:
        None: Prints multi-line transformation analysis with ANSI color formatting to stdout
        
    Example:
        >>> transforms = [TupleTransform((7, 22), 1, 4, 3, 1)]
        >>> display_tuple_transforms(transforms)
        # Prints detailed transformation analysis for the pair (7, 22)
        
    Note:
        This function produces extensive output proportional to the number
        of consecutive pairs and the value of q parameter.
    """
    print("[*] TUPLE-BASED TRANSFORM AND RECONSTRUCTION:")
    print("")
    
    for i, transform in enumerate(transforms):
        _display_single_transform(i + 1, transform)


def _display_single_transform(step_num: int, transform: TupleTransform) -> None:
    """
    Displays comprehensive analysis for a single tuple transform.
    
    This function handles the detailed formatting for one consecutive pair
    transformation, showing the complete process from initial pair to final
    tuple representation and back to reconstructed pair. It coordinates
    multiple sub-display functions to create a structured, readable output.
    
    The display includes:
    1. Step number and original pair (with color highlighting)
    2. Transformation process (testing all p candidates)
    3. Final tuple representation
    4. Reconstruction verification
    
    Args:
        step_num: Sequential step number for this transformation (1-based)
        transform: TupleTransform object containing all transformation data
                  including original pair, parameters, and calculations
        
    Returns:
        None: Prints comprehensive transformation analysis to stdout
        
    Format:
        - Step N:
            [Transformation details via helper functions]
        
        [Blank line for separation]
    
    Example:
        >>> transform = TupleTransform((14, 7), 2, 1, 6, 1)
        >>> _display_single_transform(1, transform)
        # Prints complete analysis for pair (14, 7) → [2, 1, 6, 1]
        
    Note:
        This function assumes transform object contains valid data and
        does not perform additional validation.
    """
    ci, ci_plus_1 = transform.original_pair
    
    print(f"\t- Step {step_num}:")
    print(f"\t\tOriginal pair (ci, ci+1): ({Config.COLORS['GREEN_BG']} {ci} {Config.COLORS['RESET']}, "
          f"{Config.COLORS['GREEN_BG']} {ci_plus_1} {Config.COLORS['RESET']})")
    
    # Show transformation process
    _display_transformation_process(ci, ci_plus_1, transform.p, transform.q)
    
    # Show reconstruction
    _display_reconstruction_process(transform)
    
    print("")  # Blank line between steps


def _display_transformation_process(ci: int, ci_plus_1: int, winning_p: int, q: int) -> None:
    """
    Displays the detailed p-parameter search process with validation steps.
    
    This function shows how the algorithm systematically tests each possible
    value of p in the range [1, 2q] to find the one that satisfies all
    transformation constraints. For each candidate p, it displays:
    - Parity check (must match ci)
    - m calculation and integer validation
    - Non-negative constraint check
    - Final reconstruction formula verification
    
    The output uses color coding to distinguish between successful (green)
    and failed (red) candidates, making it easy to identify the winning
    parameter visually.
    
    Args:
        ci: First element of the consecutive pair being transformed
        ci_plus_1: Second element of the consecutive pair
        winning_p: The p parameter that successfully transforms this pair
                  (used for color highlighting)
        q: Transform parameter determining the search range [1, 2q]
        
    Returns:
        None: Prints p-parameter search analysis with color-coded validation to stdout
        
    Time Complexity: O(q) - tests each p in range [1, 2q]
        
    Output format:
        Transformation process:
            Testing p ∈ [1, 2q] with same parity as ci=value (even/odd):
                p=1 is odd, so parity NOK
                p=2 is even, so parity OK. Then, the formula m=... 
                ...
        Tuple-based transform: [p=X, f(p)=Y, m=Z, q=W]
    
    Example:
        >>> _display_transformation_process(14, 7, 2, 1)
        # Shows testing p=1 (fails parity), p=2 (succeeds), for pair (14,7)
        
    Note:
        This function uses precomputed validation results to avoid redundant
        calculations performed during the actual transformation.
    """
    parity = "even" if ci % 2 == 0 else "odd"
    print("\t\tTransformation process:")
    print(f"\t\t\tTesting p ∈ [1, {2*q}] with same parity as ci={ci} ({parity}):")
    
    # Precompute validation results to avoid redundant calculations
    validation_results = []
    for test_p in range(1, 2*q + 1):
        result = _compute_p_validation(test_p, ci, ci_plus_1, q)
        validation_results.append((test_p, result))
    
    # Display results using precomputed data
    for test_p, result in validation_results:
        _display_p_validation_result(test_p, result, winning_p)
    
    print(f"\t\tTuple-based transform: [p={winning_p}, f(p)={collatz_function(winning_p)}, "
          f"m={(ci - winning_p) // (2 * q)}, q={q}]")


def _compute_p_validation(test_p: int, ci: int, ci_plus_1: int, q: int) -> dict:
    """
    Computes validation results for a p candidate without side effects.
    
    This helper function performs all validation calculations for a single
    p candidate and returns the results as a structured dictionary. This
    avoids redundant calculations between the core algorithm and display functions.
    
    Args:
        test_p: The p value being validated
        ci: First element of consecutive pair
        ci_plus_1: Second element of consecutive pair
        q: Transform parameter
        
    Returns:
        Dictionary containing validation results:
        - parity_match: bool indicating if p has same parity as ci
        - parity_str: string description ("even" or "odd")
        - divisible: bool indicating if (ci-p) is divisible by 2q
        - m: calculated m value (if divisible), None otherwise
        - m_non_negative: bool indicating if m >= 0 (if m exists)
        - f_p: result of collatz_function(test_p) (computed once)
        - formula_match: bool indicating if reconstruction formula works
        - expected_ci_plus_1: calculated ci+1 value (if all validations pass)
        - ci: original ci value (for display purposes)
        - q: original q value (for display purposes)
    """
    # Compute f(p) once and reuse
    f_p = collatz_function(test_p)
    
    result = {
        'parity_match': _has_matching_parity(test_p, ci),
        'parity_str': "even" if test_p % 2 == 0 else "odd",
        'divisible': False,
        'm': None,
        'm_non_negative': False,
        'f_p': f_p,  # Store precomputed f(p)
        'formula_match': False,
        'expected_ci_plus_1': None,
        'ci': ci,  # Store for display
        'q': q     # Store for display
    }
    
    if not result['parity_match']:
        return result
    
    if (ci - test_p) % (2 * q) == 0:
        result['divisible'] = True
        result['m'] = (ci - test_p) // (2 * q)
        result['m_non_negative'] = result['m'] >= 0
        
        if result['m_non_negative']:
            # Use precomputed f_p instead of recalculating
            expected = q * result['m'] + f_p if test_p % 2 == 0 else 6 * q * result['m'] + f_p
            result['expected_ci_plus_1'] = expected
            result['formula_match'] = expected == ci_plus_1
    
    return result


def _display_p_validation_result(test_p: int, result: dict, winning_p: int) -> None:
    """
    Displays validation results for a single p candidate using precomputed data.
    
    This function formats and displays the validation results computed by
    _compute_p_validation, avoiding redundant calculations while providing
    the same detailed output as before. It uses precomputed f(p) value to
    avoid recalculating the Collatz function.
    
    Args:
        test_p: The p value being displayed
        result: Precomputed validation results from _compute_p_validation
        winning_p: The p value that ultimately succeeds (for color coding)
        
    Returns:
        None: Prints validation results with color-coded success/failure indicators to stdout
    """
    color = Config.COLORS['GREEN'] if test_p == winning_p else Config.COLORS['RED']
    ci = result['ci']
    q = result['q']
    
    if not result['parity_match']:
        print(f"\t\t\t\t{color}p={test_p}{Config.COLORS['RESET']} is {result['parity_str']}, so parity is NOK")
        return
    
    if not result['divisible']:
        print(f"\t\t\t\t{color}p={test_p}{Config.COLORS['RESET']} is {result['parity_str']}, so parity OK. "
              f"Then, the formula m = (ci - p) ÷ (2 · q) = ({ci}-{test_p}) ÷ ({2*q}) is not integer, so NOK")
        return
    
    if not result['m_non_negative']:
        print(f"\t\t\t\t{color}p={test_p}{Config.COLORS['RESET']} is {result['parity_str']}, so parity OK. "
              f"Then, the formula m = (ci - p) ÷ (2 · q) = ({ci} - {test_p}) ÷ ({2*q}) = {result['m']} < 0, so NOK")
        return
    
    # Use precomputed f(p) instead of recalculating
    formula_status = "OK" if result['formula_match'] else "NOK"
    
    print(f"\t\t\t\t{color}p={test_p}{Config.COLORS['RESET']} is {result['parity_str']}, so parity OK, "
          f"Then, the formula m = (ci - p) ÷ (2 · q) = ({ci} - {test_p}) ÷ ({2*q}) = {result['m']}, f({test_p})={result['f_p']}, so {formula_status}")


def _display_reconstruction_process(transform: TupleTransform) -> None:
    """
    Displays the inverse transformation with aligned reconstruction formulas.
    
    This function demonstrates how to reconstruct the original consecutive pair
    from its tuple representation, proving the transformation is lossless and
    bijective. It shows the exact reconstruction formulas with proper alignment
    of the equals signs for enhanced readability and highlights the recovered
    values with green background color for visual verification.
    
    The function dynamically calculates the required padding to align both
    equals signs vertically, ensuring consistent formatting regardless of
    the magnitude of the values involved.
    
    Reconstruction formulas displayed:
    - ci = 2 · q · m + p = calculation = result (always)
    - ci+1 = q · m + f(p) = calculation = result (if p is even)
    - ci+1 = 6 · q · m + f(p) = calculation = result (if p is odd)
    
    Args:
        transform: TupleTransform object containing the tuple representation
                  and original pair for verification
        
    Returns:
        None: Prints aligned reconstruction formulas with highlighted results to stdout
        
    Output format:
        Reconstruction process:
            ci   = 2 · q · m + p     = calculation = result
            ci+1 = formula           = calculation = result
    
    Example:
        >>> transform = TupleTransform((14, 7), 2, 1, 6, 1)
        >>> _display_reconstruction_process(transform)
        # Shows aligned reconstruction formulas
        
    Note:
        The reconstruction always produces the exact original pair,
        demonstrating the bijective nature of the transformation.
    """
    ci, ci_plus_1 = transform.original_pair
    p, f_p, m, q = transform.p, transform.f_p, transform.m, transform.q
    
    print("\t\tReconstruction process:")
    
    # Build the formula strings
    ci_formula = f"ci   = 2 · q · m + p    = 2 · {q} · {m} + {p}"
    if p % 2 == 0:
        ci_plus_1_formula = f"ci+1 =     q · m + f(p) = {q} · {m} + {f_p}"
    else:
        ci_plus_1_formula = f"ci+1 = 6 · q · m + f(p) = 6 · {q} · {m} + {f_p}"
    
    # Build the calculation strings
    ci_calc = f"{2*q*m} + {p}"
    if p % 2 == 0:
        ci_plus_1_calc = f"{q*m} + {f_p}"
    else:
        ci_plus_1_calc = f"{6*q*m} + {f_p}"
    
    # Calculate widths for alignment
    max_formula_width = max(len(ci_formula), len(ci_plus_1_formula))
    max_calc_width = max(len(ci_calc), len(ci_plus_1_calc))
    
    # Print with alignment
    print(f"\t\t\t{ci_formula:<{max_formula_width}} = {ci_calc:<{max_calc_width}} = {Config.COLORS['GREEN_BG']} {ci} {Config.COLORS['RESET']}")
    print(f"\t\t\t{ci_plus_1_formula:<{max_formula_width}} = {ci_plus_1_calc:<{max_calc_width}} = {Config.COLORS['GREEN_BG']} {ci_plus_1} {Config.COLORS['RESET']}")


def display_p_sequence(p_sequence: List[int]) -> None:
    """
    Displays the sequence of p-parameters extracted from tuple transforms.
    
    The p-parameter sequence represents the first component of each tuple
    transform [p, f(p), m, q] for all consecutive pairs in the Collatz sequence.
    This sequence provides an alternative view of the original sequence and may
    reveal patterns not visible in the standard Collatz representation.
    
    Properties of the p-sequence:
    - Each p value is in range [1, 2q] for the given q parameter
    - Each p has the same parity as its corresponding ci value
    - The sequence length is one less than the original Collatz sequence length
    - Pattern analysis of p-values may provide insights into Collatz behavior
    
    Args:
        p_sequence: List of p parameter values extracted from all tuple-based
                   transforms, ordered by their position in the Collatz sequence
        
    Returns:
        None: Prints p-sequence with formatted header to stdout
        
    Output format:
        [*] p-PARAMETERS SEQUENCE:
        
            [p1, p2, p3, ..., pn]
    
    Example:
        >>> display_p_sequence([1, 2, 1, 2, 1, 2])
        [*] p-PARAMETERS SEQUENCE:
        
            [1, 2, 1, 2, 1, 2]
        
    Note:
        The p-sequence may exhibit interesting patterns that correlate
        with the structure of the original Collatz sequence.
    """
    print("[*] p-PARAMETERS SEQUENCE:")
    print("")
    print(f"\t{p_sequence}")
    print("")


def display_m_sequence(m_sequence: List[int]) -> None:
    """
    Displays the sequence of m-parameters extracted from tuple transforms.
    
    The m-parameter sequence represents the third component of each tuple
    transform [p, f(p), m, q] for all consecutive pairs in the Collatz sequence.
    The m-values are quotient parameters calculated as m = (ci - p) / (2q) and
    provide an alternative mathematical representation that may reveal hidden
    patterns in Collatz sequences.
    
    Properties of the m-sequence:
    - All m values are non-negative integers by construction
    - m represents how many "units" of 2q separate ci from its corresponding p
    - The sequence length equals the p-sequence length (one less than Collatz)
    - m-values may show periodic or structured behavior worth investigating
    
    Args:
        m_sequence: List of m parameter values extracted from all tuple-based
                   transforms, ordered by their position in the Collatz sequence
        
    Returns:
        None: Prints m-sequence with formatted header to stdout
        
    Output format:
        [*] m-PARAMETERS SEQUENCE:
        
            [m1, m2, m3, ..., mn]
    
    Example:
        >>> display_m_sequence([3, 6, 2, 8, 4, 1])
        [*] m-PARAMETERS SEQUENCE:
        
            [3, 6, 2, 8, 4, 1]
        
    Note:
        The m-sequence provides a normalized view of the "gaps" between
        consecutive Collatz values and their corresponding p-parameters,
        potentially revealing structural patterns in the sequence evolution.
    """
    print("[*] m-PARAMETERS SEQUENCE:")
    print("")
    print(f"\t{m_sequence}")
    print("")

# ***********************************************************************************
# * 7. COMMAND LINE AND PROGRAM CONTROL
# ***********************************************************************************

def parse_command_line_args(args: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    Parses and validates command line arguments for the program.
    
    This function processes command line arguments for the tuple-based transform
    calculator, performing comprehensive validation to ensure proper input.
    It can accept arguments directly (for testing) or read from sys.argv (for CLI usage).
    
    Expected argument format:
        [script_name, n] or [script_name, n, q]
    
    Where:
    - n: Initial positive integer for Collatz sequence (mandatory)
    - q: Odd positive parameter for transform (optional, defaults to 1)
    
    Validation performed:
    - Checks that at least one argument (n) is provided
    - Verifies arguments can be parsed as integers
    - Ensures n is positive (n ≥ 1)
    - Ensures q is odd and positive (q ≥ 1 and q % 2 == 1)
    
    Args:
        args: Optional list of command line arguments. If None, reads from sys.argv.
              Expected format: [script_name, n] or [script_name, n, q]
        
    Returns:
        Tuple of (n, q) after successful validation
        Where:
        - n: Validated positive integer for Collatz sequence start
        - q: Validated odd positive integer for transform parameter
        
    Raises:
        ValueError: With descriptive message for any of these conditions:
        - No arguments provided (empty args or only script name)
        - Arguments cannot be parsed as integers  
        - n is not positive (n < 1)
        - q is not odd positive (q < 1 or q % 2 == 0)
        
    Examples:
        >>> parse_command_line_args(['script.py', '27'])
        (27, 1)  # Uses default q=1
        
        >>> parse_command_line_args(['script.py', '25', '3'])
        (25, 3)  # Uses provided q=3
        
        >>> parse_command_line_args(['script.py'])
        ValueError: No arguments provided
        
        >>> parse_command_line_args(['script.py', '-5'])
        ValueError: Parameter n must be positive (given n=-5)
        
        >>> parse_command_line_args(['script.py', '10', '4'])
        ValueError: Parameter q must be odd positive (given q=4)
        
        >>> parse_command_line_args(['script.py', 'abc'])
        ValueError: Please provide valid integers for n and q
    
    Time Complexity: O(1) - constant time operations
    Space Complexity: O(1) - no additional space needed
    
    Note:
        When args=None, this function reads from sys.argv. For testing purposes,
        pass arguments directly to avoid dependency on command line state.
    """
    if args is None:
        args = sys.argv
        
    if len(args) < 2:
        raise ValueError("No arguments provided")
    
    try:
        n = int(args[1])
        q = int(args[2]) if len(args) > 2 else Config.DEFAULT_Q
    except (ValueError, IndexError):
        raise ValueError("Please provide valid integers for n and q")
    
    if n < 1:
        raise ValueError(f"Parameter n must be positive (given n={n})")
        
    if q < 1 or q % 2 == 0:
        raise ValueError(f"Parameter q must be odd positive (given q={q})")
    
    return n, q


def print_usage() -> None:
    """
    Prints comprehensive usage information and examples to stdout.
    
    This function displays formatted help text explaining how to run the
    program correctly, including parameter descriptions, usage syntax,
    and practical examples. It's called when the user provides invalid
    arguments or no arguments at all.
    
    Output includes:
    - Command syntax with required and optional parameters
    - Detailed parameter descriptions and constraints
    - Practical usage examples
    - Clear formatting for easy readability
    
    Args:
        None
        
    Returns:
        None: Prints formatted help text to stdout
        
    Output format:
        [*] USAGE:
            python3 tuple_based_transform_calculator.py <n> [q]
            n: Initial positive integer for Collatz sequence (mandatory)
            q: Odd positive parameter for transform (optional, default=1)
        
        [*] EXAMPLE:
            python3 tuple_based_transform_calculator.py 27 (to calculate n=27 and q=1)
    
    Example:
        >>> print_usage()
        # Prints complete usage information to console
        
    Note:
        This function is typically called in error handling scenarios
        when argument parsing fails.
    """
    print("[*] USAGE:")
    print("\tpython3 tuple_based_transform_calculator.py <n> [q]")
    print("\t\tn: Initial positive integer for Collatz sequence (mandatory)")
    print("\t\tq: Odd positive parameter for transform (optional, default=1)")
    print("")
    print("[*] EXAMPLE:")
    print("\tpython3 tuple_based_transform_calculator.py 27 (to calculate n=27 and q=1)")
    print("")


def main() -> None:
    """
    Main execution function coordinating the entire program workflow.
    
    This function serves as the primary entry point and orchestrates all
    program components in the correct sequence. It handles the complete
    execution flow from argument parsing through final output display,
    with comprehensive error handling for graceful failure modes.
    
    Execution workflow:
    1. Display program banner for identification
    2. Parse and validate command line arguments  
    3. Compute tuple-based transform for the given parameters
    4. Display all results in formatted sections:
       - Original Collatz sequence
       - Detailed tuple transforms with step-by-step analysis
       - Extracted p-parameter sequence
       - Extracted m-parameter sequence
    5. Handle any errors with appropriate user feedback
    
    Args:
        None: Operates on command line arguments via sys.argv
        
    Returns:
        None: Prints results to stdout and may exit with error codes
        
    Side Effects:
        - Reads from sys.argv (command line arguments)
        - Calls sys.exit(1) on error conditions
        
    Exit codes:
        - 0: Successful execution and normal termination
        - 1: Error in argument parsing or computation
        
    Error handling:
    - ValueError: Invalid command line arguments → shows usage or error message
    - RuntimeError: Computation errors (e.g., sequence too long) → shows error
    - Exception: Unexpected errors → shows error message
    - All errors result in clean exit with code 1
    
    Example successful run:
        $ python3 script.py 7
        # Displays banner, computes transforms for n=7, shows all results
        
    Example error cases:
        $ python3 script.py
        # Shows usage information, exits with code 1
        
        $ python3 script.py -5  
        # Shows "Parameter n must be positive" error, exits with code 1
        
    Note:
        This function provides the complete user interface for the program
        and should handle all possible execution scenarios gracefully.
    """
    # Display banner first
    display_banner()
    
    try:
        # Parse and validate command line arguments (no side effects)
        n, q = parse_command_line_args()
        
        # Display algorithm setup
        display_algorithm_setup(n, q)
        
        # Compute tuple-based transform
        collatz_seq, transforms, p_sequence, m_sequence = tuple_based_transform(n, q)
        
        # Display all results
        display_collatz_sequence(collatz_seq)
        display_tuple_transforms(transforms)
        display_p_sequence(p_sequence)
        display_m_sequence(m_sequence)
        
    except ValueError as e:
        # Handle argument parsing errors gracefully
        if "No arguments provided" in str(e):
            print_usage()
        else:
            print("[*] ERROR:")
            print(f"\t{e}")
        sys.exit(1)
    except RuntimeError as e:
        # Handle computation errors (e.g., sequence too long)
        print(f"[*] COMPUTATION ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        # Handle unexpected errors
        print(f"[*] UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()