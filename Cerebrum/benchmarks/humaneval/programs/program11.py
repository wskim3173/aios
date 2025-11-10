from typing import List


def string_xor(a: str, b: str) -> str:
    """ Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """

    if len(a) != len(b):
        raise ValueError("Strings must be of the same length")
    
    result = []
    for bit_a, bit_b in zip(a, b):
        result_bit = '1' if bit_a != bit_b else '0'
        result.append(result_bit)
    
    return ''.join(result)



METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    assert candidate('111000', '101010') == '010010'
    assert candidate('1', '1') == '0'
    assert candidate('0101', '0000') == '0101'

check(string_xor)