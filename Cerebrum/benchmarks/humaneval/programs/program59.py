

def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """

    def largest_prime_factor(n: int):
        largest_factor = 1
        divisor = 2
        while n > 1:
            if n % divisor == 0:
                largest_factor = divisor
                n //= divisor
            else:
                divisor += 1
        return largest_factor




METADATA = {}


def check(candidate):
    assert candidate(15) == 5
    assert candidate(27) == 3
    assert candidate(63) == 7
    assert candidate(330) == 11
    assert candidate(13195) == 29


check(largest_prime_factor)