

def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """

    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    fibonacci = [0, 1]
    prime_fibs = []
    while len(prime_fibs) < 5:
        fib_n = fibonacci[-1] + fibonacci[-2]
        fibonacci.append(fib_n)
        if is_prime(fib_n):
            prime_fibs.append(fib_n)
    
    def prime_fib(n: int):
        if n < 1:
            return 0
        count = 0
        index = 0
        while count < n:
            fib_n = fibonacci[index]
            if is_prime(fib_n):
                count += 1
                if count == n:
                    return fib_n
            index += 1




METADATA = {}


def check(candidate):
    assert candidate(1) == 2
    assert candidate(2) == 3
    assert candidate(3) == 5
    assert candidate(4) == 13
    assert candidate(5) == 89
    assert candidate(6) == 233
    assert candidate(7) == 1597
    assert candidate(8) == 28657
    assert candidate(9) == 514229
    assert candidate(10) == 433494437


check(prime_fib)