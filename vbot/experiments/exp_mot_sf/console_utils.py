from termcolor import colored, cprint


def bf(text):
    """returns bold text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, attrs=['bold'])


def rb(text):
    """returns bold red text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'red', attrs=['bold'])


def mb(text):
    """returns bold magenta text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'magenta', attrs=['bold'])


def gb(text):
    """returns bold green text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'green', attrs=['bold'])


def yb(text):
    """returns bold yellow text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'yellow', attrs=['bold'])


def bb(text):
    """returns bold blue text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'blue', attrs=['bold'])


def cb(text):
    """returns bold cyan text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'cyan', attrs=['bold'])


def r(text):
    """returns red text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'red')


def m(text):
    """returns magenta text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'magenta')


def g(text):
    """returns green text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'green')


def y(text):
    """returns yellow text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'yellow', attrs=['dark'])


def b(text):
    """returns blue text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'blue')


def c(text):
    """returns cyan text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, 'cyan')