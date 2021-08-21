from termcolor import colored, cprint

def bf(text):
    """returns bold text to print on terminal

    Args:
        text (string): bolded text
    """
    return colored(text, attrs=['bold'])