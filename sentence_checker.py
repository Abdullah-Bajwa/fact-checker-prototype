import os
import webbrowser
import sys
from pandas.io.clipboard import clipboard_get
from fact_checker_via_web import FactCheckerViaWeb

fact_checker = FactCheckerViaWeb()

if len(sys.argv) > 1:
    # If a command-line argument is provided, use it as text
    text = ' '.join(sys.argv[1:])
else:
    # If no argument is provided, get text from the clipboard
    text = clipboard_get()

text, rewrite = fact_checker.fact_check_sentence(text, None)
print("Incorrect:")
print(text)
print("Correct:")
print(rewrite)

