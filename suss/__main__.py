# Standard library
import asyncio

# Local
try:
    from suss.suss import main
except ImportError:
    from suss import main

if __name__ == "__main__":
    main()
