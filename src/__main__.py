#!/usr/bin/env python3
"""
Main entry point for running src_new as a module
Usage: python -m src_new [command] [args]
"""

import sys
from .cli import CLIManager

def main():
    """Main entry point when running as module."""
    try:
        cli = CLIManager()
        return cli.run(sys.argv[1:])
    except KeyboardInterrupt:
        print("\n⏹️ Операцію перервано користувачем")
        return 1
    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 