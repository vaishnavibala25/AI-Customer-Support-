import subprocess
import sys


def main():
    print("=" * 60)
    print("AI CUSTOMER SUPPORT AGENT")
    print("=" * 60)
    print()
    print("Starting the complete support pipeline...")
    print()

    result = subprocess.run(
        [sys.executable, "generate_response.py"],
        text=True
    )

    if result.returncode != 0:
        print("\nThe support pipeline exited with an error.")
        print("Please check the message above for details.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
