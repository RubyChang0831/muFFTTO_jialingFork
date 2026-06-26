# experiments/run_jialing1105.py

from muFFTTO.jialing1105 import jialing1105_add

def main():
    a = 7
    b = 8
    result = jialing1105_add(a, b)

    print("=== jialing1105 experiment ===")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"result = {result}")

if __name__ == "__main__":
    main()