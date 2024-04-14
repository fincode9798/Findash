# third.py

# Importing functions from first.py
from first import generate_data, process_data

# Function to demonstrate data sharing and processing
def demo_data_usage():
    data = generate_data()
    processed_data = process_data(data)
    print("Generated data:", data)
    print("Processed data:", processed_data)

# Calling the function to demonstrate usage
if __name__ == "__main__":
    demo_data_usage()

