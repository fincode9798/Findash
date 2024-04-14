# second.py

# Importing functions from first.py
from first import generate_data, process_data

# Function to use the generated data and process it further
def use_data():
    data = generate_data()
    processed_data = process_data(data)
    print("Processed data:", processed_data)

# Calling the function to demonstrate usage
if __name__ == "__main__":
    use_data()

