import pandas as pd

# Define the data as a list of lists
data = [
    [128, 128, 256, 128, 256, 128, 128, 128, 512, 128, 128, 128, 768],           #1/1
    [128, 128, 768, 128, 768, 128, 128, 256, 512, 256, 128, 128, 768],           #1/5
    [128, 128, 768, 128, 768, 128, 128, 256, 768, 256, 768, 768, 'nc'],          #1/10
    [128, 128, 768, 128, 768, 128, 128, 512, 'nc', 256, 768, 768, 'nc'],         #1/15
    [128, 128, 768, 128, 'nc', 128, 128, 'nc', 'nc', 512, 768, 768, 'nc']        #1/20
]

# Create a DataFrame from the data
df = pd.DataFrame(data, 
                  index=["1/1", "1/5", "1/10", "1/15", "1/20"],
                  columns=[f"Layer {i+1}" for i in range(13)])

# Save the DataFrame to an Excel file
df.to_excel("layer_data.xlsx")

print("Data has been written to 'layer_data.xlsx'")
