from model import *

df = None # Load the data
model_save_path = None # Load the model
ctr_model = load_trained_model(device, model_save_path)

image_dir = None # Path to the image directory
output_dir = None # Path to the output directory
new_df_rows = []


for i in tqdm(range(len(df))): 

    image_path = [os.path.join(image_dir, df['image_sign'][i] + f'_{j}.png') for j in range(10)]
    # Check if all image paths exist
    if not all(os.path.exists(path) for path in image_path):
        print("Error!")
        continue  # Skip this iteration if any image path does not exist
    
    item_name = [df['item_name'][i] for _ in range(10)]
    item_category = [df['item_category'][i] for _ in range(10)]
    shop_name = [df['shop_name'][i] for _ in range(10)]
    shop_main_category_name = [df['shop_main_category_name'][i] for _ in range(10)]
    bu_flag = [df['bu_flag'][i] for _ in range(10)]
    city_name = [df['city_name'][i] for _ in range(10)]

    batch = [item_name, item_category, shop_name, shop_main_category_name, bu_flag, city_name, image_path]

    with torch.no_grad():
        predictions = ctr_model(batch)

    # Convert predictions to a list of tuples with index
    predictions_with_index = list(enumerate(predictions.squeeze().tolist()))

    # Find max for top 3, 5, and 10 predictions
    top_3_idx = max(predictions_with_index[:3], key=lambda x: x[1])[0]
    top_5_idx = max(predictions_with_index[:5], key=lambda x: x[1])[0]
    top_10_idx = max(predictions_with_index[:10], key=lambda x: x[1])[0]

    # Add the original row data with new columns top_k and oss_path to new_df for each entry
    original_data = df.iloc[i].to_dict()
    
    # Create new rows for new_df based on top_k and image_sign
    new_df_rows.append({**original_data, 'top_k': 1, 'path': f"{output_dir}/{df['image_sign'][i]}_0.png"})
    
    # Add rows for top_k = 3, 5, and 10
    new_df_rows.append({**original_data, 'top_k': 3, 'path': f"{output_dir}/{df['image_sign'][i]}_{top_3_idx}.png"})
    new_df_rows.append({**original_data, 'top_k': 5, 'path': f"{output_dir}/{df['image_sign'][i]}_{top_5_idx}.png"})
    new_df_rows.append({**original_data, 'top_k': 10, 'path': f"{output_dir}/{df['image_sign'][i]}_{top_10_idx}.png"})

# Convert new_df_rows to a DataFrame
new_df = pd.DataFrame(new_df_rows)
new_df.to_excel('ctr_selection.xlsx', index=False)
