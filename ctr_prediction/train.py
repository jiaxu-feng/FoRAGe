from model import *

def main(new_cutoff, trained_folder, existing_epoch, lr):

    final_data = None # Load the final data here

    # Configuration and Initialization
    batch_size = 16
    hidden_dim = (1024, 256)
    num_head = 4
    num_tranformer_layers = 4
    epochs = 100
    seed = 42
    
    # Define the base directory for saving all models
    base_dir = f"SGD_cutoff{new_cutoff}_batch{batch_size}_fc{hidden_dim[0]}-{hidden_dim[1]}_numhead{num_head}_numlayer_{num_tranformer_layers}_lr{lr}"

    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory at: {base_dir}")

    if trained_folder == None:

        # Initialize the model
        ctr_model = CTRPredictionModel(
            embedding_dim = 768, 
            num_embeddings = 7, 
            hidden_dim = hidden_dim, 
            num_head = num_head, 
            num_tranformer_layers = num_tranformer_layers, 
            device = device)
        ctr_model.to(device)

        address = os.path.join(base_dir, "data_split_indices.pkl")
        train_dataloader, valid_dataloader, test_dataloader = get_dataloader(final_data, batch_size, address, seed)

    else:

        model_save_path = os.path.join(trained_folder, f"ctr_model_epoch_{existing_epoch}.pth")
        ctr_model = load_trained_model(device, model_save_path)
        split_address = os.path.join(trained_folder, 'data_split_indices.pkl')
        train_dataloader, valid_dataloader, test_dataloader = load_splits(final_data, batch_size, split_address)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ctr_model.parameters(), lr=lr)

    history_path = os.path.join(base_dir, "training_history.csv")

    # Ensure the CSV file is initialized with headers
    if not os.path.exists(history_path):
        with open(history_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'valid_loss'])
            writer.writeheader()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        model_save_path = os.path.join(base_dir, f"ctr_model_epoch_{epoch+1}.pth")
        
        avg_train_loss = train_model(model=ctr_model, train_dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, device=device, model_save_path=model_save_path)
        avg_valid_loss = eval_model(model=ctr_model, dataloader=valid_dataloader, criterion=criterion, device=device)

        # Record the current epoch history
        epoch_history = {'epoch': epoch + 1, 'train_loss': avg_train_loss, 'valid_loss': avg_valid_loss}
        
        # Save the current epoch history to the CSV file
        with open(history_path, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'valid_loss'])
            writer.writerow(epoch_history)

        print(f"Epoch {epoch + 1} history saved to {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse command-line arguments.')
    parser.add_argument('--cutoff', type=int, default=50, help='Exposure count cutoff for new data')
    parser.add_argument('--trained_folder', type=str, default=None, help='Folder containing trained models')
    parser.add_argument('--existing_epoch', type=int, default=0, help='Epochs of the train model to start from')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    args = parser.parse_args()
    main(args.cutoff, args.trained_folder, args.existing_epoch, args.lr)


