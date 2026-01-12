import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import json
import random
import math
from tqdm import tqdm
import time
from transformers import GPT2Tokenizer
from mixture_of_recursion import RecursiveLanguageModel, RecursiveLanguageModelConfig
class LanguageModelDataset(Dataset):
    def __init__(self, text_data, tokenizer, sequence_length=256):
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
    def __len__(self):
        return len(self.text_data)
    def __getitem__(self, index):
        text = self.text_data[index]['text']
        encoded_tokens = self.tokenizer.encode(
            text, 
            max_length=self.sequence_length, 
            truncation=True
        )   
        padding_length = self.sequence_length - len(encoded_tokens)
        input_sequence = encoded_tokens + [self.tokenizer.pad_token_id] * padding_length
        label_sequence = encoded_tokens + [-100] * padding_length    
        return {
            'input_ids': torch.tensor(input_sequence, dtype=torch.long),
            'labels': torch.tensor(label_sequence, dtype=torch.long)
        }
def evaluate_model(model, validation_dataloader, device):
    model.eval()
    total_validation_loss = 0.0
    total_valid_tokens = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)        
            model_outputs = model(input_ids, labels=labels)
            batch_loss = model_outputs.loss        
            if not torch.isnan(batch_loss):
                valid_token_count = (labels != -100).sum().item()
                total_validation_loss += batch_loss.item() * valid_token_count
                total_valid_tokens += valid_token_count
    average_loss = total_validation_loss / total_valid_tokens if total_valid_tokens > 0 else 0
    perplexity = math.exp(min(average_loss, 20))
    return average_loss, perplexity
def train_model():
    DATASET_PATH = "/content/gpt_cache (1).json"
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 8
    LEARNING_RATE = 5e-4
    SEQUENCE_LENGTH = 256
    with open(DATASET_PATH, 'r') as file:
        dataset = json.load(file)
    random.shuffle(dataset)
    train_test_split_index = int(len(dataset) * 0.95)
    training_data = dataset[:train_test_split_index]
    validation_data = dataset[train_test_split_index:]
    print(f"Training samples: {len(training_data)}, Validation samples: {len(validation_data)}")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained("tokenizer")
    print(f"Vocabulary size: {len(tokenizer)}")
    training_dataset = LanguageModelDataset(training_data, tokenizer, SEQUENCE_LENGTH)
    validation_dataset = LanguageModelDataset(validation_data, tokenizer, SEQUENCE_LENGTH)
    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=2
    )
    model_config = RecursiveLanguageModelConfig(
        vocab_size=len(tokenizer),
        embedding_dim=512,
        num_layers=6,
        num_attention_heads=8,
        max_recursion_steps=2,
        dropout_rate=0.1,
        max_position_embeddings=SEQUENCE_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )
    device = torch.device("cuda")
    model = RecursiveLanguageModel(model_config).to(device)
    total_parameters = sum(param.numel() for param in model.parameters())
    print(f"Total model parameters: {total_parameters:,}")
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    gradient_scaler = GradScaler('cuda')
    total_training_steps = (len(training_dataloader) * NUM_EPOCHS) // GRADIENT_ACCUMULATION_STEPS
    learning_rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        total_steps=total_training_steps
    )
    best_validation_loss = float('inf')
    training_start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_total_loss = 0
        progress_bar = tqdm(training_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        optimizer.zero_grad()
        for batch_index, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)   
            with autocast('cuda'):
                model_outputs = model(input_ids, labels=labels)
                batch_loss = model_outputs.loss / GRADIENT_ACCUMULATION_STEPS   
            gradient_scaler.scale(batch_loss).backward()    
            if (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                learning_rate_scheduler.step()
                optimizer.zero_grad()    
            epoch_total_loss += batch_loss.item() * GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item() * GRADIENT_ACCUMULATION_STEPS:.3f}'
            })
        average_training_loss = epoch_total_loss / len(training_dataloader)
        validation_loss, validation_perplexity = evaluate_model(
            model, 
            validation_dataloader, 
            device
        )
        print(f"\nEpoch {epoch+1}: "
              f"Training Loss={average_training_loss:.4f}, "
              f"Validation Loss={validation_loss:.4f}, "
              f"Perplexity={validation_perplexity:.2f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), "best_model_checkpoint.pt")
        torch.cuda.empty_cache()
    model_save_directory = Path("trained_model")
    model_save_directory.mkdir(exist_ok=True)
    model.save_pretrained(model_save_directory, safe_serialization=True)
    model_config.save_pretrained(model_save_directory)
    import shutil
    shutil.copytree("tokenizer", model_save_directory / "tokenizer", dirs_exist_ok=True)
    training_duration_hours = (time.time() - training_start_time) / 3600
    print(f"Training completed in {training_duration_hours:.2f} hours")
    print(f"Best validation loss: {best_validation_loss:.4f}")
if __name__ == "__main__":
    train_model()
