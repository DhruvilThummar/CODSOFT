import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Define Text Dataset class for sequence generation
class TextDataset(Dataset):
    def __init__(self, text, seq_length, char_to_ix):
        self.seq_length = seq_length
        self.char_to_ix = char_to_ix
        
        # Prepare inputs and targets
        self.data_X = []
        self.data_y = []
        
        for i in range(0, len(text) - seq_length, 3): # Step of 3 to reduce dataset size and speed up training
            seq_in = text[i:i + seq_length]
            char_out = text[i + seq_length]
            
            self.data_X.append([char_to_ix[char] for char in seq_in])
            self.data_y.append(char_to_ix[char_out])
            
    def __len__(self):
        return len(self.data_X)
        
    def __getitem__(self, idx):
        return torch.tensor(self.data_X[idx]), torch.tensor(self.data_y[idx])

# Define the Char-LSTM Model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, state=None):
        out = self.embedding(x)
        out, state = self.lstm(out, state)
        out = self.fc(out[:, -1, :]) # Predict only the next character
        return out, state

def generate_text(model, seed_text, chars, char_to_ix, ix_to_char, gen_length=400, temperature=0.7):
    model.eval()
    vocab_size = len(chars)
    
    # Initialize hidden state
    state = None
    
    # Process seed text
    current_seq = [char_to_ix[c] for c in seed_text]
    generated_text = seed_text
    
    with torch.no_grad():
        for _ in range(gen_length):
            # Format input sequence
            x = torch.tensor([current_seq[-50:]]).long() # Keep sequence length max 50
            
            # Forward pass
            logits, state = model(x)
            
            # Temperature scaling
            logits = logits[0] / temperature
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            # Sample next character index
            next_char_idx = np.random.choice(vocab_size, p=probs)
            next_char = ix_to_char[next_char_idx]
            
            # Append to text
            generated_text += next_char
            current_seq.append(next_char_idx)
            
    return generated_text

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        # Check if line breaks are explicit
        if '\n' in word:
            sub_words = word.split('\n')
            for i, sw in enumerate(sub_words):
                if i > 0:
                    lines.append(' '.join(current_line))
                    current_line = [sw]
                else:
                    current_line.append(sw)
        else:
            current_line.append(word)
            test_line = ' '.join(current_line)
            # Use getlength if getsize is deprecated, or fallback
            try:
                line_width = font.getlength(test_line)
            except AttributeError:
                line_width = font.getbbox(test_line)[2]
                
            if line_width > max_width:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
                
    if current_line:
        lines.append(' '.join(current_line))
        
    return lines

def render_handwriting_image(text, font_path, output_path):
    print("Rendering handwritten text to image...")
    img_width, img_height = 900, 700
    
    # Create a light cream notebook paper background
    image = Image.new('RGB', (img_width, img_height), color='#fdfbf2')
    draw = ImageDraw.Draw(image)
    
    # Draw horizontal notebook lines (light blue lines)
    line_spacing = 30
    margin_top = 80
    margin_left = 60
    
    for y in range(margin_top, img_height - 40, line_spacing):
        draw.line([(margin_left - 10, y), (img_width - 40, y)], fill='#d0e3f0', width=1)
        
    # Draw red vertical margin line
    draw.line([(margin_left - 10, 0), (margin_left - 10, img_height)], fill='#f5a6b0', width=2)
    
    # Load Font
    try:
        # 24 pt handwriting style
        font = ImageFont.truetype(font_path, size=24)
    except Exception as e:
        print(f"Error loading TTF font: {e}, falling back to default.")
        font = ImageFont.load_default()
        
    # Wrap text to fit margin widths
    max_text_width = img_width - margin_left - 60
    wrapped_lines = wrap_text(text, font, max_text_width)
    
    # Write text onto lines
    y_text = margin_top - 6 # Offset slightly to sit on the lines
    for line in wrapped_lines:
        if y_text + line_spacing > img_height - 30:
            break # Avoid writing off the bottom of the page
        draw.text((margin_left, y_text), line, fill='#253457', font=font) # Classic blue pen color
        y_text += line_spacing
        
    # Save Image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Saved handwritten image to: {output_path}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transcripts_path = os.path.join(current_dir, "handwriting_transcripts.txt")
    font_path = os.path.join(current_dir, "Caveat-Regular.ttf")
    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load Text Dataset
    print("Loading transcripts...")
    with open(transcripts_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text)} characters.")
    
    # Vocabulary mappings
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Unique characters (Vocabulary size): {vocab_size}")
    
    char_to_ix = {c: i for i, c in enumerate(chars)}
    ix_to_char = {i: c for i, c in enumerate(chars)}
    
    # Hyperparameters
    seq_length = 50
    batch_size = 128
    embedding_dim = 128
    hidden_dim = 256
    epochs = 12
    learning_rate = 0.003
    
    # Datasets & Loader
    print("Preparing sequences...")
    dataset = TextDataset(text, seq_length, char_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Total sequences: {len(dataset)}")
    
    # Initialize Model, Loss, Optimizer
    model = CharLSTM(vocab_size, embedding_dim, hidden_dim, num_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n--- Training Character-Level RNN (LSTM) ---")
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(batch_X)
            loss = criterion(logits, batch_y)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        epoch_loss = total_loss / len(dataloader)
        loss_history.append(epoch_loss)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Time: {elapsed:.2f}s")
        
        # Output a sample text prediction at the end of every 3 epochs
        if epoch % 3 == 0:
            sample_seed = "ROMEO: "
            if all(c in char_to_ix for c in sample_seed):
                
                sample_gen = generate_text(model, sample_seed, chars, char_to_ix, ix_to_char, gen_length=100, temperature=0.7)
                print(f"--> Generated Sample: {repr(sample_gen)}")
                
    # Save training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='#d62728', linewidth=2)
    plt.title('Training Loss vs Epochs (Char-LSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_history.png'), dpi=300)
    plt.close()
    
    # Generate final narrative text
    print("\n--- Generating Handwritten Text ---")
    final_seed = "ROMEO:\nShall I speak "
    generated_text = generate_text(model, final_seed, chars, char_to_ix, ix_to_char, gen_length=500, temperature=0.6)
    print("\nGenerated Text:\n")
    print(generated_text)
    
    # Render final generated text to a handwritten image sheet
    output_image_path = os.path.join(plots_dir, "generated_handwriting.png")
    render_handwriting_image(generated_text, font_path, output_image_path)
    
    # Save Model & Mappings
    model_path = os.path.join(current_dir, "best_rnn_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'chars': chars,
        'char_to_ix': char_to_ix,
        'ix_to_char': ix_to_char,
        'hyperparams': {
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'seq_length': seq_length
        }
    }, model_path)
    print(f"\nSaved PyTorch model checkpoints to: {model_path}")
    
    # Save metrics summary
    metrics_path = os.path.join(current_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== Char-LSTM Model Details ===\n")
        f.write(f"Sequence Length: {seq_length}\n")
        f.write(f"Embedding Dim: {embedding_dim}\n")
        f.write(f"Hidden Dim: {hidden_dim}\n")
        f.write(f"Epochs trained: {epochs}\n")
        f.write(f"Final Cross-Entropy Loss: {loss_history[-1]:.4f}\n")
        f.write("\n=== Sample Generated Text ===\n")
        f.write(generated_text)
        f.write("\n")
    print(f"Results summary written to {metrics_path}")

if __name__ == "__main__":
    main()
