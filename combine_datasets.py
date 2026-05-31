import pandas as pd

def load_dataset(path, label):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if 'user_input' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'user_input'})
    return pd.DataFrame({
        'user_input': df['user_input'].astype(str),
        'label': label,
    })

if __name__ == '__main__':
    datasets = [
        ('data.csv', 1),
        ('toxic_chat_dataset.csv', 0),
        ('banking_chatbot_adversarial_prompts.csv', 0),
    ]

    combined = pd.concat([load_dataset(path, label) for path, label in datasets], ignore_index=True)
    combined.to_csv('joined_dataset.csv', index=False)
    print(f'Joined {len(combined)} rows into joined_dataset.csv')
